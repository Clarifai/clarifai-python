import asyncio
import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.auth.register import RpcCallable, V2Stub
from clarifai.utils.logging import logger

throttle_status_codes = {
    status_code_pb2.CONN_THROTTLED,
    status_code_pb2.CONN_EXCEED_HOURLY_LIMIT,
}

retry_codes_grpc = {
    grpc.StatusCode.UNAVAILABLE,
}

_threadpool = ThreadPoolExecutor(100)


def validate_response(response, attempt, max_attempts):
    # Helper function to handle simple response validation
    def handle_simple_response(response):
        if hasattr(response, 'status') and hasattr(response.status, 'code'):
            if (response.status.code in throttle_status_codes) and attempt < max_attempts:
                logger.debug('Retrying with status %s' % str(response.status))
                return None  # Indicates a retry is needed
            else:
                return response

        # Check if response is an async iterator
        if hasattr(response, '__aiter__'):
            return response  # Return async iterator directly for handling in _async_call__

    # Check if the response is an instance of a gRPC streaming call
    if isinstance(response, grpc._channel._MultiThreadedRendezvous):
        try:
            # Check just the first response in the stream for validation
            first_res = next(response)
            validated_response = handle_simple_response(first_res)
            if validated_response is not None:
                # Have to return that first response and the rest of the stream.
                return itertools.chain([validated_response], response)
            return None  # Indicates a retry is needed
        except grpc.RpcError as e:
            logger.error('Error processing streaming response: %s' % str(e))
            return None  # Indicates an error
    else:
        # Handle simple response validation
        return handle_simple_response(response)


def create_stub(
    auth_helper: ClarifaiAuthHelper = None, max_retry_attempts: int = 10, is_async: bool = False
) -> V2Stub:
    """
    Create client stub that handles authorization and basic retries for
    unavailable or throttled connections.

    Args:
      auth_helper:  ClarifaiAuthHelper to use for auth metadata (default: from env)
      max_retry_attempts:  max attempts to retry rpcs with retryable failures
    """
    stub = AuthorizedStub(auth_helper, is_async=is_async)
    if max_retry_attempts > 0:
        return RetryStub(stub, max_retry_attempts, is_async=is_async)
    return stub


class AuthorizedStub(V2Stub):
    """V2Stub proxy that inserts metadata authorization in rpc calls."""

    def __init__(self, auth_helper: ClarifaiAuthHelper = None, is_async: bool = False):
        if auth_helper is None:
            auth_helper = ClarifaiAuthHelper.from_env()
        self.is_async = is_async
        self.stub = auth_helper.get_async_stub() if is_async else auth_helper.get_stub()
        self.metadata = auth_helper.metadata

    def __getattr__(self, name):
        value = getattr(self.stub, name)
        if isinstance(value, RpcCallable):
            value = _AuthorizedRpcCallable(value, self.metadata, self.is_async)
        return value


class _AuthorizedRpcCallable(RpcCallable):
    """Adds metadata(authorization header) to rpc calls"""

    def __init__(self, func, metadata, is_async):
        self.f = func
        self.metadata = metadata
        self.is_async = is_async or asyncio.iscoroutinefunction(func)

    def __repr__(self):
        return repr(self.f)

    def __call__(self, *args, **kwargs):
        metadata = kwargs.pop('metadata', self.metadata)

        return self.f(
            *args,
            **kwargs,
            metadata=metadata,
        )

    def future(self, *args, **kwargs):
        metadata = kwargs.pop('metadata', self.metadata)
        return self.f.future(*args, **kwargs, metadata=metadata)

    def __getattr__(self, name):
        return getattr(self.f, name)


class RetryStub(V2Stub):
    """
    V2Stub proxy that retries requests (currently on unavailable server or throttle codes)
    """

    def __init__(self, stub, max_attempts=10, backoff_time=5, is_async=False):
        self.stub = stub
        self.max_attempts = max_attempts
        self.backoff_time = backoff_time
        self.is_async = is_async

    def __getattr__(self, name):
        value = getattr(self.stub, name)
        if isinstance(value, RpcCallable):
            value = _RetryRpcCallable(value, self.max_attempts, self.backoff_time, self.is_async)
        return value


class _RetryRpcCallable(RpcCallable):
    """Retries rpc calls on unavailable server or throttle codes"""

    def __init__(self, func, max_attempts, backoff_time, is_async=False):
        self.f = func
        self.max_attempts = max_attempts
        self.backoff_time = backoff_time
        self.is_async = is_async or asyncio.iscoroutinefunction(func)

    def __repr__(self):
        return repr(self.f)

    async def _async_call__(self, *args, **kwargs):
        """Handle async RPC calls with retries and validation"""
        for attempt in range(1, self.max_attempts + 1):
            if attempt != 1:
                await asyncio.sleep(self.backoff_time)

            try:
                response = self.f(*args, **kwargs)

                # Handle streaming response
                if hasattr(response, '__aiter__'):
                    return await self._handle_streaming_response(response, attempt)

                # Handle regular async response
                validated_response = await self._handle_regular_response(response, attempt)
                if validated_response is not None:
                    return validated_response

            except grpc.RpcError as e:
                if not self._should_retry(e, attempt):
                    raise
                logger.debug(
                    f'Retrying after error {e.code()} (attempt {attempt}/{self.max_attempts})'
                )

        raise Exception(f'Max attempts reached ({self.max_attempts}) without success')

    async def _handle_streaming_response(self, response, attempt):
        """Handle streaming response validation and processing"""

        async def validated_stream():
            try:
                async for item in response:
                    if not self._is_valid_response(item):
                        if attempt < self.max_attempts:
                            yield None  # Signal for retry
                        raise Exception(
                            f'Validation failed on streaming response (attempt {attempt})'
                        )
                    yield item
            except grpc.RpcError as e:
                if not self._should_retry(e, attempt):
                    raise
                yield None  # Signal for retry

        return validated_stream()

    async def _handle_regular_response(self, response, attempt):
        """Handle regular async response validation and processing"""
        try:
            result = await response
            if not self._is_valid_response(result):
                if attempt < self.max_attempts:
                    return None  # Signal for retry
                raise Exception(f'Validation failed on response (attempt {attempt})')
            return result
        except grpc.RpcError as e:
            if not self._should_retry(e, attempt):
                raise
            return None  # Signal for retry

    def _is_valid_response(self, response):
        """Check if response status is valid"""
        return not (
            hasattr(response, 'status')
            and hasattr(response.status, 'code')
            and response.status.code in throttle_status_codes
        )

    def _should_retry(self, error, attempt):
        """Determine if we should retry based on error and attempt count"""
        return (
            isinstance(error, grpc.RpcError)
            and error.code() in retry_codes_grpc
            and attempt < self.max_attempts
        )

    def _sync_call__(self, *args, **kwargs):
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            if attempt != 1:
                time.sleep(self.backoff_time)  # TODO better backoff between attempts
            try:
                response = self.f(*args, **kwargs)
                v = validate_response(response, attempt, self.max_attempts)
                if v is not None:
                    return v
            except grpc.RpcError as e:
                if (e.code() in retry_codes_grpc) and attempt < self.max_attempts:
                    logger.debug('Retrying with status %s' % e.code())
                else:
                    raise

    def __call__(self, *args, **kwargs):
        if self.is_async:
            return self._async_call__(*args, **kwargs)
        return self._sync_call__(*args, **kwargs)

    async def __call_async__(self, *args, **kwargs):
        """Explicit async call method"""
        return await self._async_call(*args, **kwargs)

    def future(self, *args, **kwargs):
        # TODO use single result event loop thread with asyncio
        if self.is_async:
            return asyncio.create_task(self._async_call(*args, **kwargs))
        return _threadpool.submit(self, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.f, name)
