import logging
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.auth.register import RpcCallable, V2Stub

throttle_status_codes = {
    status_code_pb2.CONN_THROTTLED,
    status_code_pb2.CONN_EXCEED_HOURLY_LIMIT,
}

retry_codes_grpc = {
    grpc.StatusCode.UNAVAILABLE,
}

_threadpool = ThreadPoolExecutor(100)


def create_stub(auth_helper: ClarifaiAuthHelper = None, max_retry_attempts: int = 10) -> V2Stub:
  """
  Create client stub that handles authorization and basic retries for
  unavailable or throttled connections.

  Args:
    auth_helper:  ClarifaiAuthHelper to use for auth metadata (default: from env)
    max_retry_attempts:  max attempts to retry rpcs with retryable failures
  """
  stub = AuthorizedStub(auth_helper)
  if max_retry_attempts > 0:
    return RetryStub(stub, max_retry_attempts)
  return stub


class AuthorizedStub(V2Stub):
  """V2Stub proxy that inserts metadata authorization in rpc calls."""

  def __init__(self, auth_helper: ClarifaiAuthHelper = None):
    if auth_helper is None:
      auth_helper = ClarifaiAuthHelper.from_env()
    self.stub = auth_helper.get_stub()
    self.metadata = auth_helper.metadata

  def __getattr__(self, name):
    value = getattr(self.stub, name)
    if isinstance(value, RpcCallable):
      value = _AuthorizedRpcCallable(value, self.metadata)
    return value


class _AuthorizedRpcCallable(RpcCallable):
  """Adds metadata(authorization header) to rpc calls"""

  def __init__(self, func, metadata):
    self.f = func
    self.metadata = metadata

  def __repr__(self):
    return repr(self.f)

  def __call__(self, *args, **kwargs):
    metadata = kwargs.pop('metadata', self.metadata)
    return self.f(*args, **kwargs, metadata=metadata)

  def future(self, *args, **kwargs):
    metadata = kwargs.pop('metadata', self.metadata)
    return self.f.future(*args, **kwargs, metadata=metadata)

  def __getattr__(self, name):
    return getattr(self.f, name)


class RetryStub(V2Stub):
  """
  V2Stub proxy that retries requests (currently on unavailable server or throttle codes)
  """

  def __init__(self, stub, max_attempts=10, backoff_time=5):
    self.stub = stub
    self.max_attempts = max_attempts
    self.backoff_time = backoff_time

  def __getattr__(self, name):
    value = getattr(self.stub, name)
    if isinstance(value, RpcCallable):
      value = _RetryRpcCallable(value, self.max_attempts, self.backoff_time)
    return value


class _RetryRpcCallable(RpcCallable):
  """Retries rpc calls on unavailable server or throttle codes"""

  def __init__(self, func, max_attempts, backoff_time):
    self.f = func
    self.max_attempts = max_attempts
    self.backoff_time = backoff_time

  def __repr__(self):
    return repr(self.f)

  def __call__(self, *args, **kwargs):
    attempt = 0
    while attempt < self.max_attempts:
      attempt += 1
      if attempt != 1:
        time.sleep(self.backoff_time)  # TODO better backoff between attempts
      try:
        response = self.f(*args, **kwargs)
        if (response.status.code in throttle_status_codes) and attempt < self.max_attempts:
          logging.debug('Retrying with status %s' % str(response.status))
        else:
          return response
      except grpc.RpcError as e:
        if (e.code() in retry_codes_grpc) and attempt < self.max_attempts:
          logging.debug('Retrying with status %s' % e.code())
        else:
          raise

  def future(self, *args, **kwargs):
    # TODO use single result event loop thread with asyncio
    return _threadpool.submit(self, *args, **kwargs)

  def __getattr__(self, name):
    return getattr(self.f, name)
