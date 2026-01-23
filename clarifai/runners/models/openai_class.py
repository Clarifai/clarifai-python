"""Base class for creating OpenAI-compatible API server."""

from typing import Any, Dict, Iterator

import httpx
from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from clarifai.runners.models.model_class import ModelClass
from clarifai.utils.logging import logger


class OpenAIModelClass(ModelClass):
    """Base class for wrapping OpenAI-compatible servers as a model running in Clarifai.
    This handles all the transport between the API and the OpenAI-compatible server.

    To use this class, create a subclass and set the following class attributes:
    - client: The OpenAI-compatible client instance
    - model: The name of the model to use with the client

    Example:
        class MyOpenAIModel(OpenAIModelClass):
            client = OpenAI(api_key="your-key")
            model = "gpt-4"
    """

    # API Endpoints
    ENDPOINT_CHAT_COMPLETIONS = "/chat/completions"
    ENDPOINT_IMAGES_GENERATE = "/images/generations"
    ENDPOINT_EMBEDDINGS = "/embeddings"
    ENDPOINT_RESPONSES = "/responses"

    # Default endpoint
    DEFAULT_ENDPOINT = ENDPOINT_CHAT_COMPLETIONS

    # These should be overridden in subclasses
    client = None
    model = None

    def __init__(self) -> None:
        super().__init__()
        if self.client is None:
            raise NotImplementedError("Subclasses must set the 'client' class attribute")
        if self.model is None:
            try:
                self.model = self._retry_models_list().data[0].id
            except Exception as e:
                raise NotImplementedError(
                    "Subclasses must set the 'model' class attribute or ensure the client can list models"
                ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.ConnectError),
    )
    def _retry_models_list(self, **kwargs):
        """List models with retry logic."""
        return self.client.models.list(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.ConnectError),
    )
    def _retry_chat_completions_create(self, **kwargs):
        """Create chat completions with retry logic."""
        return self.client.chat.completions.create(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.ConnectError),
    )
    def _retry_images_generate(self, **kwargs):
        """Generate images with retry logic."""
        return self.client.images.generate(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.ConnectError),
    )
    def _retry_embeddings_create(self, **kwargs):
        """Create embeddings with retry logic."""
        return self.client.embeddings.create(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.ConnectError),
    )
    def _retry_responses_create(self, **kwargs):
        """Create responses with retry logic."""
        return self.client.responses.create(**kwargs)

    def _create_completion_args(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create the completion arguments dictionary from parameters.

        Args:
            params: Dictionary of parameters extracted from request

        Returns:
            Dict containing the completion arguments
        """
        completion_args = {**params}
        completion_args.update({"model": self.model})
        stream = completion_args.pop("stream", False)
        if stream:
            # Force to use usage
            stream_options = params.pop("stream_options", {})
            stream_options.update({"include_usage": True})
            completion_args["stream_options"] = stream_options
        completion_args["stream"] = stream

        return completion_args

    def handle_liveness_probe(self) -> bool:
        """Handle liveness probe by checking if the client can list models."""
        try:
            _ = self._retry_models_list()
            return True
        except Exception as e:
            logger.error(f"Liveness probe failed: {e}", exc_info=True)
            return False

    def handle_readiness_probe(self) -> bool:
        """Handle readiness probe by checking if the client can list models."""
        try:
            _ = self._retry_models_list()
            return True
        except Exception as e:
            logger.error(f"Readiness probe failed: {e}", exc_info=True)
            return False

    def _set_usage(self, resp):
        """Set token usage {prompt_tokens, completion_tokens} from response object.

        Args:
            resp (Union[Response, ResponseStreamEvent, ChatCompletion, ChatCompletionChunk]):
        """
        # logger.info(f"response received: {resp}")
        # of stream and non-stream chat.completions.create, non-stream responses.create
        # {ChatCompletion, ChatCompletionChunk, Response}.usage
        has_usage = getattr(resp, "usage", None)
        # of stream responses.create
        # ResponseStreamEvent.response.usage
        has_response_usage = getattr(resp, "response", None) and getattr(
            resp.response, "usage", None
        )
        assert not (has_response_usage and has_usage), (
            "Both resp.usage and resp.response.usage are present, ambiguous which to use."
        )
        if has_response_usage or has_usage:
            prompt_tokens = 0
            completion_tokens = 0
            if has_usage:
                prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or getattr(
                    resp.usage, "input_tokens", 0
                )
                completion_tokens = getattr(resp.usage, "completion_tokens", 0) or getattr(
                    resp.usage, "output_tokens", 0
                )
            # stream responses.create
            else:
                prompt_tokens = getattr(resp.response.usage, "input_tokens", 0)
                completion_tokens = getattr(resp.response.usage, "output_tokens", 0)
            if prompt_tokens is None:
                prompt_tokens = 0
            if completion_tokens is None:
                completion_tokens = 0
            assert prompt_tokens > 0 or completion_tokens > 0, ValueError(
                f"Invalid token usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}. Must be greater than 0."
            )
            self.set_output_context(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            # logger.info(f"Token usage - prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")

    def _handle_chat_completions(self, request_data: Dict[str, Any]):
        """Handle chat completion requests."""
        completion_args = self._create_completion_args(request_data)
        completion = self._retry_chat_completions_create(**completion_args)
        self._set_usage(completion)
        return completion

    def _handle_images_generate(self, request_data: Dict[str, Any]):
        """Handle image generation requests."""
        image_args = {**request_data}
        image_args.update({"model": self.model})
        response = self._retry_images_generate(**image_args)
        return response

    def _handle_embeddings(self, request_data: Dict[str, Any]):
        """Handle embedding requests."""
        embedding_args = {**request_data}
        embedding_args.update({"model": self.model})
        response = self._retry_embeddings_create(**embedding_args)
        return response

    def _handle_responses(self, request_data: Dict[str, Any]):
        """Handle response requests."""
        response_args = {**request_data}
        response_args.update({"model": self.model})
        response = self._retry_responses_create(**response_args)
        self._set_usage(response)
        return response

    def _route_request(self, endpoint: str, request_data: Dict[str, Any]):
        """Route the request to appropriate handler based on endpoint."""
        handlers = {
            self.ENDPOINT_CHAT_COMPLETIONS: self._handle_chat_completions,
            self.ENDPOINT_IMAGES_GENERATE: self._handle_images_generate,
            self.ENDPOINT_EMBEDDINGS: self._handle_embeddings,
            self.ENDPOINT_RESPONSES: self._handle_responses,
        }

        handler = handlers.get(endpoint)
        if not handler:
            raise ValueError(f"Unsupported endpoint: {endpoint}")

        return handler(request_data)

    def _update_old_fields(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update old fields in the request data to match current API expectations.

        This is needed because API callers may have an old openAI client sending old fields
        compared to the client within the model.

        Note: this updates the request data in place and returns it.
        """
        # Sync max_tokens and max_completion_tokens, preferring max_completion_tokens if both exist
        max_tokens = request_data.get('max_tokens')
        max_completion_tokens = request_data.get('max_completion_tokens')

        if max_completion_tokens is not None and max_tokens is not None:
            # Both exist - prefer max_completion_tokens and sync max_tokens to it
            request_data['max_tokens'] = max_completion_tokens
        elif max_completion_tokens is not None:
            # Only max_completion_tokens exists - copy to max_tokens for older backends
            request_data['max_tokens'] = max_completion_tokens
        elif max_tokens is not None:
            # Only max_tokens exists - copy to max_completion_tokens for newer backends
            request_data['max_completion_tokens'] = max_tokens
        if 'top_p' in request_data:
            request_data['top_p'] = float(request_data['top_p'])
        if 'top_k' in request_data:
            top_k = int(request_data.pop("top_k", -1))
            if top_k > 0:
                extra_body = request_data.get("extra_body", {})
                assert isinstance(extra_body, dict), ValueError(
                    "`extra_body` must be a dictionary"
                )
                extra_body.update({"top_k": top_k})
                request_data.update({"extra_body": extra_body})

        # Note(zeiler): temporary fix for our playground sending additional fields.
        # FIXME: remove this once the playground is updated.
        # Shouldn't need to do anything with the responses API since playground isn't yet using it.
        if 'messages' in request_data:
            for m in request_data['messages']:
                m.pop('id', None)
                m.pop('file', None)
                m.pop('panelId', None)

        # Handle the "Currently only named tools are supported." error we see from trt-llm
        if 'tools' in request_data and request_data['tools'] is None:
            request_data.pop('tools', None)
        if 'tool_choice' in request_data and request_data['tool_choice'] is None:
            request_data.pop('tool_choice', None)

        return request_data

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Process an OpenAI-compatible request and send it to the appropriate OpenAI endpoint.

        Args:
            msg: JSON string containing the request parameters including 'openai_endpoint'

        Returns:
            JSON string containing the response or error
        """
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            response = self._route_request(endpoint, request_data)
            return response.model_dump_json()
        except Exception as e:
            logger.exception(e)
            error_obj = {
                "code": status_code_pb2.MODEL_PREDICTION_FAILED,
                "description": "Model prediction failed",
                "details": str(e),
            }
            return to_json(error_obj)

    @ModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        """Process an OpenAI-compatible request and return a streaming response iterator.
        This method is used when stream=True and returns an iterator of strings directly,
        without converting to a list or JSON serializing. Supports chat completions and responses endpoints.

        Args:
            msg: The request as a JSON string.

        Returns:
            Iterator[str]: An iterator yielding text chunks from the streaming response.
        """
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming is only supported for chat completions and responses.")

            if endpoint == self.ENDPOINT_RESPONSES:
                # Handle responses endpoint
                stream_response = self._route_request(endpoint, request_data)
                for chunk in stream_response:
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
            else:
                completion_args = self._create_completion_args(request_data)
                stream_completion = self._retry_chat_completions_create(**completion_args)
                for chunk in stream_completion:
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()

        except Exception as e:
            logger.exception(e)
            error_obj = {
                "code": status_code_pb2.MODEL_PREDICTION_FAILED,
                "description": "Model prediction failed",
                "details": str(e),
            }
            yield to_json(error_obj)
