"""Base class for creating OpenAI-compatible API server."""

from typing import Any, Dict, Iterator

from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json

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
                self.model = self.client.models.list().data[0].id
            except Exception as e:
                raise NotImplementedError(
                    "Subclasses must set the 'model' class attribute or ensure the client can list models"
                ) from e

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

    def _set_usage(self, resp):
        if resp.usage and resp.usage.prompt_tokens and resp.usage.completion_tokens:
            self.set_output_context(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )

    def _handle_chat_completions(self, request_data: Dict[str, Any]):
        """Handle chat completion requests."""
        completion_args = self._create_completion_args(request_data)
        completion = self.client.chat.completions.create(**completion_args)
        self._set_usage(completion)
        return completion

    def _handle_images_generate(self, request_data: Dict[str, Any]):
        """Handle image generation requests."""
        image_args = {**request_data}
        image_args.update({"model": self.model})
        response = self.client.images.generate(**image_args)
        return response

    def _handle_embeddings(self, request_data: Dict[str, Any]):
        """Handle embedding requests."""
        embedding_args = {**request_data}
        embedding_args.update({"model": self.model})
        response = self.client.embeddings.create(**embedding_args)
        return response

    def _handle_responses(self, request_data: Dict[str, Any]):
        """Handle response requests."""
        response_args = {**request_data}
        response_args.update({"model": self.model})
        response = self.client.responses.create(**response_args)
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
        if 'max_tokens' in request_data:
            request_data['max_completion_tokens'] = request_data.pop('max_tokens')
        if 'top_p' in request_data:
            request_data['top_p'] = float(request_data['top_p'])
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
                stream_completion = self.client.chat.completions.create(**completion_args)
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
