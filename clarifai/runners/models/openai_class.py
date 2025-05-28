"""Base class for creating OpenAI-compatible API server."""

import json
from typing import Any, Dict, Iterator

from clarifai.runners.models.model_class import ModelClass


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

    # These should be overridden in subclasses
    client = None
    model = None

    def __init__(self) -> None:
        if self.client is None:
            raise NotImplementedError("Subclasses must set the 'client' class attribute")
        if self.model is None:
            try:
                self.model = self.client.models.list().data[0].id
            except Exception as e:
                raise NotImplementedError(
                    "Subclasses must set the 'model' class attribute or ensure the client can list models"
                ) from e

    def _extract_request_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate common openai arguments parameters from the request data.

        Args:
            request_data: The parsed JSON request data

        Returns:
            Dict containing the extracted parameters
        """
        return {
            "messages": request_data.get("messages", []),
            "temperature": request_data.get("temperature", 1.0),
            "max_tokens": request_data.get("max_tokens"),
            "max_completion_tokens": request_data.get("max_completion_tokens"),
            "n": request_data.get("n", 1),
            "frequency_penalty": request_data.get("frequency_penalty"),
            "presence_penalty": request_data.get("presence_penalty"),
            "top_p": request_data.get("top_p", 1.0),
            "reasoning_effort": request_data.get("reasoning_effort"),
            "response_format": request_data.get("response_format"),
            "stop": request_data.get("stop"),
            "tools": request_data.get("tools"),
            "tool_choice": request_data.get("tool_choice"),
            "tool_resources": request_data.get("tool_resources"),
            "modalities": request_data.get("modalities"),
            "stream_options": request_data.get("stream_options", {"include_usage": True}),
        }

    def _create_completion_args(
        self, params: Dict[str, Any], stream: bool = False
    ) -> Dict[str, Any]:
        """Create the completion arguments dictionary from parameters.

        Args:
            params: Dictionary of parameters extracted from request
            stream: Whether this is a streaming request

        Returns:
            Dict containing the completion arguments
        """
        completion_args = {
            "model": self.model,
            "messages": params["messages"],
            "temperature": params["temperature"],
        }

        if stream:
            completion_args["stream"] = True
            if params.get("stream_options"):
                completion_args["stream_options"] = params["stream_options"]

        # Add optional parameters if they exist
        optional_params = [
            "max_tokens",
            "max_completion_tokens",
            "n",
            "frequency_penalty",
            "presence_penalty",
            "top_p",
            "reasoning_effort",
            "response_format",
            "stop",
            "tools",
            "tool_choice",
            "tool_resources",
            "modalities",
        ]

        for param in optional_params:
            if params.get(param) is not None:
                completion_args[param] = params[param]

        return completion_args

    def _format_error_response(self, error: Exception) -> str:
        """Format an error response in OpenAI-compatible format.

        Args:
            error: The exception that occurred

        Returns:
            JSON string containing the error response
        """
        error_response = {
            "error": {
                "message": str(error),
                "type": "InvalidRequestError",
                "code": "invalid_request_error",
            }
        }
        return json.dumps(error_response)

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """The single model method to get the OpenAI-compatible request and send it to the OpenAI server
        then return its response.

        Args:
            msg: JSON string containing the request parameters

        Returns:
            JSON string containing the response or error
        """
        try:
            request_data = json.loads(msg)
            params = self._extract_request_params(request_data)
            stream = request_data.get("stream", False)

            if stream:
                chunks = self._process_streaming_request(**params)
                response_list = []
                for chunk in chunks:
                    response_list.append(chunk)
                return json.dumps(response_list)
            else:
                completion = self._process_request(**params)
                if completion.get('usage'):
                    if completion['usage'].get('prompt_tokens') and completion['usage'].get(
                        'completion_tokens'
                    ):
                        self.set_output_context(
                            prompt_tokens=completion['usage']['prompt_tokens'],
                            completion_tokens=completion['usage']['completion_tokens'],
                        )

                return json.dumps(completion)

        except Exception as e:
            return self._format_error_response(e)

    @ModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        """Process an OpenAI-compatible request and return a streaming response iterator.
        This method is used when stream=True and returns an iterator of strings directly,
        without converting to a list or JSON serializing.

        Args:
            msg: The request as a JSON string.

        Returns:
            Iterator[str]: An iterator yielding text chunks from the streaming response.
        """
        try:
            request_data = json.loads(msg)
            params = self._extract_request_params(request_data)
            for chunk in self._process_streaming_request(**params):
                if chunk.get('usage'):
                    if chunk['usage'].get('prompt_tokens') and chunk['usage'].get(
                        'completion_tokens'
                    ):
                        self.set_output_context(
                            prompt_tokens=chunk['usage']['prompt_tokens'],
                            completion_tokens=chunk['usage']['completion_tokens'],
                        )
                yield json.dumps(chunk)
        except Exception as e:
            yield f"Error: {str(e)}"

    def _process_request(self, **kwargs) -> Any:
        """Process a standard (non-streaming) request using the OpenAI client.

        Args:
            **kwargs: Request parameters

        Returns:
            The completion response from the OpenAI client
        """
        completion_args = self._create_completion_args(kwargs)
        return self.client.chat.completions.create(**completion_args).to_dict()

    def _process_streaming_request(self, **kwargs) -> Iterator[str]:
        """Process a streaming request using the OpenAI client.

        Args:
            **kwargs: Request parameters

        Returns:
            Iterator yielding response chunks
        """
        completion_args = self._create_completion_args(kwargs, stream=True)
        completion_stream = self.client.chat.completions.create(**completion_args)

        for chunk in completion_stream:
            yield chunk.to_dict()
