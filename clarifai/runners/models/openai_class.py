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
            completion_args = self._create_completion_args(request_data)
            completion = self.client.chat.completions.create(**completion_args)
            self._set_usage(completion)
            return json.dumps(completion.model_dump())

        except Exception as e:
            return f"Error: {e}"

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
            completion_args = self._create_completion_args(request_data)
            completion_stream = self.client.chat.completions.create(**completion_args)
            for chunk in completion_stream:
                self._set_usage(chunk)
                yield json.dumps(chunk.model_dump())

        except Exception as e:
            yield f"Error: {e}"
