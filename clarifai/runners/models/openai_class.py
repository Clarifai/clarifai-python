"""Base class for creating OpenAI-compatible API server."""

import json
from typing import Any, Dict, List, Optional, Iterator

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.openai_convertor import openai_response


class OpenAIModelClass(ModelClass):
    """Base class for wrapping OpenAI-compatible servers as a model running in Clarifai.
    This handles all the transport between the API and the OpenAI-compatible server.
    Simply subclass this and implement the get_openai_client() method to return 
    the OpenAI-compatible client instance. The client is then used to handle all 
    the requests and responses.
    """

    def load_model(self):
        """Initialize the OpenAI client."""
        self.client = self.get_openai_client()

    def get_openai_client(self) -> Any:
        """Required method for each subclass to implement to return the OpenAI-compatible client to use."""
        raise NotImplementedError("Subclasses must implement get_openai_client() method")

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """The single model method to get the OpenAI-compatible request and send it to the OpenAI server
        then return its response.
        """
        # Parse the incoming message as JSON
        request_data = json.loads(msg)
        
        # Extract key parameters from the request
        model = request_data.get("model", "")
        messages = request_data.get("messages", [])
        stream = request_data.get("stream", False)
        temperature = request_data.get("temperature", 1.0)
        max_tokens = request_data.get("max_tokens", None)
        
        # Process the request using the provided OpenAI client
        try:
            if stream:
                # For streaming responses
                chunks = self._process_streaming_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Format the response as an OpenAI streaming response
                response = openai_response(
                    generated_text=chunks,
                    model=model,
                    stream=True
                )
                # Since this is a generator, we need to convert it to a list for JSON serialization
                response_list = list(response)
                return json.dumps(response_list)
            else:
                # For non-streaming responses
                completion = self._process_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Format the response as an OpenAI response
                response = openai_response(
                    generated_text=completion,
                    model=model,
                    stream=False
                )
                return json.dumps(response)
        except Exception as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "InvalidRequestError",
                    "code": "invalid_request_error"
                }
            }
            return json.dumps(error_response)

    @ModelClass.method
    def openai_stream_transport(self, req: str) -> Iterator[str]:
        """Process an OpenAI-compatible request and return a streaming response iterator.
        
        This method is used when stream=True and returns an iterator of strings directly,
        without converting to a list or JSON serializing.
        
        Args:
            req: The request as a JSON string.
            
        Returns:
            Iterator[str]: An iterator yielding text chunks from the streaming response.
        """
        # Parse the incoming message
        request_data = json.loads(req)
        
        # Extract key parameters from the request
        model = request_data.get("model", "")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 1.0)
        max_tokens = request_data.get("max_tokens", None)
        
        # Process the streaming request and return the iterator directly
        try:
            return self._process_streaming_request(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            # For errors, yield a single error message
            yield f"Error: {str(e)}"

    def _process_request(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> str:
        """Process a standard (non-streaming) request using the OpenAI client.
        
        Override this method in your subclass if you need custom processing logic.
        """
        # Default implementation - subclasses should override with actual client interaction
        completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        
        completion = self.client.chat.completions.create(**completion_args)
        return completion.choices[0].message.content

    def _process_streaming_request(
        self, 
        model: str, 
        messages: List[Dict[str, Any]], 
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """Process a streaming request using the OpenAI client.
        
        Override this method in your subclass if you need custom streaming logic.
        """
        # Default implementation - subclasses should override with actual client interaction
        completion_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        if max_tokens is not None:
            completion_args["max_tokens"] = max_tokens
        
        completion_stream = self.client.chat.completions.create(**completion_args)
        for chunk in completion_stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content