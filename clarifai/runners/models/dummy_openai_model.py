"""Dummy OpenAI model implementation for testing."""

import json
from typing import Any, Dict, Iterator

from clarifai.runners.models.openai_class import OpenAIModelClass


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    class Completions:
        def create(self, **kwargs):
            """Mock create method for compatibility."""
            if kwargs.get("stream", False):
                return MockCompletionStream(**kwargs)
            else:
                return MockCompletion(**kwargs)

    def __init__(self):
        self.chat = self  # Make self.chat point to self for compatibility
        self.completions = self.Completions()  # For compatibility with some clients


class MockCompletion:
    """Mock completion object that mimics the OpenAI completion response structure."""

    class Usage:
        def __init__(self, prompt_tokens, completion_tokens, total_tokens):
            self.total_tokens = total_tokens
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

        def to_dict(self):
            return dict(
                total_tokens=self.total_tokens,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )

    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"

        def __init__(self, content):
            self.message = self.Message(content)
            self.finish_reason = "stop"
            self.index = 0

    def __init__(self, **kwargs):
        # Generate a simple response based on the last message
        messages = kwargs.get("messages")
        last_message = messages[-1] if messages else {"content": ""}
        response_text = f"Echo: {last_message.get('content', '')}"

        self.choices = [self.Choice(response_text)]
        self.usage = self.Usage(
            **{
                "prompt_tokens": len(str(messages)),
                "completion_tokens": len(response_text),
                "total_tokens": len(str(messages)) + len(response_text),
            }
        )

        self.id = "dummy-completion-id"
        self.created = 1234567890
        self.model = "dummy-model"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the completion object to a dictionary."""
        return {
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "message": {"role": choice.message.role, "content": choice.message.content},
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                }
                for choice in self.choices
            ],
            "usage": self.usage.to_dict(),
        }

    def model_dump(self):
        return self.to_dict()

    def model_dump_json(self):
        """Return the completion as a JSON string."""
        return json.dumps(self.to_dict())


class MockCompletionStream:
    """Mock completion stream that mimics the OpenAI streaming response structure."""

    class Chunk:
        class Choice:
            class Delta:
                def __init__(self, content=None):
                    self.content = content
                    self.role = "assistant" if content is None else None

            class Usage:
                def __init__(self, prompt_tokens, completion_tokens, total_tokens):
                    self.total_tokens = total_tokens
                    self.prompt_tokens = prompt_tokens
                    self.completion_tokens = completion_tokens

                def to_dict(self):
                    return dict(
                        total_tokens=self.total_tokens,
                        prompt_tokens=self.prompt_tokens,
                        completion_tokens=self.completion_tokens,
                    )

            def __init__(self, content=None, include_usage=False):
                self.delta = self.Delta(content)
                self.finish_reason = None if content else "stop"
                self.index = 0
                self.usage = (
                    self.Usage(**{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                    if include_usage
                    else self.Usage(None, None, None)
                )

        def __init__(self, content=None, include_usage=False):
            self.choices = [self.Choice(content, include_usage)]
            self.id = "dummy-chunk-id"
            self.created = 1234567890
            self.model = "dummy-model"
            self.usage = self.choices[0].usage

        def to_dict(self) -> Dict[str, Any]:
            """Convert the chunk to a dictionary."""
            result = {
                "id": self.id,
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "delta": {"role": choice.delta.role, "content": choice.delta.content}
                        if choice.delta.content is not None
                        else {"role": choice.delta.role},
                        "finish_reason": choice.finish_reason,
                        "index": choice.index,
                    }
                    for choice in self.choices
                ],
            }
            if self.usage:
                result["usage"] = self.usage.to_dict()
            return result

        def model_dump(self):
            return self.to_dict()

        def model_dump_json(self):
            """Return the chunk as a JSON string."""
            return json.dumps(self.to_dict())

    def __init__(self, **kwargs):
        # Generate a simple response based on the last message
        messages = kwargs.get("messages")

        last_message = messages[-1] if messages else {"content": ""}
        self.response_text = f"Echo: {last_message.get('content', '')}"
        # Create chunks that ensure the full text is included in the first chunk
        self.chunks = [
            self.response_text,  # First chunk contains the full text
            "",  # Final chunk is empty to indicate completion
        ]
        self.current_chunk = 0
        self.include_usage = kwargs.get("stream_options", {}).get("include_usage")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk < len(self.chunks):
            chunk = self.Chunk(self.chunks[self.current_chunk], self.include_usage)
            self.current_chunk += 1
            return chunk
        else:
            raise StopIteration


class DummyOpenAIModel(OpenAIModelClass):
    """Dummy OpenAI model implementation for testing."""

    client = MockOpenAIClient()
    model = "dummy-model"

    def _process_request(self, **kwargs) -> Dict[str, Any]:
        """Process a request for non-streaming responses."""
        completion_args = self._create_completion_args(kwargs)
        return self.client.chat.completions.create(**completion_args).model_dump()

    def _process_streaming_request(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Process a request for streaming responses."""
        completion_stream = self.client.chat.completions.create(**kwargs)

        for chunk in completion_stream:
            yield chunk.model_dump()

    # Override the method directly for testing
    @OpenAIModelClass.method
    def openai_stream_transport(self, req: str) -> Iterator[str]:
        """Direct implementation for testing purposes."""
        try:
            request_data = json.loads(req)
            request_data = self._create_completion_args(request_data)
            # Validate messages
            if not request_data.get("messages"):
                yield "Error: No messages provided"
                return

            for message in request_data["messages"]:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    yield "Error: Invalid message format"
                    return

            for chunk in self._process_streaming_request(**request_data):
                yield json.dumps(chunk)
        except Exception as e:
            yield f"Error: {str(e)}"

    # Additional example method that could be added for specific model implementations
    @OpenAIModelClass.method
    def test_method(self, prompt: str) -> str:
        """Test method that simply echoes the input."""
        return f"Test: {prompt}"
