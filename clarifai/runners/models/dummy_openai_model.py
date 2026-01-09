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

    class Responses:
        def create(self, **kwargs):
            """Mock create method for responses API."""
            if kwargs.get("stream", False):
                return MockResponseStream(**kwargs)
            else:
                return MockResponse(**kwargs)

    class Models:
        class Model:
            def __init__(self, model_id):
                self.id = model_id

        def list(self, **kwargs):
            """Mock list method for models."""

            class ModelList:
                def __init__(self):
                    self.data = [MockOpenAIClient.Models.Model("dummy-model")]

            return ModelList()

    class Images:
        def generate(self, **kwargs):
            """Mock generate method for images."""

            # Return a simple mock image response
            class ImageResponse:
                def __init__(self):
                    self.data = [{"url": "https://example.com/image.png"}]

                def model_dump_json(self):
                    return json.dumps({"data": self.data})

            return ImageResponse()

    class Embeddings:
        def create(self, **kwargs):
            """Mock create method for embeddings."""

            # Return a simple mock embedding response
            class EmbeddingResponse:
                def __init__(self):
                    self.data = [{"embedding": [0.1, 0.2, 0.3]}]
                    self.usage = {"prompt_tokens": 10, "total_tokens": 10}

                def model_dump_json(self):
                    return json.dumps({"data": self.data, "usage": self.usage})

            return EmbeddingResponse()

    def __init__(self):
        self.chat = self  # Make self.chat point to self for compatibility
        self.completions = self.Completions()  # For compatibility with some clients
        self.responses = self.Responses()  # For responses API
        self.models = self.Models()  # For models.list() compatibility
        self.images = self.Images()  # For images.generate() compatibility
        self.embeddings = self.Embeddings()  # For embeddings.create() compatibility


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


class MockResponseStream:
    """Mock response stream that mimics the OpenAI streaming responses.create() structure."""

    class Event:
        class Usage:
            def __init__(self, input_tokens, output_tokens, total_tokens):
                self.total_tokens = total_tokens
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens

            def to_dict(self):
                return dict(
                    total_tokens=self.total_tokens,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                )

        class Delta:
            def __init__(self, text=None):
                self.type = "text"
                self.text = text

        class Content:
            def __init__(self, text=None):
                self.type = "text"
                self.text = text if text is not None else ""

        class Response:
            def __init__(self, response_id, status, created_at, output=None, usage=None):
                self.id = response_id
                self.object = "realtime.response"
                self.status = status
                self.created_at = created_at
                self.output = output
                self.usage = usage

        def __init__(self, event_type, response_id, **kwargs):
            self.type = event_type
            self.response_id = response_id

            # Different event types have different structures
            if event_type == "response.created":
                self.response = self.Response(
                    response_id=response_id,
                    status="in_progress",
                    created_at=kwargs.get("created_at", 1234567890),
                )
            elif event_type == "response.content.started":
                self.content_index = kwargs.get("content_index", 0)
                self.content = self.Content()
            elif event_type == "response.content.delta":
                self.content_index = kwargs.get("content_index", 0)
                self.delta = self.Delta(kwargs.get("text"))
            elif event_type == "response.content.completed":
                self.content_index = kwargs.get("content_index", 0)
                self.content = self.Content(kwargs.get("text"))
            elif event_type == "response.completed":
                self.response = self.Response(
                    response_id=response_id,
                    status="completed",
                    created_at=kwargs.get("created_at", 1234567890),
                    output=kwargs.get("output"),
                    usage=kwargs.get("usage"),
                )

        def to_dict(self) -> Dict[str, Any]:
            """Convert the event to a dictionary."""
            result = {
                "type": self.type,
                "response_id": self.response_id,
            }

            if hasattr(self, "response"):
                response_dict = {
                    "id": self.response.id,
                    "object": self.response.object,
                    "status": self.response.status,
                    "created_at": self.response.created_at,
                }
                if self.response.output is not None:
                    response_dict["output"] = self.response.output
                if self.response.usage is not None:
                    response_dict["usage"] = self.response.usage.to_dict()
                result["response"] = response_dict

            if hasattr(self, "content_index"):
                result["content_index"] = self.content_index

            if hasattr(self, "content"):
                result["content"] = {"type": self.content.type, "text": self.content.text}

            if hasattr(self, "delta"):
                result["delta"] = {"type": self.delta.type, "text": self.delta.text}

            return result

        def model_dump(self):
            return self.to_dict()

        def model_dump_json(self):
            """Return the event as a JSON string."""
            return json.dumps(self.to_dict())

    def __init__(self, **kwargs):
        # Generate a simple response based on the last message
        messages = kwargs.get("messages", [])
        last_message = messages[-1] if messages else {"content": ""}
        self.response_text = f"Echo: {last_message.get('content', '')}"

        self.response_id = "dummy-response-id"
        self.created_at = 1234567890
        self.model = kwargs.get("model", "gpt-4")

        # Create events
        self.events = []

        # Event 1: response.created
        self.events.append(
            self.Event("response.created", self.response_id, created_at=self.created_at)
        )

        # Event 2: response.content.started
        self.events.append(
            self.Event("response.content.started", self.response_id, content_index=0)
        )

        # Event 3: response.content.delta (full text in first chunk)
        self.events.append(
            self.Event(
                "response.content.delta",
                self.response_id,
                content_index=0,
                text=self.response_text,
            )
        )

        # Event 4: response.content.completed
        self.events.append(
            self.Event(
                "response.content.completed",
                self.response_id,
                content_index=0,
                text=self.response_text,
            )
        )

        # Event 5: response.completed with usage
        usage = self.Event.Usage(
            input_tokens=len(str(messages)),
            output_tokens=len(self.response_text),
            total_tokens=len(str(messages)) + len(self.response_text),
        )

        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": self.response_text}],
            }
        ]

        self.events.append(
            self.Event(
                "response.completed",
                self.response_id,
                created_at=self.created_at,
                output=output,
                usage=usage,
            )
        )

        self.current_event = 0
        self.include_usage = kwargs.get("stream_options", {}).get("include_usage", True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_event < len(self.events):
            event = self.events[self.current_event]
            self.current_event += 1

            # Skip usage if not requested (only in final event)
            if not self.include_usage and event.type == "response.completed":
                if hasattr(event, "response") and event.response.usage:
                    event.response.usage = None

            return event
        else:
            raise StopIteration


class MockResponse:
    """Mock response object that mimics the OpenAI responses.create() response structure."""

    class Usage:
        def __init__(self, input_tokens, output_tokens, total_tokens):
            self.total_tokens = total_tokens
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

        def to_dict(self):
            return dict(
                total_tokens=self.total_tokens,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )

    class Output:
        class Content:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        def __init__(self, content_text):
            self.type = "message"
            self.role = "assistant"
            self.content = [self.Content(content_text)]

    def __init__(self, **kwargs):
        # Generate a simple response based on the last message
        messages = kwargs.get("messages", [])
        last_message = messages[-1] if messages else {"content": ""}
        response_text = f"Echo: {last_message.get('content', '')}"

        self.output = [self.Output(response_text)]
        self.usage = self.Usage(
            **{
                "input_tokens": len(str(messages)),
                "output_tokens": len(response_text),
                "total_tokens": len(str(messages)) + len(response_text),
            }
        )

        self.id = "dummy-response-id"
        self.created_at = 1234567890
        self.model = kwargs.get("model", "gpt-4")
        self.status = "completed"
        self.object = "realtime.response"
        self.output_text = response_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response object to a dictionary."""
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "model": self.model,
            "status": self.status,
            "output_text": self.output_text,
            "output": [
                {
                    "type": output.type,
                    "role": output.role,
                    "content": [
                        {"type": content.type, "text": content.text} for content in output.content
                    ],
                }
                for output in self.output
            ],
            "usage": self.usage.to_dict(),
        }

    def model_dump(self):
        return self.to_dict()

    def model_dump_json(self):
        """Return the response as a JSON string."""
        return json.dumps(self.to_dict())


class DummyOpenAIModel(OpenAIModelClass):
    """Dummy OpenAI model implementation for testing."""

    client = MockOpenAIClient()
    model = "dummy-model"

    def _process_streaming_request(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Process a request for streaming chat.completions."""
        completion_stream = self.client.chat.completions.create(**kwargs)

        for chunk in completion_stream:
            yield chunk

    def _process_streaming_responses_request(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Process a request for streaming responses."""
        stream = self.client.responses.create(**kwargs)

        for chunk in stream:
            yield chunk

    # Override the method directly for testing
    @OpenAIModelClass.method
    def openai_stream_transport(self, req: str) -> Iterator[str]:
        """Direct implementation for testing purposes."""
        try:
            request_data = json.loads(req)
            request_data = self._update_old_fields(request_data)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming is only supported for chat completions and responses.")

            if endpoint == self.ENDPOINT_RESPONSES:
                # Handle responses endpoint
                for chunk in self._process_streaming_responses_request(**request_data):
                    self._set_usage(chunk)
                    yield json.dumps(chunk.model_dump())
            else:
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
                    self._set_usage(chunk)
                    yield json.dumps(chunk.model_dump())
        except Exception as e:
            yield f"Error: {str(e)}"

    # Additional example method that could be added for specific model implementations
    @OpenAIModelClass.method
    def test_method(self, prompt: str) -> str:
        """Test method that simply echoes the input."""
        return f"Test: {prompt}"
