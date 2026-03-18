"""Test for OpenAIModelClass functionality."""

import json

import pytest

# Import directly from the file we created, not through the import chain
from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel, MockOpenAIClient
from clarifai.runners.models.openai_class import OpenAIModelClass


class _MinimalOpenAIModel(OpenAIModelClass):
    """Minimal OpenAIModelClass subclass that does NOT override openai_stream_transport.
    Used to test the parent-class streaming path (_raw_sse_stream).
    """

    client = MockOpenAIClient()
    model = "test-model"


class TestOpenAIModelClass:
    """Tests for OpenAIModelClass."""

    def test_openai_model_initialization(self):
        """Test that OpenAIModelClass can be initialized."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)

        # Test that subclass must have `client` attribute
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().client

    def test_openai_transport_non_streaming(self):
        """Test OpenAI transport method with non-streaming request."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": False,
        }

        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)

        # Verify the response format
        assert "id" in response
        assert "created" in response
        assert "model" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        assert "Echo: Hello, world!" in response["choices"][0]["message"]["content"]
        assert "usage" in response

    def test_openai_transport(self):
        """Test OpenAI transport method with streaming request."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request with streaming
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": False,
        }

        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))

        response = json.loads(response_str)
        assert "id" in response
        assert "created" in response
        assert "model" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        if response["choices"][0].get("content"):
            assert "Echo: Hello, world!" in response["choices"][0]["content"]

    def test_openai_stream_transport(self):
        """Test the new openai_stream_transport method."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": True,
        }

        # Test with string input
        response_iter = model.openai_stream_transport(json.dumps(request))
        chunks_text = list(response_iter)
        chunks = [json.loads(resp) for resp in chunks_text]
        # Verify response format - should be raw text chunks
        assert len(chunks) > 0
        combined = ''.join(chunks_text)
        assert "Echo: Hello, world!" in combined
        assert chunks[-1]["usage"]["total_tokens"] > 0
        assert chunks[-1]["usage"]["prompt_tokens"] > 0
        assert chunks[-1]["usage"]["completion_tokens"] > 0

        # Test error handling
        bad_request = json.dumps({"messages": [{"role": "invalid"}]})
        response_iter = model.openai_stream_transport(bad_request)
        chunks = list(response_iter)
        assert len(chunks) == 1
        assert chunks[0].startswith("Error:")

        # Test return usage even if not set it
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": True,
            "stream_options": {"include_usage": False},
        }
        response_iter = model.openai_stream_transport(json.dumps(request))
        chunks_text = list(response_iter)
        chunks = [json.loads(resp) for resp in chunks_text]

        assert len(chunks) > 0
        combined = ''.join(chunks_text)
        # Verify response format - should be raw text chunks
        assert "Echo: Hello, world!" in combined
        # Verify usage still returns
        assert chunks[-1]["usage"]["total_tokens"] > 0
        assert chunks[-1]["usage"]["prompt_tokens"] > 0
        assert chunks[-1]["usage"]["completion_tokens"] > 0

    def test_custom_method(self):
        """Test custom method on the DummyOpenAIModel."""
        model = DummyOpenAIModel()
        result = model.test_method("test input")
        assert result == "Test: test input"

    def test_openai_stream_responses_api(self):
        model = DummyOpenAIModel()
        model.load_model()
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": True,
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }
        response_iter = model.openai_stream_transport(json.dumps(request))

        events = [json.loads(each) for each in response_iter]

        # Should have 5 events
        assert len(events) == 5

        # Check event types in order
        expected_types = [
            "response.created",
            "response.content.started",
            "response.content.delta",
            "response.content.completed",
            "response.completed",
        ]

        actual_types = [event["type"] for event in events]
        assert actual_types == expected_types

        ## Test usage
        final_event = events[-1]

        assert final_event["type"] == "response.completed"
        usage = final_event["response"].get("usage")

        assert usage is not None
        assert usage['input_tokens'] > 0
        assert usage['output_tokens'] > 0
        assert usage['total_tokens'] == usage['input_tokens'] + usage['output_tokens']
        received_in_toks, received_out_toks = getattr(
            model._thread_local, "token_contexts", [(None, None)]
        )[0]
        assert usage['input_tokens'] == received_in_toks
        assert usage['output_tokens'] == received_out_toks

    def test_openai_transport_responses_api(self):
        """Test OpenAI transport method with streaming request."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request with streaming
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": False,
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }

        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)
        assert "id" in response
        assert "created_at" in response
        assert "model" in response
        assert "output" in response
        assert len(response["output"]) > 0
        if response["output_text"]:
            assert "Echo: Hello, world!" in response["output_text"]

    def test_raw_sse_stream_with_iter_events(self):
        """Test _raw_sse_stream with a mock stream exposing _iter_events (fast path)."""

        class MockSSEEvent:
            def __init__(self, data):
                self.data = data

        class MockStreamWithIterEvents:
            """Mock stream that exposes _iter_events, simulating the OpenAI SDK fast path."""

            def __init__(self, events):
                self._events = events
                self._closed = False

            def _iter_events(self):
                for event in self._events:
                    yield event

            def close(self):
                self._closed = True

        # Build SSE events: one content chunk, one usage chunk, then [DONE]
        content_data = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [{"delta": {"content": "Hello, world!"}, "index": 0}],
                "usage": None,
            }
        )
        usage_data = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

        events = [
            MockSSEEvent(content_data),
            MockSSEEvent(usage_data),
            MockSSEEvent("[DONE]"),
        ]
        mock_stream = MockStreamWithIterEvents(events)

        # Patch _retry_chat_completions_create to return the mock stream
        model = DummyOpenAIModel()
        model.load_model()
        model._retry_chat_completions_create = lambda **kw: mock_stream

        completion_args = {"model": "gpt-4", "messages": [], "stream": True}
        chunks = list(model._raw_sse_stream(completion_args))

        # Should yield content and usage chunks but not [DONE]
        assert len(chunks) == 2
        assert json.loads(chunks[0])["choices"][0]["delta"]["content"] == "Hello, world!"
        assert json.loads(chunks[1])["usage"]["total_tokens"] == 15

        # Verify the stream was closed
        assert mock_stream._closed

        # Verify usage was extracted and set
        token_contexts = getattr(model._thread_local, 'token_contexts', [])
        assert len(token_contexts) == 1
        pt, ct = token_contexts[0]
        assert pt == 10
        assert ct == 5

    def test_raw_sse_stream_fallback_without_iter_events(self):
        """Test _raw_sse_stream falls back to Pydantic path when _iter_events is missing."""

        class MockChunk:
            def __init__(self, usage=None):
                self.usage = usage

            def model_dump_json(self):
                return json.dumps({"choices": [], "usage": None})

        class MockStreamWithoutIterEvents:
            """Mock stream without _iter_events, simulating an older/different SDK."""

            def __init__(self):
                self._closed = False
                self._chunks = [MockChunk()]

            def __iter__(self):
                return iter(self._chunks)

            def close(self):
                self._closed = True

        mock_stream = MockStreamWithoutIterEvents()
        model = DummyOpenAIModel()
        model.load_model()
        model._retry_chat_completions_create = lambda **kw: mock_stream

        completion_args = {"model": "gpt-4", "messages": [], "stream": True}
        chunks = list(model._raw_sse_stream(completion_args))

        # Fallback path should still yield chunks
        assert len(chunks) == 1
        assert mock_stream._closed

    def test_raw_sse_stream_via_openai_stream_transport(self):
        """Test that openai_stream_transport routes chat completions through _raw_sse_stream."""

        class MockSSEEvent:
            def __init__(self, data):
                self.data = data

        class MockStreamWithIterEvents:
            def __init__(self, events):
                self._events = events
                self._closed = False

            def _iter_events(self):
                for event in self._events:
                    yield event

            def close(self):
                self._closed = True

        content_data = json.dumps(
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [{"delta": {"content": "Echo: Hello!"}, "index": 0}],
                "usage": None,
            }
        )
        usage_data = json.dumps(
            {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "model": "gpt-4",
                "choices": [],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        )

        events = [MockSSEEvent(content_data), MockSSEEvent(usage_data), MockSSEEvent("[DONE]")]
        mock_stream = MockStreamWithIterEvents(events)

        # Use the minimal subclass that inherits openai_stream_transport without overriding it.
        model = _MinimalOpenAIModel()
        model._retry_chat_completions_create = lambda **kw: mock_stream

        request = json.dumps(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            }
        )
        chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 2
        first = json.loads(chunks[0])
        assert first["choices"][0]["delta"]["content"] == "Echo: Hello!"
        last = json.loads(chunks[1])
        assert last["usage"]["total_tokens"] == 12
        assert mock_stream._closed
