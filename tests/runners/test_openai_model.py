"""Test for OpenAIModelClass functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

# Import directly from the file we created, not through the import chain
from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel
from clarifai.runners.models.openai_class import OpenAIModelClass


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


class TestRawSseStream:
    """Tests for the _raw_sse_stream fast-path that bypasses Pydantic parsing."""

    @staticmethod
    def _make_mock_stream(sse_data_lines):
        """Create a mock stream with _iter_events() and close()."""

        class _SSE:
            def __init__(self, data):
                self.data = data

        stream = MagicMock()
        stream._iter_events.return_value = iter([_SSE(d) for d in sse_data_lines])
        stream.close = MagicMock()
        return stream

    def test_raw_sse_yields_json_strings(self):
        """Verify _raw_sse_stream yields raw JSON strings and respects [DONE]."""
        model = DummyOpenAIModel()
        model.load_model()

        chunk1 = json.dumps({"id": "c1", "choices": [{"delta": {"content": "Hello"}}]})
        chunk2 = json.dumps({"id": "c2", "choices": [{"delta": {"content": " world"}}]})
        mock_stream = self._make_mock_stream([chunk1, chunk2, "[DONE]"])

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            results = list(model._raw_sse_stream({"model": "test", "stream": True}))

        assert results == [chunk1, chunk2]
        mock_stream.close.assert_called_once()

    def test_raw_sse_captures_usage_with_prompt_tokens(self):
        """Verify usage is captured when the last chunk contains prompt_tokens."""
        model = DummyOpenAIModel()
        model.load_model()

        content_chunk = json.dumps({"id": "c1", "choices": [{"delta": {"content": "Hi"}}]})
        usage_chunk = json.dumps(
            {
                "id": "c2",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )
        mock_stream = self._make_mock_stream([content_chunk, usage_chunk, "[DONE]"])

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            results = list(model._raw_sse_stream({"model": "test", "stream": True}))

        assert len(results) == 2
        # Verify token context was set
        token_contexts = getattr(model._thread_local, 'token_contexts', [])
        assert len(token_contexts) == 1
        assert token_contexts[0] == (10, 20)

    def test_raw_sse_captures_usage_with_input_tokens(self):
        """Verify usage is captured for providers using input_tokens (e.g. Anthropic-style)."""
        model = DummyOpenAIModel()
        model.load_model()

        usage_chunk = json.dumps(
            {
                "id": "c1",
                "choices": [],
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 25,
                    "total_tokens": 40,
                },
            }
        )
        mock_stream = self._make_mock_stream([usage_chunk, "[DONE]"])

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            results = list(model._raw_sse_stream({"model": "test", "stream": True}))

        assert len(results) == 1
        token_contexts = getattr(model._thread_local, 'token_contexts', [])
        assert len(token_contexts) == 1
        assert token_contexts[0] == (15, 25)

    def test_raw_sse_uses_total_minus_prompt_for_completion(self):
        """Verify completion_tokens = total_tokens - prompt_tokens when total is present."""
        model = DummyOpenAIModel()
        model.load_model()

        usage_chunk = json.dumps(
            {
                "id": "c1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 50,
                },
            }
        )
        mock_stream = self._make_mock_stream([usage_chunk, "[DONE]"])

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            list(model._raw_sse_stream({"model": "test", "stream": True}))

        token_contexts = getattr(model._thread_local, 'token_contexts', [])
        assert token_contexts[0] == (10, 40)  # completion = 50 - 10

    def test_raw_sse_fallback_when_no_iter_events(self):
        """Verify graceful fallback to Pydantic path when _iter_events is missing."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a stream that is a plain iterator (no _iter_events)
        class _MockChunk:
            def __init__(self, data):
                self._data = data
                self.usage = None

            def model_dump_json(self):
                return self._data

        class _PlainStream:
            """Plain iterable stream without _iter_events, but with close()."""

            def __init__(self, chunks):
                self._chunks = chunks
                self.close_called = False

            def __iter__(self):
                return iter(self._chunks)

            def close(self):
                self.close_called = True

        chunks = [_MockChunk(json.dumps({"id": "c1"})), _MockChunk(json.dumps({"id": "c2"}))]
        plain_stream = _PlainStream(chunks)

        with patch.object(model, '_retry_chat_completions_create', return_value=plain_stream):
            results = list(model._raw_sse_stream({"model": "test", "stream": True}))

        assert len(results) == 2
        # close() should still be called if available
        assert plain_stream.close_called is True

    def test_raw_sse_no_close_on_plain_iterator(self):
        """Verify no error when stream lacks close() (e.g. MockCompletionStream)."""
        model = DummyOpenAIModel()
        model.load_model()

        chunk = json.dumps({"id": "c1", "choices": [{"delta": {"content": "ok"}}]})

        class _MockCompletionStream:
            """Stream with _iter_events but without close()."""

            def __init__(self, events):
                self._events = events

            def _iter_events(self):
                return iter(self._events)

        events = [MagicMock(data=chunk), MagicMock(data="[DONE]")]
        mock_stream = _MockCompletionStream(events)

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            results = list(model._raw_sse_stream({"model": "test", "stream": True}))

        assert results == [chunk]

    def test_raw_sse_close_called_on_exception(self):
        """Verify stream is closed even when an exception occurs during iteration."""
        model = DummyOpenAIModel()
        model.load_model()

        def _exploding_iter():
            yield MagicMock(data=json.dumps({"id": "c1"}))
            raise RuntimeError("network error")

        mock_stream = MagicMock()
        mock_stream._iter_events.return_value = _exploding_iter()
        mock_stream.close = MagicMock()

        with patch.object(model, '_retry_chat_completions_create', return_value=mock_stream):
            with pytest.raises(RuntimeError, match="network error"):
                list(model._raw_sse_stream({"model": "test", "stream": True}))

        mock_stream.close.assert_called_once()
