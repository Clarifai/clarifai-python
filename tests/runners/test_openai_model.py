"""Test for OpenAIModelClass functionality."""

import json

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
