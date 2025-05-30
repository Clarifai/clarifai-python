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

    def test_openai_transport_streaming(self):
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
            "stream": True,
        }

        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response_chunks = json.loads(response_str)

        # Verify the response format for streaming
        assert isinstance(response_chunks, list)
        assert len(response_chunks) > 0
        for chunk in response_chunks:
            assert "id" in chunk
            assert "created" in chunk
            assert "model" in chunk
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0
            assert "delta" in chunk["choices"][0]
            if chunk["choices"][0]["delta"].get("content"):
                assert "Echo: Hello, world!" in chunk["choices"][0]["delta"]["content"]

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
        }

        # Test with string input
        response_iter = model.openai_stream_transport(json.dumps(request))
        chunks = list(response_iter)

        # Verify response format - should be raw text chunks
        assert len(chunks) > 0
        combined = ''.join(chunks)
        assert "Echo: Hello, world!" in combined

        # Test error handling
        bad_request = json.dumps({"messages": [{"role": "invalid"}]})
        response_iter = model.openai_stream_transport(bad_request)
        chunks = list(response_iter)
        assert len(chunks) == 1
        assert chunks[0].startswith("Error:")

    def test_custom_method(self):
        """Test custom method on the DummyOpenAIModel."""
        model = DummyOpenAIModel()
        result = model.test_method("test input")
        assert result == "Test: test input"
