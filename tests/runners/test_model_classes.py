"""Test cases for MCPModelClass and OpenAIModelClass."""

import json

import pytest

from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel
from clarifai.runners.models.mcp_class import MCPModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass


class TestModelClasses:
    """Tests for model classes."""

    def test_mcp_model_initialization(self):
        """Test that MCPModelClass requires subclass implementation."""
        # Test that subclass must implement get_server()
        with pytest.raises(NotImplementedError):
            MCPModelClass().get_server()

    def test_openai_model_initialization(self):
        """Test that OpenAIModelClass can be initialized."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)

        # Test that subclass must implement get_openai_client()
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().get_openai_client()

        # Test that client has required attributes
        client = model.get_openai_client()
        assert hasattr(client, 'chat')
        assert hasattr(client, 'completions')

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

        # Verify response structure
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

        assert isinstance(response_chunks, list)
        assert len(response_chunks) > 0

        # Check first chunk for content
        first_chunk = response_chunks[0]
        assert "id" in first_chunk
        assert "created" in first_chunk
        assert "model" in first_chunk
        assert "choices" in first_chunk
        assert len(first_chunk["choices"]) > 0
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]
        assert "Echo: Hello world" in first_chunk["choices"][0]["delta"]["content"]

    def test_custom_method(self):
        """Test custom method on the DummyOpenAIModel."""
        model = DummyOpenAIModel()
        result = model.test_method("test input")
        assert result == "Test: test input"
