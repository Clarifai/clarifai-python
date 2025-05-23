"""Test for OpenAIModelClass functionality."""

import json
import pytest

# Import directly from the file we created, not through the import chain
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel


class TestOpenAIModelClass:
    """Tests for OpenAIModelClass."""
    
    def test_openai_model_initialization(self):
        """Test that OpenAIModelClass can be initialized."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)
        
        # Test that subclass must implement get_openai_client()
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().get_openai_client()
    
    def test_openai_transport_non_streaming(self):
        """Test OpenAI transport method with non-streaming request."""
        model = DummyOpenAIModel()
        model.load_model()
        
        # Create a simple chat request
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"}
            ],
            "stream": False
        }
        
        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)
        
        # Verify the response format
        assert "id" in response
        assert "object" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        assert "Echo: Hello, world!" in response["choices"][0]["message"]["content"]
    
    def test_openai_transport_streaming(self):
        """Test OpenAI transport method with streaming request."""
        model = DummyOpenAIModel()
        model.load_model()
        
        # Create a simple chat request with streaming
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"}
            ],
            "stream": True
        }
        
        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response_chunks = json.loads(response_str)
        
        # Verify the response format for streaming
        assert isinstance(response_chunks, list)
        assert len(response_chunks) > 0
        
        # Check all chunks except the last one
        for chunk in response_chunks[:-1]:
            assert "id" in chunk
            assert "object" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0
            assert "delta" in chunk["choices"][0]
            assert "content" in chunk["choices"][0]["delta"]
        
        # Check the last chunk (should have empty delta and finish_reason)
        last_chunk = response_chunks[-1]
        assert "choices" in last_chunk
        assert len(last_chunk["choices"]) > 0
        assert "delta" in last_chunk["choices"][0]
        assert len(last_chunk["choices"][0]["delta"]) == 0
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
    
    def test_custom_method(self):
        """Test custom method on the DummyOpenAIModel."""
        model = DummyOpenAIModel()
        result = model.test_method("test input")
        assert result == "Test: test input"