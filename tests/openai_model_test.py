"""Tests for OpenAI class.

This test uses conftest.py in the same directory to set up mocks for dependencies.
"""

import json
import os
import sys

# Import pytest after sys.modules updates in conftest.py
import pytest

# Add the base directory to the path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Now we can import our module
from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass


class TestOpenAIModelClass:
    def test_inheritance(self):
        """Test that OpenAIModelClass inherits from ModelClass."""
        assert issubclass(OpenAIModelClass, ModelClass)

    def test_abstract_method(self):
        """Test that get_openai_client is abstract."""
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().get_openai_client()

    def test_dummy_model(self):
        """Test that DummyOpenAIModel works."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)

        client = model.get_openai_client()
        assert client is not None

    def test_transport_method_non_streaming(self):
        """Test the openai_transport method with non-streaming."""
        model = DummyOpenAIModel()
        model.load_model()

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello world"}],
            "stream": False,
        }

        response = model.openai_transport(json.dumps(request))
        data = json.loads(response)

        # With PR #597 we now get the raw OpenAI response back
        # The DummyOpenAIModel returns "Echo: [message content]"
        assert "Echo: Hello world" in response
        
        # Verify that the response is exactly what we expect from the dummy model
        # which returns a simple string with the format "Echo: [message content]"
        expected_response = "Echo: Hello world"
        assert expected_response == data

    def test_transport_method_streaming(self):
        """Test the openai_transport method with streaming."""
        model = DummyOpenAIModel()
        model.load_model()

        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello world"}],
            "stream": True,
        }

        response = model.openai_transport(json.dumps(request))
        data = json.loads(response)

        # With PR #597 we now get the raw chunks back as a list
        assert isinstance(data, list)
        assert len(data) > 0

        # The DummyOpenAIModel's _process_streaming_request yields chunk objects
        # with the entire content, not just the formatted objects
        full_response = ""
        
        # Verify that each chunk is a string (as expected from our DummyOpenAIModel)
        for i, chunk in enumerate(data):
            # Every chunk should be a string based on our implementation
            assert isinstance(chunk, str)
            
            # Verify the chunking is done correctly (5 chars per chunk in DummyOpenAIModel)
            expected_chunk = "Echo: Hello world"[i*5:(i+1)*5]
            assert chunk == expected_chunk
            
            full_response += chunk
                
        # Verify the full reconstructed response
        assert full_response == "Echo: Hello world"
