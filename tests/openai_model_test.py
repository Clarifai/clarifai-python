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
        """Test that has `client` attribute."""
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().client

    def test_dummy_model(self):
        """Test that DummyOpenAIModel works."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)

        client = model.client
        assert client is not None
        assert hasattr(client, 'chat')
        assert hasattr(client, 'completions')

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

        # Verify response structure
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert "Echo: Hello world" in data["choices"][0]["message"]["content"]
        assert "usage" in data

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

        assert isinstance(data, list)
        assert len(data) > 0

        # Check first chunk for content
        first_chunk = data[0]
        assert "id" in first_chunk
        assert "created" in first_chunk
        assert "model" in first_chunk
        assert "choices" in first_chunk
        assert len(first_chunk["choices"]) > 0
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]
        assert "Echo: Hello world" in first_chunk["choices"][0]["delta"]["content"]

        # Check remaining chunks for structure
        for chunk in data[1:]:
            assert "id" in chunk
            assert "created" in chunk
            assert "model" in chunk
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0
            assert "delta" in chunk["choices"][0]
