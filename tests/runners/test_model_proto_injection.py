"""
Unit tests for model proto injection in ModelServicer.
Tests that the model proto is properly passed to the servicer and injected into requests.
"""

import sys
import os

# Add parent directory to path to import without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from unittest.mock import MagicMock, Mock, patch

# Mock the imports that require heavy dependencies
sys.modules['clarifai.runners.utils.data_types'] = MagicMock()
sys.modules['clarifai.runners.utils.data_types.data_types'] = MagicMock()

from clarifai.runners.models.model_servicer import ModelServicer


class TestModelProtoInjection:
    """Test that ModelServicer properly injects model proto into requests."""

    def test_servicer_accepts_model_proto(self):
        """Test that ModelServicer can be initialized with model_proto parameter."""
        mock_model = Mock()
        mock_model_proto = resources_pb2.Model(id="test-model")

        servicer = ModelServicer(mock_model, model_proto=mock_model_proto)

        assert servicer.model == mock_model
        assert servicer.model_proto == mock_model_proto

    def test_servicer_without_model_proto(self):
        """Test that ModelServicer can be initialized without model_proto (backwards compatibility)."""
        mock_model = Mock()

        servicer = ModelServicer(mock_model)

        assert servicer.model == mock_model
        assert servicer.model_proto is None

    def test_post_model_outputs_injects_proto(self):
        """Test that PostModelOutputs injects model proto when request doesn't have it."""
        mock_model = Mock()
        mock_model_proto = resources_pb2.Model(id="test-model")
        mock_model_proto.model_version.CopyFrom(
            resources_pb2.ModelVersion(id="test-version")
        )

        servicer = ModelServicer(mock_model, model_proto=mock_model_proto)

        # Create a request without model field
        request = service_pb2.PostModelOutputsRequest()
        request.inputs.append(
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw="test"))
            )
        )

        # Mock the predict_wrapper to return a response
        mock_response = service_pb2.MultiOutputResponse()
        mock_model.predict_wrapper.return_value = mock_response

        # Call PostModelOutputs
        servicer.PostModelOutputs(request)

        # Verify that model proto was injected into the request
        assert request.HasField("model")
        assert request.model.id == "test-model"
        assert request.model.model_version.id == "test-version"

    def test_post_model_outputs_preserves_existing_proto(self):
        """Test that PostModelOutputs doesn't overwrite existing model proto in request."""
        mock_model = Mock()
        mock_model_proto = resources_pb2.Model(id="servicer-model")

        servicer = ModelServicer(mock_model, model_proto=mock_model_proto)

        # Create a request with existing model field
        request = service_pb2.PostModelOutputsRequest()
        request.model.CopyFrom(resources_pb2.Model(id="request-model"))
        request.inputs.append(
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw="test"))
            )
        )

        # Mock the predict_wrapper to return a response
        mock_response = service_pb2.MultiOutputResponse()
        mock_model.predict_wrapper.return_value = mock_response

        # Call PostModelOutputs
        servicer.PostModelOutputs(request)

        # Verify that original model proto was preserved
        assert request.HasField("model")
        assert request.model.id == "request-model"

    def test_generate_model_outputs_injects_proto(self):
        """Test that GenerateModelOutputs injects model proto when request doesn't have it."""
        mock_model = Mock()
        mock_model_proto = resources_pb2.Model(id="test-model")

        servicer = ModelServicer(mock_model, model_proto=mock_model_proto)

        # Create a request without model field
        request = service_pb2.PostModelOutputsRequest()
        request.inputs.append(
            resources_pb2.Input(
                data=resources_pb2.Data(text=resources_pb2.Text(raw="test"))
            )
        )

        # Mock the generate_wrapper to return a generator
        mock_response = service_pb2.MultiOutputResponse()
        mock_model.generate_wrapper.return_value = iter([mock_response])

        # Call GenerateModelOutputs and consume the generator
        list(servicer.GenerateModelOutputs(request))

        # Verify that model proto was injected into the request
        assert request.HasField("model")
        assert request.model.id == "test-model"

    def test_stream_model_outputs_injects_proto(self):
        """Test that StreamModelOutputs injects model proto into all requests in stream."""
        mock_model = Mock()
        mock_model_proto = resources_pb2.Model(id="test-model")

        servicer = ModelServicer(mock_model, model_proto=mock_model_proto)

        # Create multiple requests without model field
        requests = []
        for i in range(3):
            request = service_pb2.PostModelOutputsRequest()
            request.inputs.append(
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=f"test{i}"))
                )
            )
            requests.append(request)

        # Mock the stream_wrapper to return a generator
        mock_response = service_pb2.MultiOutputResponse()
        mock_model.stream_wrapper.return_value = iter([mock_response])

        # Call StreamModelOutputs and consume the generator
        list(servicer.StreamModelOutputs(iter(requests)))

        # Verify that model proto was injected into all requests
        for request in requests:
            assert request.HasField("model")
            assert request.model.id == "test-model"
