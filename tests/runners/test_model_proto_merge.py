"""Test that model proto merging works correctly in model_runner and model_servicer."""

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.struct_pb2 import Struct

from clarifai.runners.models.model_runner import pmo_iterator
from clarifai.runners.models.model_servicer import ModelServicer


class DummyModel:
    """A minimal model class for testing."""

    def predict_wrapper(self, request):
        """Return a dummy response."""
        return service_pb2.MultiOutputResponse(
            status=status_pb2.Status(
                code=status_code_pb2.SUCCESS,
                description="Success",
            ),
            outputs=[
                resources_pb2.Output(
                    status=status_pb2.Status(
                        code=status_code_pb2.SUCCESS,
                        description="Success",
                    )
                )
            ],
        )

    def generate_wrapper(self, request):
        """Return a dummy generator."""
        yield service_pb2.MultiOutputResponse(
            status=status_pb2.Status(
                code=status_code_pb2.SUCCESS,
                description="Success",
            ),
            outputs=[
                resources_pb2.Output(
                    status=status_pb2.Status(
                        code=status_code_pb2.SUCCESS,
                        description="Success",
                    )
                )
            ],
        )

    def stream_wrapper(self, request_iterator):
        """Return a dummy generator."""
        for req in request_iterator:
            yield service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.SUCCESS,
                    description="Success",
                ),
                outputs=[
                    resources_pb2.Output(
                        status=status_pb2.Status(
                            code=status_code_pb2.SUCCESS,
                            description="Success",
                        )
                    )
                ],
            )


class TestModelProtoMerge:
    """Test model proto merging in model runner and servicer."""

    def test_pmo_iterator_with_empty_request_model(self):
        """Test that cached model proto is used when request has no model."""
        # Create a cached model proto with some fields
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
            app_id="cached-app-id",
        )

        # Create a request without a model field
        runner_item = service_pb2.RunnerItem(
            post_model_outputs_request=service_pb2.PostModelOutputsRequest(
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw="test input"))
                    )
                ]
            )
        )

        # Process through pmo_iterator
        result_iterator = pmo_iterator([runner_item], model_proto=cached_model)
        result_request = next(result_iterator)

        # Verify the cached model proto was applied
        assert result_request.model.id == "cached-model-id"
        assert result_request.model.user_id == "cached-user-id"
        assert result_request.model.app_id == "cached-app-id"

    def test_pmo_iterator_with_request_model_override(self):
        """Test that request model proto overrides are preserved when both exist."""
        # Create a cached model proto
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
            app_id="cached-app-id",
        )

        # Create inference params for the request
        inference_params = Struct()
        inference_params["temperature"] = 0.8
        inference_params["max_tokens"] = 100

        # Create a request with a minimal model that has output_info
        runner_item = service_pb2.RunnerItem(
            post_model_outputs_request=service_pb2.PostModelOutputsRequest(
                model=resources_pb2.Model(
                    id="request-model-id",
                    model_version=resources_pb2.ModelVersion(
                        id="request-version-id",
                        output_info=resources_pb2.OutputInfo(params=inference_params),
                    ),
                ),
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw="test input"))
                    )
                ],
            )
        )

        # Process through pmo_iterator
        result_iterator = pmo_iterator([runner_item], model_proto=cached_model)
        result_request = next(result_iterator)

        # Verify both cached and request fields are present
        # Request model ID should override cached
        assert result_request.model.id == "request-model-id"
        # But cached user_id and app_id should be preserved
        assert result_request.model.user_id == "cached-user-id"
        assert result_request.model.app_id == "cached-app-id"
        # Request's output_info should be present
        assert result_request.model.model_version.id == "request-version-id"
        assert result_request.model.model_version.output_info.params["temperature"] == 0.8
        assert result_request.model.model_version.output_info.params["max_tokens"] == 100

    def test_servicer_post_model_outputs_with_empty_request(self):
        """Test ModelServicer PostModelOutputs with empty request model."""
        # Create a cached model proto
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
            app_id="cached-app-id",
        )

        servicer = ModelServicer(DummyModel(), model_proto=cached_model)

        # Create request without model
        request = service_pb2.PostModelOutputsRequest(
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw="test")))
            ]
        )

        # Call the servicer
        response = servicer.PostModelOutputs(request)

        # Verify response is successful and model was set
        assert response.status.code == status_code_pb2.SUCCESS
        assert request.model.id == "cached-model-id"
        assert request.model.user_id == "cached-user-id"

    def test_servicer_post_model_outputs_with_request_override(self):
        """Test ModelServicer PostModelOutputs merges request overrides correctly."""
        # Create a cached model proto
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
            app_id="cached-app-id",
        )

        servicer = ModelServicer(DummyModel(), model_proto=cached_model)

        # Create inference params for the request
        inference_params = Struct()
        inference_params["temperature"] = 0.7

        # Create request with minimal model containing overrides
        request = service_pb2.PostModelOutputsRequest(
            model=resources_pb2.Model(
                id="request-model-id",
                model_version=resources_pb2.ModelVersion(
                    id="request-version-id",
                    output_info=resources_pb2.OutputInfo(params=inference_params),
                ),
            ),
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw="test")))
            ],
        )

        # Call the servicer
        response = servicer.PostModelOutputs(request)

        # Verify response is successful
        assert response.status.code == status_code_pb2.SUCCESS

        # Verify merged model has both cached and request fields
        assert request.model.id == "request-model-id"  # Request override
        assert request.model.user_id == "cached-user-id"  # From cache
        assert request.model.app_id == "cached-app-id"  # From cache
        assert request.model.model_version.id == "request-version-id"  # Request override
        assert request.model.model_version.output_info.params["temperature"] == 0.7

    def test_servicer_generate_model_outputs_merge(self):
        """Test ModelServicer GenerateModelOutputs merges correctly."""
        # Create a cached model proto
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
        )

        servicer = ModelServicer(DummyModel(), model_proto=cached_model)

        # Create request with model override
        request = service_pb2.PostModelOutputsRequest(
            model=resources_pb2.Model(id="request-model-id"),
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw="test")))
            ],
        )

        # Call the servicer
        responses = list(servicer.GenerateModelOutputs(request))

        # Verify we got responses
        assert len(responses) > 0
        assert responses[0].status.code == status_code_pb2.SUCCESS

        # Verify merge happened
        assert request.model.id == "request-model-id"
        assert request.model.user_id == "cached-user-id"

    def test_servicer_stream_model_outputs_merge(self):
        """Test ModelServicer StreamModelOutputs merges correctly."""
        # Create a cached model proto
        cached_model = resources_pb2.Model(
            id="cached-model-id",
            user_id="cached-user-id",
        )

        servicer = ModelServicer(DummyModel(), model_proto=cached_model)

        # Create request with model override
        def request_iterator():
            yield service_pb2.PostModelOutputsRequest(
                model=resources_pb2.Model(id="request-model-id"),
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(text=resources_pb2.Text(raw="test"))
                    )
                ],
            )

        # Call the servicer
        responses = list(servicer.StreamModelOutputs(request_iterator()))

        # Verify we got responses
        assert len(responses) > 0
        assert responses[0].status.code == status_code_pb2.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
