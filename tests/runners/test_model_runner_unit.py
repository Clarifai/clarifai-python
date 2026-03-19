"""Unit tests for ModelRunner status aggregation and iterator handling."""

import types
from unittest.mock import MagicMock

from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.runners.models.model_runner import ModelRunner


def _make_runner(mock_model):
    """Create a minimal ModelRunner-like object bound to mock_model, bypassing __init__."""

    class FakeRunner:
        pass

    runner = FakeRunner()
    runner.model = mock_model
    runner._auth_helper = None
    runner.runner_item_predict = types.MethodType(ModelRunner.runner_item_predict, runner)
    runner.runner_item_generate = types.MethodType(ModelRunner.runner_item_generate, runner)
    runner.runner_item_stream = types.MethodType(ModelRunner.runner_item_stream, runner)
    return runner


def _make_runner_item():
    """Return a RunnerItem with an empty PostModelOutputsRequest."""
    return service_pb2.RunnerItem(post_model_outputs_request=service_pb2.PostModelOutputsRequest())


def _make_resp(*output_codes):
    """Build a MultiOutputResponse whose outputs have the given status codes."""
    resp = service_pb2.MultiOutputResponse()
    for code in output_codes:
        output = resp.outputs.add()
        output.status.code = code
    return resp


class TestRunnerItemPredictStatus:
    """Tests for runner_item_predict status aggregation."""

    def test_empty_outputs_treated_as_success(self):
        """Empty resp.outputs should map to SUCCESS (consistent with all([]) == True)."""
        mock_model = MagicMock()
        mock_model.predict_wrapper.return_value = service_pb2.MultiOutputResponse()
        runner = _make_runner(mock_model)

        result = runner.runner_item_predict(_make_runner_item())
        assert result.multi_output_response.status.code == status_code_pb2.SUCCESS

    def test_all_success_outputs(self):
        mock_model = MagicMock()
        mock_model.predict_wrapper.return_value = _make_resp(
            status_code_pb2.SUCCESS, status_code_pb2.SUCCESS
        )
        runner = _make_runner(mock_model)

        result = runner.runner_item_predict(_make_runner_item())
        assert result.multi_output_response.status.code == status_code_pb2.SUCCESS
        assert result.multi_output_response.status.description == "Success"

    def test_mixed_outputs(self):
        mock_model = MagicMock()
        mock_model.predict_wrapper.return_value = _make_resp(
            status_code_pb2.SUCCESS, status_code_pb2.FAILURE
        )
        runner = _make_runner(mock_model)

        result = runner.runner_item_predict(_make_runner_item())
        assert result.multi_output_response.status.code == status_code_pb2.MIXED_STATUS

    def test_all_failed_outputs(self):
        mock_model = MagicMock()
        mock_model.predict_wrapper.return_value = _make_resp(status_code_pb2.FAILURE)
        runner = _make_runner(mock_model)

        result = runner.runner_item_predict(_make_runner_item())
        assert result.multi_output_response.status.code == status_code_pb2.FAILURE

    def test_stale_status_fields_are_cleared(self):
        """resp.status.Clear() must wipe stale fields (details, internal_details, etc.)."""
        resp = _make_resp(status_code_pb2.SUCCESS)
        resp.status.code = status_code_pb2.SUCCESS
        resp.status.description = "old"
        resp.status.details = "stale details"
        resp.status.internal_details = "stale internal"

        mock_model = MagicMock()
        mock_model.predict_wrapper.return_value = resp
        runner = _make_runner(mock_model)

        result = runner.runner_item_predict(_make_runner_item())
        out = result.multi_output_response.status
        assert out.code == status_code_pb2.SUCCESS
        assert out.description == "Success"
        assert out.details == ""
        assert out.internal_details == ""


class TestRunnerItemGenerateStatus:
    """Tests for runner_item_generate status aggregation."""

    def test_empty_outputs_treated_as_success(self):
        mock_model = MagicMock()
        mock_model.generate_wrapper.return_value = iter([service_pb2.MultiOutputResponse()])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_generate(_make_runner_item()))
        assert len(results) == 1
        assert results[0].multi_output_response.status.code == status_code_pb2.SUCCESS

    def test_stale_status_fields_are_cleared(self):
        resp = _make_resp(status_code_pb2.SUCCESS)
        resp.status.details = "stale"
        resp.status.internal_details = "stale internal"

        mock_model = MagicMock()
        mock_model.generate_wrapper.return_value = iter([resp])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_generate(_make_runner_item()))
        out = results[0].multi_output_response.status
        assert out.details == ""
        assert out.internal_details == ""


class TestRunnerItemStreamStatus:
    """Tests for runner_item_stream status aggregation and iterator handling."""

    def test_empty_outputs_treated_as_success(self):
        """Empty resp.outputs in stream should map to SUCCESS (aligned with predict/generate)."""
        mock_model = MagicMock()
        mock_model.stream_wrapper.return_value = iter([service_pb2.MultiOutputResponse()])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_stream(iter([_make_runner_item()])))
        assert len(results) == 1
        assert results[0].multi_output_response.status.code == status_code_pb2.SUCCESS

    def test_accepts_iterable_not_just_iterator(self):
        """runner_item_stream must accept any iterable (e.g. list), not only an iterator."""
        mock_model = MagicMock()
        resp = _make_resp(status_code_pb2.SUCCESS)
        mock_model.stream_wrapper.return_value = iter([resp])
        runner = _make_runner(mock_model)

        # Pass a plain list (iterable) instead of an iterator
        results = list(runner.runner_item_stream([_make_runner_item()]))
        assert len(results) == 1
        assert results[0].multi_output_response.status.code == status_code_pb2.SUCCESS

    def test_empty_stream_yields_nothing(self):
        """An empty input stream should produce no output items."""
        mock_model = MagicMock()
        mock_model.stream_wrapper.return_value = iter([])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_stream(iter([])))
        assert results == []

    def test_stale_status_fields_are_cleared(self):
        resp = _make_resp(status_code_pb2.SUCCESS)
        resp.status.details = "stale"
        resp.status.internal_details = "stale internal"

        mock_model = MagicMock()
        mock_model.stream_wrapper.return_value = iter([resp])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_stream([_make_runner_item()]))
        out = results[0].multi_output_response.status
        assert out.details == ""
        assert out.internal_details == ""

    def test_mixed_outputs(self):
        mock_model = MagicMock()
        resp = _make_resp(status_code_pb2.SUCCESS, status_code_pb2.FAILURE)
        mock_model.stream_wrapper.return_value = iter([resp])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_stream([_make_runner_item()]))
        assert results[0].multi_output_response.status.code == status_code_pb2.MIXED_STATUS

    def test_all_failed_outputs(self):
        mock_model = MagicMock()
        resp = _make_resp(status_code_pb2.FAILURE)
        mock_model.stream_wrapper.return_value = iter([resp])
        runner = _make_runner(mock_model)

        results = list(runner.runner_item_stream([_make_runner_item()]))
        assert results[0].multi_output_response.status.code == status_code_pb2.FAILURE
