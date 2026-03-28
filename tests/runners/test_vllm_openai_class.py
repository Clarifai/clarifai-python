"""Unit tests for VLLMOpenAIModelClass and VLLMCancellationHandler."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from clarifai.runners.models.dummy_openai_model import MockOpenAIClient
from clarifai.runners.models.vllm_openai_class import (
    VLLMCancellationHandler,
    VLLMMetricsPoller,
    VLLMOpenAIModelClass,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclass — no real vLLM server needed
# ---------------------------------------------------------------------------
class DummyVLLMModel(VLLMOpenAIModelClass):
    client = MockOpenAIClient()
    model = "dummy-model"


# ---------------------------------------------------------------------------
# VLLMCancellationHandler
# ---------------------------------------------------------------------------
class TestVLLMCancellationHandler:
    def _make_handler(self):
        with patch("clarifai.runners.models.vllm_openai_class.register_item_abort_callback"):
            return VLLMCancellationHandler()

    def test_register_request_returns_unset_event(self):
        handler = self._make_handler()
        event = handler.register_request("item-1")
        assert isinstance(event, threading.Event)
        assert not event.is_set()

    def test_handle_abort_sets_event_for_registered_item(self):
        handler = self._make_handler()
        event = handler.register_request("item-1")
        handler._handle_abort("item-1")
        assert event.is_set()

    def test_handle_abort_closes_response(self):
        handler = self._make_handler()
        mock_response = MagicMock()
        handler.register_request("item-1", response=mock_response)
        handler._handle_abort("item-1")
        mock_response.close.assert_called_once()

    def test_early_abort_sets_event_on_late_register(self):
        """Abort arrives before register_request — event is immediately set on registration."""
        handler = self._make_handler()
        handler._handle_abort("item-early")
        event = handler.register_request("item-early")
        assert event.is_set()

    def test_handle_abort_unknown_item_recorded_as_early_abort(self):
        handler = self._make_handler()
        handler._handle_abort("unknown-item")
        assert "unknown-item" in handler._early_aborts

    def test_unregister_removes_all_state(self):
        handler = self._make_handler()
        mock_response = MagicMock()
        handler.register_request("item-1", response=mock_response)
        handler.unregister_request("item-1")
        assert "item-1" not in handler._cancel_events
        assert "item-1" not in handler._responses
        assert "item-1" not in handler._early_aborts


# ---------------------------------------------------------------------------
# VLLMOpenAIModelClass — health probes
# ---------------------------------------------------------------------------
class TestVLLMOpenAIModelClassProbes:
    def test_liveness_probe_no_server_delegates_to_super(self):
        model = DummyVLLMModel()
        # server is None → falls back to OpenAIModelClass.handle_liveness_probe() which returns True
        assert model.handle_liveness_probe() is True

    def test_readiness_probe_no_server_delegates_to_super(self):
        model = DummyVLLMModel()
        assert model.handle_readiness_probe() is True

    def test_liveness_probe_returns_true_on_http_200(self):
        model = DummyVLLMModel()
        model.server = MagicMock(host="localhost", port=8000)
        mock_resp = MagicMock(status_code=200)
        with patch("clarifai.runners.models.vllm_openai_class.httpx.get", return_value=mock_resp):
            assert model.handle_liveness_probe() is True

    def test_liveness_probe_returns_false_on_non_200(self):
        model = DummyVLLMModel()
        model.server = MagicMock(host="localhost", port=8000)
        mock_resp = MagicMock(status_code=503)
        with patch("clarifai.runners.models.vllm_openai_class.httpx.get", return_value=mock_resp):
            assert model.handle_liveness_probe() is False

    def test_liveness_probe_returns_false_on_exception(self):
        model = DummyVLLMModel()
        model.server = MagicMock(host="localhost", port=8000)
        with patch(
            "clarifai.runners.models.vllm_openai_class.httpx.get", side_effect=Exception("timeout")
        ):
            assert model.handle_liveness_probe() is False

    def test_readiness_probe_returns_true_on_http_200(self):
        model = DummyVLLMModel()
        model.server = MagicMock(host="localhost", port=8000)
        mock_resp = MagicMock(status_code=200)
        with patch("clarifai.runners.models.vllm_openai_class.httpx.get", return_value=mock_resp):
            assert model.handle_readiness_probe() is True

    def test_readiness_probe_returns_false_on_exception(self):
        model = DummyVLLMModel()
        model.server = MagicMock(host="localhost", port=8000)
        with patch(
            "clarifai.runners.models.vllm_openai_class.httpx.get",
            side_effect=Exception("conn refused"),
        ):
            assert model.handle_readiness_probe() is False


# ---------------------------------------------------------------------------
# VLLMOpenAIModelClass — openai_stream_transport with cancellation
# ---------------------------------------------------------------------------
def _make_mock_stream(*chunk_texts):
    """Return a mock streaming response whose chunks have the expected interface.

    _set_usage asserts that a chunk doesn't have both .usage and .response.usage set,
    so we explicitly set both to None on each chunk.
    """
    chunks = []
    for text in chunk_texts:
        chunk = MagicMock()
        chunk.usage = None
        chunk.response = None
        chunk.model_dump_json.return_value = json.dumps(
            {"choices": [{"delta": {"content": text}}], "usage": None}
        )
        chunks.append(chunk)
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(chunks))
    mock_stream.response = MagicMock()
    return mock_stream


class TestVLLMStreamTransportCancellation:
    def _model_with_mock_client_and_handler(self, cancel_event):
        model = DummyVLLMModel()
        mock_handler = MagicMock()
        mock_handler.register_request.return_value = cancel_event
        model.cancellation_handler = mock_handler
        mock_stream = _make_mock_stream("Hello", " world")
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = mock_stream
        return model, mock_handler

    def test_cancel_before_iteration_yields_no_chunks(self):
        cancel_event = threading.Event()
        cancel_event.set()  # already cancelled
        model, mock_handler = self._model_with_mock_client_and_handler(cancel_event)

        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )
        with patch(
            "clarifai.runners.models.vllm_openai_class.get_item_id", return_value="item-abc"
        ):
            chunks = list(model.openai_stream_transport(request))

        assert chunks == []
        mock_handler.unregister_request.assert_called_once_with("item-abc")

    def test_no_cancel_yields_all_chunks(self):
        cancel_event = threading.Event()  # never set
        model, mock_handler = self._model_with_mock_client_and_handler(cancel_event)

        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )
        with patch(
            "clarifai.runners.models.vllm_openai_class.get_item_id", return_value="item-xyz"
        ):
            chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 2
        mock_handler.unregister_request.assert_called_once_with("item-xyz")

    def test_unregister_called_even_when_get_item_id_fails(self):
        """If get_item_id raises, no cancellation handler is used but stream still works."""
        model = DummyVLLMModel()
        mock_stream = _make_mock_stream("chunk1")
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = mock_stream

        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )
        with patch(
            "clarifai.runners.models.vllm_openai_class.get_item_id",
            side_effect=Exception("no context"),
        ):
            chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 1

    def test_invalid_endpoint_raises_value_error(self):
        model = DummyVLLMModel()
        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "openai_endpoint": "/unsupported",
            }
        )
        with patch("clarifai.runners.models.vllm_openai_class.get_item_id", side_effect=Exception):
            with pytest.raises(ValueError, match="Only"):
                list(model.openai_stream_transport(request))


# ---------------------------------------------------------------------------
# VLLMOpenAIModelClass — admission control
# ---------------------------------------------------------------------------
class TestAdmissionControl:
    def test_check_admission_no_poller_admits(self):
        """No metrics poller set → fail-open, always admit."""
        model = DummyVLLMModel()
        assert model._metrics_poller is None
        assert model.check_admission() is True

    def test_check_admission_stale_poller_admits(self):
        """Stale poller → fail-open."""
        model = DummyVLLMModel()
        mock_poller = MagicMock(spec=VLLMMetricsPoller)
        mock_poller.is_stale = True
        model._metrics_poller = mock_poller
        assert model.check_admission() is True

    def test_check_admission_kv_cache_over_threshold_rejects(self):
        model = DummyVLLMModel()
        mock_poller = MagicMock(spec=VLLMMetricsPoller)
        mock_poller.is_stale = False
        mock_poller.KV_CACHE_REJECT_THRESHOLD = 0.95
        mock_poller.MAX_WAITING_REQUESTS = 10
        mock_poller.snapshot.return_value = (0.96, 0)  # kv_cache above threshold
        model._metrics_poller = mock_poller
        assert model.check_admission() is False

    def test_check_admission_waiting_over_limit_rejects(self):
        model = DummyVLLMModel()
        mock_poller = MagicMock(spec=VLLMMetricsPoller)
        mock_poller.is_stale = False
        mock_poller.KV_CACHE_REJECT_THRESHOLD = 0.95
        mock_poller.MAX_WAITING_REQUESTS = 10
        mock_poller.snapshot.return_value = (0.50, 11)  # waiting above limit
        model._metrics_poller = mock_poller
        assert model.check_admission() is False

    def test_check_admission_healthy_admits(self):
        model = DummyVLLMModel()
        mock_poller = MagicMock(spec=VLLMMetricsPoller)
        mock_poller.is_stale = False
        mock_poller.KV_CACHE_REJECT_THRESHOLD = 0.95
        mock_poller.MAX_WAITING_REQUESTS = 10
        mock_poller.snapshot.return_value = (0.50, 3)
        model._metrics_poller = mock_poller
        assert model.check_admission() is True

    def test_admission_control_backoff_default(self):
        assert DummyVLLMModel().admission_control_backoff == 1.0
