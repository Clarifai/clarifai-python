"""Unit tests for SGLangOpenAIModelClass and SGLangCancellationHandler."""

import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from clarifai.runners.models.dummy_openai_model import MockOpenAIClient
from clarifai.runners.models.sglang_openai_class import (
    SGLangCancellationHandler,
    SGLangOpenAIModelClass,
)


# ---------------------------------------------------------------------------
# Minimal concrete subclass — no real sglang server needed
# ---------------------------------------------------------------------------
class DummySGLangModel(SGLangOpenAIModelClass):
    client = MockOpenAIClient()
    model = "dummy-model"


# ---------------------------------------------------------------------------
# SGLangCancellationHandler
# ---------------------------------------------------------------------------
class TestSGLangCancellationHandler:
    def _make_handler(self, base_url="http://localhost:23333"):
        with patch("clarifai.runners.models.sglang_openai_class.register_item_abort_callback"):
            return SGLangCancellationHandler(base_url)

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
        handler.register_rid("item-1", "rid-xyz")
        handler.unregister_request("item-1")
        assert "item-1" not in handler._cancel_events
        assert "item-1" not in handler._responses
        assert "item-1" not in handler._rids
        assert "item-1" not in handler._early_aborts

    def test_handle_abort_after_rid_registered_calls_abort_request(self):
        handler = self._make_handler("http://host:23333")
        handler.register_request("item-1")
        handler.register_rid("item-1", "rid-abc")
        with patch("clarifai.runners.models.sglang_openai_class.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, text="ok")
            handler._handle_abort("item-1")
            mock_post.assert_called_once_with(
                "http://host:23333/abort_request",
                json={"rid": "rid-abc"},
                timeout=2,
            )

    def test_handle_abort_without_rid_does_not_call_abort_request(self):
        handler = self._make_handler()
        handler.register_request("item-1")
        with patch("clarifai.runners.models.sglang_openai_class.requests.post") as mock_post:
            handler._handle_abort("item-1")
            mock_post.assert_not_called()

    def test_register_rid_after_cancel_triggers_abort_request(self):
        """If the request was cancelled before the rid was known, register_rid
        should immediately call /abort_request."""
        handler = self._make_handler("http://host:23333")
        handler.register_request("item-1")
        # Cancel first — no rid yet, so _handle_abort won't call /abort_request.
        with patch("clarifai.runners.models.sglang_openai_class.requests.post") as mock_post:
            handler._handle_abort("item-1")
            mock_post.assert_not_called()
        # Later, rid arrives on first chunk — now we should abort.
        with patch("clarifai.runners.models.sglang_openai_class.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, text="ok")
            handler.register_rid("item-1", "rid-late")
            mock_post.assert_called_once_with(
                "http://host:23333/abort_request",
                json={"rid": "rid-late"},
                timeout=2,
            )

    def test_abort_request_network_failure_is_swallowed(self):
        handler = self._make_handler()
        handler.register_request("item-1")
        handler.register_rid("item-1", "rid-abc")
        with patch(
            "clarifai.runners.models.sglang_openai_class.requests.post",
            side_effect=Exception("conn refused"),
        ):
            handler._handle_abort("item-1")  # must not raise


# ---------------------------------------------------------------------------
# SGLangOpenAIModelClass — health probes
# ---------------------------------------------------------------------------
class TestSGLangOpenAIModelClassProbes:
    def test_liveness_probe_no_server_delegates_to_super(self):
        model = DummySGLangModel()
        assert model.handle_liveness_probe() is True

    def test_readiness_probe_no_server_delegates_to_super(self):
        model = DummySGLangModel()
        assert model.handle_readiness_probe() is True

    def test_liveness_probe_returns_true_on_http_200(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="localhost", port=23333)
        mock_resp = MagicMock(status_code=200)
        with patch(
            "clarifai.runners.models.sglang_openai_class.httpx.get", return_value=mock_resp
        ):
            assert model.handle_liveness_probe() is True

    def test_liveness_probe_returns_false_on_non_200(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="localhost", port=23333)
        mock_resp = MagicMock(status_code=503)
        with patch(
            "clarifai.runners.models.sglang_openai_class.httpx.get", return_value=mock_resp
        ):
            assert model.handle_liveness_probe() is False

    def test_liveness_probe_returns_false_on_exception(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="localhost", port=23333)
        with patch(
            "clarifai.runners.models.sglang_openai_class.httpx.get",
            side_effect=Exception("timeout"),
        ):
            assert model.handle_liveness_probe() is False

    def test_readiness_probe_returns_true_on_http_200(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="localhost", port=23333)
        mock_resp = MagicMock(status_code=200)
        with patch(
            "clarifai.runners.models.sglang_openai_class.httpx.get", return_value=mock_resp
        ):
            assert model.handle_readiness_probe() is True

    def test_readiness_probe_returns_false_on_exception(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="localhost", port=23333)
        with patch(
            "clarifai.runners.models.sglang_openai_class.httpx.get",
            side_effect=Exception("conn refused"),
        ):
            assert model.handle_readiness_probe() is False

    def test_probe_prefers_explicit_base_url(self):
        model = DummySGLangModel()
        model.server = MagicMock(host="wronghost", port=0)
        model.base_url = "http://right:9999"
        captured = {}

        def fake_get(url, timeout):
            captured["url"] = url
            return MagicMock(status_code=200)

        with patch("clarifai.runners.models.sglang_openai_class.httpx.get", side_effect=fake_get):
            assert model.handle_liveness_probe() is True
        assert captured["url"] == "http://right:9999/health"


# ---------------------------------------------------------------------------
# SGLangOpenAIModelClass — openai_stream_transport with cancellation
# ---------------------------------------------------------------------------
def _make_mock_stream(*chunk_texts, chunk_id="rid-123"):
    """Return a mock streaming response whose chunks expose .id (the sglang rid)."""
    chunks = []
    for text in chunk_texts:
        chunk = MagicMock()
        chunk.id = chunk_id
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


def _make_mock_responses_stream(*chunk_texts, inner_id="rid-inner"):
    """Mimic /responses streaming events where `chunk.id` is None but
    `chunk.response.id` holds the rid — exercises the fallback extraction path."""
    chunks = []
    for text in chunk_texts:
        chunk = MagicMock()
        chunk.id = None  # top-level id absent on response stream events
        chunk.response = MagicMock(id=inner_id, usage=None)
        chunk.usage = None
        chunk.model_dump_json.return_value = json.dumps(
            {"type": "response.output_text.delta", "delta": text}
        )
        chunks.append(chunk)
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(chunks))
    mock_stream.response = MagicMock()
    return mock_stream


def _make_mock_stream_raising(n_good_chunks: int, exc: Exception):
    """Yield `n_good_chunks` chunks, then raise `exc` mid-iteration."""

    def gen():
        for i in range(n_good_chunks):
            chunk = MagicMock()
            chunk.id = "rid-123"
            chunk.usage = None
            chunk.response = None
            chunk.model_dump_json.return_value = json.dumps(
                {"choices": [{"delta": {"content": f"c{i}"}}]}
            )
            yield chunk
        raise exc

    mock_stream = MagicMock()
    mock_stream.__iter__ = lambda self: gen()
    mock_stream.response = MagicMock()
    return mock_stream


class TestSGLangStreamTransportCancellation:
    def _model_with_mock_client_and_handler(self, cancel_event, chunk_id="rid-123"):
        model = DummySGLangModel()
        mock_handler = MagicMock()
        mock_handler.register_request.return_value = cancel_event
        model.cancellation_handler = mock_handler
        mock_stream = _make_mock_stream("Hello", " world", chunk_id=chunk_id)
        model.client = MagicMock()
        model.client.chat.completions.create.return_value = mock_stream
        return model, mock_handler

    def test_cancel_before_iteration_yields_no_chunks(self):
        cancel_event = threading.Event()
        cancel_event.set()
        model, mock_handler = self._model_with_mock_client_and_handler(cancel_event)

        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )
        with patch(
            "clarifai.runners.models.sglang_openai_class.get_item_id", return_value="item-abc"
        ):
            chunks = list(model.openai_stream_transport(request))

        assert chunks == []
        mock_handler.unregister_request.assert_called_once_with("item-abc")

    def test_no_cancel_yields_all_chunks_and_registers_rid(self):
        cancel_event = threading.Event()
        model, mock_handler = self._model_with_mock_client_and_handler(
            cancel_event, chunk_id="rid-from-chunk"
        )

        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }
        )
        with patch(
            "clarifai.runners.models.sglang_openai_class.get_item_id", return_value="item-xyz"
        ):
            chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 2
        mock_handler.register_request.assert_called_once()
        # rid should be captured from the first chunk and registered exactly once.
        mock_handler.register_rid.assert_called_once_with("item-xyz", "rid-from-chunk")
        mock_handler.unregister_request.assert_called_once_with("item-xyz")

    def test_stream_still_works_when_get_item_id_fails(self):
        """If get_item_id raises, item_id is None and the cancellation_handler is
        bypassed entirely — the stream should still yield all chunks."""
        model = DummySGLangModel()
        mock_handler = MagicMock()
        model.cancellation_handler = mock_handler
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
            "clarifai.runners.models.sglang_openai_class.get_item_id",
            side_effect=Exception("no context"),
        ):
            chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 1
        # With no item_id, the handler must not be touched at all.
        mock_handler.register_request.assert_not_called()
        mock_handler.register_rid.assert_not_called()
        mock_handler.unregister_request.assert_not_called()

    def test_invalid_endpoint_raises_value_error(self):
        model = DummySGLangModel()
        request = json.dumps(
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "openai_endpoint": "/unsupported",
            }
        )
        with patch(
            "clarifai.runners.models.sglang_openai_class.get_item_id", side_effect=Exception
        ):
            with pytest.raises(ValueError, match="Only"):
                list(model.openai_stream_transport(request))

    def test_responses_endpoint_streams_and_registers_rid_from_response_id(self):
        """/responses stream chunks don't have top-level .id — rid must be
        pulled from chunk.response.id via the fallback."""
        cancel_event = threading.Event()
        model = DummySGLangModel()
        mock_handler = MagicMock()
        mock_handler.register_request.return_value = cancel_event
        model.cancellation_handler = mock_handler

        mock_stream = _make_mock_responses_stream("hello", " world", inner_id="rid-from-response")
        model.client = MagicMock()
        model.client.responses.create.return_value = mock_stream

        request = json.dumps(
            {
                "model": "dummy-model",
                "input": "Hello",
                "stream": True,
                "openai_endpoint": "/responses",
            }
        )
        with patch(
            "clarifai.runners.models.sglang_openai_class.get_item_id",
            return_value="item-resp",
        ):
            chunks = list(model.openai_stream_transport(request))

        assert len(chunks) == 2
        model.client.responses.create.assert_called_once()
        model.client.chat.completions.create.assert_not_called()
        # rid pulled from chunk.response.id fallback, once.
        mock_handler.register_rid.assert_called_once_with("item-resp", "rid-from-response")
        mock_handler.unregister_request.assert_called_once_with("item-resp")

    def test_httpx_read_error_mid_stream_is_swallowed_and_unregisters(self):
        """httpx.ReadError raised during iteration is caught; finally still runs."""
        cancel_event = threading.Event()
        model = DummySGLangModel()
        mock_handler = MagicMock()
        mock_handler.register_request.return_value = cancel_event
        model.cancellation_handler = mock_handler

        mock_stream = _make_mock_stream_raising(2, httpx.ReadError("peer closed"))
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
            "clarifai.runners.models.sglang_openai_class.get_item_id",
            return_value="item-read-err",
        ):
            chunks = list(model.openai_stream_transport(request))

        # 2 chunks emitted before the error, then silent swallow.
        assert len(chunks) == 2
        mock_handler.unregister_request.assert_called_once_with("item-read-err")
