import re
import threading
import time
from typing import Iterator

import httpx
from clarifai_protocol import get_item_id, register_item_abort_callback

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger


class VLLMMetricsPoller:
    """Polls vLLM /metrics in background; caches kv_cache_usage and waiting count.

    Start in load_model() to enable admission control:

        self._metrics_poller = VLLMMetricsPoller(f"http://{host}:{port}")
    """

    KV_CACHE_REJECT_THRESHOLD = 0.95
    MAX_WAITING_REQUESTS = 10

    def __init__(self, base_url: str, poll_interval: float = 0.5):
        self.base_url = base_url
        self.poll_interval = poll_interval
        self._kv_cache = 0.0
        self._waiting = 0
        self._lock = threading.Lock()
        self._last_success = time.time()
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _poll_loop(self):
        while True:
            try:
                resp = httpx.get(f"{self.base_url}/metrics", timeout=1.0)
                if resp.status_code == 200:
                    text = resp.text
                    waiting = int(
                        self._parse(text, r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)')
                    )
                    kv_cache = self._parse(text, r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)')
                    with self._lock:
                        self._waiting = waiting
                        self._kv_cache = kv_cache
                        self._last_success = time.time()
            except Exception as e:
                logger.warning(f"[VLLMMetricsPoller] Poll failed: {e}")
            time.sleep(self.poll_interval)

    def _parse(self, text: str, pattern: str) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    def snapshot(self):
        """Return (kv_cache, waiting) atomically."""
        with self._lock:
            return self._kv_cache, self._waiting

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return time.time() - self._last_success > 5.0


class VLLMCancellationHandler:
    # Important: closing the httpx response kills the TCP connection;
    # vLLM detects is_disconnected(), triggers engine.abort() and frees KV cache.
    def __init__(self):
        self._cancel_events = {}
        self._responses = {}
        self._early_aborts = set()
        self._lock = threading.Lock()
        register_item_abort_callback(self._handle_abort)

    def _handle_abort(self, item_id: str) -> None:
        with self._lock:
            event = self._cancel_events.get(item_id)
            response = self._responses.get(item_id)
            if event:
                event.set()
            if response:
                try:
                    response.close()
                except Exception:
                    pass
            else:
                self._early_aborts.add(item_id)

    def register_request(self, item_id: str, response=None) -> threading.Event:
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[item_id] = cancel_event
            if response is not None:
                self._responses[item_id] = response
            if item_id in self._early_aborts:
                cancel_event.set()
                self._early_aborts.discard(item_id)
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
        return cancel_event

    def unregister_request(self, item_id: str) -> None:
        with self._lock:
            self._cancel_events.pop(item_id, None)
            self._responses.pop(item_id, None)
            self._early_aborts.discard(item_id)


class VLLMOpenAIModelClass(OpenAIModelClass):
    """vLLM-backed OpenAI model with /health probes and cancellation support.

    Subclasses must set client, model, server and cancellation_handler in load_model(), for example:

        def load_model(self):
            self.server = vllm_openai_server(checkpoints, **server_args)
            self.client = OpenAI(base_url=f"http://{self.server.host}:{self.server.port}/v1", api_key="x")
            self.model = self.client.models.list().data[0].id
            self.cancellation_handler = VLLMCancellationHandler()

    For cancellation in generate() or custom streaming methods, follow this pattern:

        def generate(self, prompt, ...) -> Iterator[str]:
            item_id = None
            cancel_event = None
            try:
                item_id = get_item_id()
            except Exception:
                pass
            try:
                response = self.client.chat.completions.create(..., stream=True)
                if item_id:
                    cancel_event = self.cancellation_handler.register_request(item_id, response=response.response)
                for chunk in response:
                    if cancel_event and cancel_event.is_set():
                        return
                    yield ...
            except httpx.ReadError:
                pass
            finally:
                if item_id:
                    self.cancellation_handler.unregister_request(item_id)
    """

    server = None
    cancellation_handler = None
    _metrics_poller = None

    @property
    def admission_control_backoff(self) -> float:
        """Seconds to wait before retrying after admission rejection. Override to customize."""
        return 1.0

    def check_admission(self) -> bool:
        """Fail-open: reject only when KV cache is saturated or waiting queue is too deep.

        Called by the runner before dispatching a request. Returns True to admit, False to reject.
        Admission control is disabled (always admits) when _metrics_poller is not set or is stale.
        Enable by setting self._metrics_poller in load_model():

            self._metrics_poller = VLLMMetricsPoller(f"http://{host}:{port}")
        """
        if self._metrics_poller is None or self._metrics_poller.is_stale:
            return True
        p = self._metrics_poller
        kv_cache, waiting = p.snapshot()
        if kv_cache > p.KV_CACHE_REJECT_THRESHOLD:
            logger.info(f"[AdmissionControl] REJECT kv_cache={kv_cache:.2%}")
            return False
        if waiting > p.MAX_WAITING_REQUESTS:
            logger.info(f"[AdmissionControl] REJECT waiting={waiting}")
            return False
        return True

    def handle_liveness_probe(self) -> bool:
        if self.server is None:
            return super().handle_liveness_probe()
        # /health is a non-blocking fast endpoint dedicated for health check
        try:
            resp = httpx.get(f"http://{self.server.host}:{self.server.port}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def handle_readiness_probe(self) -> bool:
        if self.server is None:
            return super().handle_readiness_probe()
        # /health is a non-blocking fast endpoint dedicated for health check
        try:
            resp = httpx.get(f"http://{self.server.host}:{self.server.port}/health", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    @OpenAIModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        from pydantic_core import from_json

        item_id = None
        try:
            item_id = get_item_id()
        except Exception:
            pass
        cancel_event = None
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError(
                    f"Only {self.ENDPOINT_CHAT_COMPLETIONS} and {self.ENDPOINT_RESPONSES} endpoints are supported for streaming."
                )

            if endpoint == self.ENDPOINT_RESPONSES:
                # /responses endpoint — direct call (no retry), same Stream[T] interface
                response_args = {**request_data}
                response_args.update({"model": self.model})
                response = self.client.responses.create(**response_args)
            else:
                # /chat/completions endpoint
                completion_args = self._create_completion_args(request_data)
                response = self.client.chat.completions.create(**completion_args)

            if item_id and self.cancellation_handler:
                cancel_event = self.cancellation_handler.register_request(
                    item_id, response=response.response
                )

            for chunk in response:
                if cancel_event and cancel_event.is_set():
                    return
                self._set_usage(chunk)
                yield chunk.model_dump_json()
        except httpx.ReadError:
            pass
        finally:
            if item_id and self.cancellation_handler:
                self.cancellation_handler.unregister_request(item_id)
