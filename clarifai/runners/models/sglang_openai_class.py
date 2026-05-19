import threading
from typing import Iterator

import httpx
import requests
from clarifai_protocol import get_item_id, register_item_abort_callback

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger


class SGLangCancellationHandler:
    # Important: in addition to closing the httpx response (which kills the TCP
    # connection and lets sglang detect the disconnect), we also POST the captured
    # request id (rid) to sglang's /abort_request endpoint. This frees the KV cache
    # immediately instead of waiting for the engine to notice the disconnect.
    def __init__(self, base_url: str):
        self._cancel_events = {}
        self._responses = {}
        self._rids = {}
        self._early_aborts = set()
        self._lock = threading.Lock()
        self._base_url = base_url
        register_item_abort_callback(self._handle_abort)

    def _call_abort_request(self, rid: str) -> None:
        try:
            resp = requests.post(f"{self._base_url}/abort_request", json={"rid": rid}, timeout=2)
            logger.info(
                f"[SGLangCancellationHandler] /abort_request rid={rid} "
                f"status={resp.status_code} body={resp.text}"
            )
        except Exception as e:
            logger.warning(f"[SGLangCancellationHandler] /abort_request failed: {e}")

    def _handle_abort(self, item_id: str) -> None:
        rid = None
        with self._lock:
            event = self._cancel_events.get(item_id)
            response = self._responses.get(item_id)
            rid = self._rids.get(item_id)
            if event:
                event.set()
            if response:
                try:
                    response.close()
                except Exception:
                    pass
            else:
                self._early_aborts.add(item_id)
        # Call outside the lock to avoid holding it during network I/O.
        if rid:
            self._call_abort_request(rid)

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

    def register_rid(self, item_id: str, rid: str) -> None:
        """Register the sglang request id once captured from the first chunk.
        If the request was already cancelled before the rid was known, issue
        /abort_request now so the engine frees the KV cache immediately.
        """
        should_abort = False
        with self._lock:
            if item_id in self._cancel_events:
                self._rids[item_id] = rid
                if self._cancel_events[item_id].is_set():
                    should_abort = True
        if should_abort:
            self._call_abort_request(rid)

    def unregister_request(self, item_id: str) -> None:
        with self._lock:
            self._cancel_events.pop(item_id, None)
            self._responses.pop(item_id, None)
            self._rids.pop(item_id, None)
            self._early_aborts.discard(item_id)


class SGLangOpenAIModelClass(OpenAIModelClass):
    """SGLang-backed OpenAI model with /health probes and cancellation support.

    Subclasses must set client, model, server, base_url and cancellation_handler in
    load_model(), for example:

        def load_model(self):
            self.server = sglang_openai_server(checkpoints, **server_args)
            self.base_url = f"http://{self.server.host}:{self.server.port}"
            self.client = OpenAI(base_url=f"{self.base_url}/v1", api_key="x")
            self.model = self.client.models.list().data[0].id
            self.cancellation_handler = SGLangCancellationHandler(self.base_url)

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
                    cancel_event = self.cancellation_handler.register_request(
                        item_id, response=response.response
                    )
                rid_registered = False
                for chunk in response:
                    if item_id and not rid_registered:
                        rid = getattr(chunk, 'id', None)
                        if rid:
                            self.cancellation_handler.register_rid(item_id, rid)
                            rid_registered = True
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
    base_url = None
    cancellation_handler = None

    def _health_url(self) -> str:
        if self.base_url:
            return f"{self.base_url}/health"
        return f"http://{self.server.host}:{self.server.port}/health"

    def handle_liveness_probe(self) -> bool:
        if self.server is None:
            return super().handle_liveness_probe()
        try:
            resp = httpx.get(self._health_url(), timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def handle_readiness_probe(self) -> bool:
        if self.server is None:
            return super().handle_readiness_probe()
        try:
            resp = httpx.get(self._health_url(), timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    @OpenAIModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        from pydantic_core import from_json

        try:
            item_id = get_item_id()
        except Exception:
            item_id = None

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
                response_args = {**request_data}
                response_args.update({"model": self.model})
                response = self.client.responses.create(**response_args)
            else:
                completion_args = self._create_completion_args(request_data)
                response = self.client.chat.completions.create(**completion_args)

            if item_id and self.cancellation_handler:
                cancel_event = self.cancellation_handler.register_request(
                    item_id, response=response.response
                )

            rid_registered = False
            for chunk in response:
                if item_id and self.cancellation_handler and not rid_registered:
                    rid = getattr(chunk, 'id', None) or getattr(
                        getattr(chunk, 'response', None), 'id', None
                    )
                    if rid:
                        self.cancellation_handler.register_rid(item_id, rid)
                        rid_registered = True
                if cancel_event and cancel_event.is_set():
                    return
                self._set_usage(chunk)
                yield chunk.model_dump_json()
        except httpx.ReadError:
            pass
        finally:
            if item_id and self.cancellation_handler:
                self.cancellation_handler.unregister_request(item_id)
