"""MCP model base class with a *single* long‑lived FastMCP client.

The implementation creates a **background thread + event‑loop** that owns
the FastMCP client. All incoming MCP calls are forwarded to that loop via
``asyncio.run_coroutine_threadsafe`` and the result is returned synchronously.
"""

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

from clarifai.runners.models.model_class import ModelClass

if TYPE_CHECKING:  # pragma: no cover
    from fastmcp import Client, FastMCP
    from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)


class MCPModelClass(ModelClass):
    """
    Base class for wrapping a FastMCP server as a Clarifai model.

    Sub‑classes must implement :meth:`get_server` and return a ready‑to‑use
    ``FastMCP`` instance.
    """

    def __init__(self):
        super().__init__()
        self._fastmcp_server: Optional["FastMCP"] = None

        # FastMCP client that talks to the server (created inside the background loop)
        self._client: Optional["Client"] = None
        self._client_session: Optional["ClientSession"] = None

        # Background thread / loop handling
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._initialized = threading.Event()
        self._init_error: Optional[Exception] = None

    def load_model(self):
        """Called by Clarifai to initialize the model. Starts the background loop."""
        self._start_background_loop()

        # Wait for initialization to complete (with timeout)
        if not self._initialized.wait(timeout=60):
            raise RuntimeError("Background MCP initialization timed out")

        if self._init_error is not None:
            raise self._init_error

    def get_server(self) -> "FastMCP":
        """Required method for each subclass to implement to return the FastMCP server to use."""
        raise NotImplementedError("Subclasses must implement get_server() method")

    def _start_background_loop(self) -> None:
        """Spin up a daemon thread that runs its own asyncio event‑loop."""

        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            try:
                # Run the initialisation coroutine
                init_coro = self._background_initialise()
                self._loop.run_until_complete(init_coro)
            except Exception as e:
                self._init_error = e
                self._initialized.set()
                return

            self._initialized.set()

            # Keep the loop alive until we are asked to stop.
            self._loop.run_forever()

            # Clean‑up when loop stops
            self._loop.run_until_complete(self._background_shutdown())
            self._loop.close()
            logger.debug("Background MCP thread stopped")

        self._thread = threading.Thread(target=runner, name="MCP-background-loop", daemon=True)
        self._thread.start()
        logger.debug("Background MCP thread started")

    async def _background_initialise(self) -> None:
        """
        Create the FastMCP server and client.
        All objects are bound to the *same* event‑loop (the background loop).
        """
        try:
            from fastmcp import Client
        except ImportError:
            raise ImportError(
                "fastmcp package is required to use MCP functionality. "
                "Install it with: pip install fastmcp"
            )

        # Create FastMCP server (this triggers lifespan which may do tool discovery)
        self._fastmcp_server = self.get_server()

        # Create FastMCP client
        self._client = Client(self._fastmcp_server)
        await self._client.__aenter__()
        self._client_session = self._client.session

        logger.debug("Background MCP initialisation complete")

    async def _background_shutdown(self) -> None:
        """Clean up resources when shutting down."""
        # Close FastMCP client
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error while closing FastMCP client")
            self._client = None
            self._client_session = None

        logger.debug("Background MCP shutdown complete")

    def _run_in_background(self, coro) -> Any:
        """
        Schedule *coro* on the background loop and block until it finishes.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Background event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)  # 5 minute timeout

    async def _bg_send_request(self, payload: dict) -> Any:
        """
        Runs inside the background loop. Forwards the request to FastMCP client.
        """
        from mcp import types
        from mcp.shared.exceptions import McpError

        msg_id = payload.get("id", "")

        raw_msg = types.ClientRequest.model_validate(payload)
        clean_dict = raw_msg.model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
            exclude={"jsonrpc", "id"},
        )
        client_message = types.ClientRequest.model_validate(clean_dict)

        # Determine the expected result type
        result_type = self._get_result_type(client_message)

        if result_type is None:
            # Special case: InitializeRequest
            if isinstance(client_message.root, types.InitializeRequest):
                return await self._client_session.initialize()
            raise NotImplementedError(
                f"Method {getattr(client_message, 'method', 'unknown')} not implemented"
            )

        try:
            return await self._client_session.send_request(client_message, result_type)
        except McpError as e:
            return types.JSONRPCError(jsonrpc="2.0", id=msg_id, error=e.error)

    def _get_result_type(self, client_message):
        """Map request type to result type."""
        from mcp import types

        type_map = {
            types.PingRequest: types.EmptyResult,
            types.InitializeRequest: None,  # Special handling
            types.SetLevelRequest: types.EmptyResult,
            types.ListResourcesRequest: types.ListResourcesResult,
            types.ListResourceTemplatesRequest: types.ListResourceTemplatesResult,
            types.ReadResourceRequest: types.ReadResourceResult,
            types.SubscribeRequest: types.EmptyResult,
            types.UnsubscribeRequest: types.EmptyResult,
            types.ListPromptsRequest: types.ListPromptsResult,
            types.GetPromptRequest: types.GetPromptResult,
            types.CompleteRequest: types.CompleteResult,
            types.ListToolsRequest: types.ListToolsResult,
            types.CallToolRequest: types.CallToolResult,
        }

        for req_type, res_type in type_map.items():
            if isinstance(client_message.root, req_type):
                return res_type
        return None

    async def _bg_send_notification(self, payload: dict) -> None:
        """
        Runs inside the background loop. Forwards notification to FastMCP client.
        """
        from mcp import types
        from mcp.shared.exceptions import McpError

        raw_msg = types.ClientNotification.model_validate(payload)
        clean_dict = raw_msg.model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
            exclude={"jsonrpc"},
        )
        client_message = types.ClientNotification.model_validate(clean_dict)

        try:
            await self._client_session.send_notification(client_message)
        except McpError:
            logger.exception("Error while sending notification to FastMCP")

    @ModelClass.method
    def mcp_transport(self, msg: str) -> str:
        """
        Synchronous entry point used by Clarifai.
        """
        from mcp import types

        payload = json.loads(msg)

        if not payload.get("method", "").startswith("notifications/"):
            # Normal request – we need a response.
            result = self._run_in_background(self._bg_send_request(payload))

            if result is None:
                result = types.JSONRPCError(
                    jsonrpc="2.0",
                    id=payload.get("id", ""),
                    error=types.ErrorData(
                        code=types.INTERNAL_ERROR,
                        message="Empty response from MCP server.",
                    ),
                )
            return result.model_dump_json(by_alias=True, exclude_none=True)
        else:
            # Notification – fire‑and‑forget
            self._run_in_background(self._bg_send_notification(payload))
            return "{}"

    def shutdown(self) -> None:
        """Stop the background thread and close everything."""
        if self._loop is None:
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._loop = None
        self._thread = None
        logger.info("MCP bridge shut down")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
