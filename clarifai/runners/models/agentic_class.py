"""Base class for creating OpenAI-compatible API server with MCP (Model Context Protocol) support."""

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import logger


@dataclass
class MCPConnection:
    """Single MCP server connection."""

    client: Any
    tools: List[Any]
    tool_names: Set[str]
    url: str
    last_used: float = field(default_factory=time.time)

    def touch(self):
        self.last_used = time.time()


class MCPConnectionPool:
    """
    Singleton, thread-safe connection pool for managing MCP server connections.
    Lifecycle:
        - The pool is implemented as a singleton. The first instantiation creates the instance;
          subsequent instantiations return the same object.
        - Initialization sets up internal data structures, a background asyncio event loop,
          and a dedicated thread for running asynchronous tasks.
        - The event loop is started in a background daemon thread and is used to run async
          operations (such as connecting and disconnecting).
    Thread Safety:
        - All access to shared state (connections, tool caches) is protected by a reentrant lock (`self._lock`).
        - The singleton instance is protected by a class-level lock (`_instance_lock`) to ensure only one instance is created.
        - The background event loop is started and accessed in a thread-safe manner.
    Cleanup Behavior:
        - Idle connections are cleaned up passively: whenever `get_connections()` is called, the pool checks for
          connections that have been idle longer than `MAX_IDLE_TIME` and disconnects them.
        - Cleanup is rate-limited by `CLEANUP_INTERVAL` to avoid excessive checks.
        - Disconnection is performed asynchronously in the background event loop.
        - Tool caches are invalidated when a connection is removed.
        - There is no explicit shutdown; background threads and event loops are daemonized and will exit with the process.
    Usage Notes:
        - Users do not need to manage the pool directly; it is managed automatically as a singleton.
        - Connections are created, reused, and cleaned up transparently.
        - The pool is safe for concurrent use from multiple threads.
    """

    _instance: Optional['MCPConnectionPool'] = None
    _instance_lock = threading.Lock()

    # Timeouts and thresholds (configurable via environment variables)
    # Default: 30s. Time to wait for a connection to be established. Increase if MCP servers are slow to respond.
    CONNECT_TIMEOUT = float(os.environ.get("CLARIFAI_MCP_CONNECT_TIMEOUT", 30.0))
    # Default: 60s. Maximum time to wait for a tool call to complete. Increase for long-running tools.
    TOOL_CALL_TIMEOUT = float(os.environ.get("CLARIFAI_MCP_TOOL_CALL_TIMEOUT", 60.0))
    # Default: 2min. Connections idle for more than this are verified before reuse.
    VERIFY_IDLE_THRESHOLD = float(os.environ.get("CLARIFAI_MCP_VERIFY_IDLE_THRESHOLD", 60 * 2))
    # Default: (15min). Connections idle for more than this are removed from the pool.
    MAX_IDLE_TIME = float(os.environ.get("CLARIFAI_MCP_MAX_IDLE_TIME", 15 * 60))
    # Default: 2min. Cleanup runs at most this often to remove idle connections.
    CLEANUP_INTERVAL = float(os.environ.get("CLARIFAI_MCP_CLEANUP_INTERVAL", 2 * 60))

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._connections: Dict[str, MCPConnection] = {}
        self._lock = threading.RLock()

        # Tool caches
        self._tool_to_url: Dict[str, str] = {}
        self._all_tools: Dict[str, dict] = {}

        # Cleanup tracking
        self._last_cleanup = 0.0

        # Background event loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._start_event_loop()

        self._initialized = True

    def _start_event_loop(self):
        """Start background event loop."""
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run, daemon=True, name="mcp_pool")
        self._loop_thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("Background event loop failed to start within 5 seconds")

    def _run_async(self, coro, timeout: float = 30.0) -> Any:
        """Run coroutine in background loop."""
        # Double-checked locking pattern to prevent race condition
        # when multiple threads try to restart a closed loop
        if self._loop is None or self._loop.is_closed():
            with self._lock:
                # Check again after acquiring lock (another thread may have started it)
                if self._loop is None or self._loop.is_closed():
                    self._start_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ==================== Cleanup Logic ====================

    def _maybe_cleanup_idle(self):
        """Passive cleanup - removes connections idle too long.

        Called at the start of get_connections() to clean up
        without needing a background thread.
        """
        now = time.time()

        # Rate limit cleanup checks
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return

        self._last_cleanup = now

        # Find idle connections
        with self._lock:
            to_remove = [
                url
                for url, conn in self._connections.items()
                if now - conn.last_used > self.MAX_IDLE_TIME
            ]

        # Remove them (outside lock to avoid deadlock during async close)
        for url in to_remove:
            self._disconnect(url)

    def _disconnect(self, url: str):
        """Disconnect and remove a connection."""
        with self._lock:
            conn = self._connections.pop(url, None)

            # Invalidate tool cache entries for this URL
            if conn:
                for tool_name in conn.tool_names:
                    self._tool_to_url.pop(tool_name, None)
                    self._all_tools.pop(tool_name, None)

        if conn:
            try:
                self._run_async(self._close_connection(conn), timeout=10.0)
                logger.info(f"Disconnected idle connection from {url}")
            except Exception as e:
                logger.warning(f"Error disconnecting from {url}: {e}")

    async def _close_connection(self, conn: MCPConnection):
        """Close a connection gracefully."""
        try:
            if hasattr(conn.client, 'close'):
                await asyncio.wait_for(conn.client.close(), timeout=5.0)
            else:
                await asyncio.wait_for(conn.client.__aexit__(None, None, None), timeout=5.0)
        except Exception as e:
            logger.warning(f"Error closing connection to {conn.url}: {e}")

    # ==================== Connection Management ====================

    async def _create_connection(self, url: str) -> MCPConnection:
        """Create new MCP connection."""
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError:
            raise ImportError("fastmcp required: pip install fastmcp")

        transport = StreamableHttpTransport(
            url=url,
            headers={"Authorization": "Bearer " + os.environ.get("CLARIFAI_PAT", "")},
        )

        client = Client(transport)
        await asyncio.wait_for(client.__aenter__(), timeout=self.CONNECT_TIMEOUT)
        tools = await asyncio.wait_for(client.list_tools(), timeout=10.0)

        return MCPConnection(
            client=client,
            tools=tools,
            tool_names={t.name for t in tools},
            url=url,
        )

    async def _verify_connection(self, conn: MCPConnection) -> bool:
        """Check if connection is still valid."""
        try:
            await asyncio.wait_for(conn.client.list_tools(), timeout=5.0)
            return True
        except Exception:
            return False

    def _needs_verification(self, conn: MCPConnection) -> bool:
        """Check if connection should be verified."""
        return time.time() - conn.last_used > self.VERIFY_IDLE_THRESHOLD

    def _update_tool_cache(self, conn: MCPConnection):
        """Cache tool info from connection."""
        for tool in conn.tools:
            self._tool_to_url[tool.name] = conn.url
            self._all_tools[tool.name] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }

    def get_connections(self, urls: List[str]) -> Dict[str, MCPConnection]:
        """Get connections for URLs, with passive cleanup."""
        # Passive cleanup of idle connections
        self._maybe_cleanup_idle()

        result = {}
        to_verify = []
        to_create = []

        # Categorize URLs
        with self._lock:
            for url in urls:
                if url in self._connections:
                    conn = self._connections[url]
                    if self._needs_verification(conn):
                        to_verify.append(url)
                    else:
                        conn.touch()
                        result[url] = conn
                else:
                    to_create.append(url)

        # Verify stale connections in parallel
        if to_verify:

            async def verify_all():
                tasks = {
                    url: self._verify_connection(self._connections[url])
                    for url in to_verify
                    if url in self._connections
                }
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                return dict(zip(tasks.keys(), results))

            try:
                verify_results = self._run_async(verify_all(), timeout=15.0)
                with self._lock:
                    for url, is_valid in verify_results.items():
                        if is_valid is True:
                            conn = self._connections[url]
                            conn.touch()
                            result[url] = conn
                        else:
                            # Invalid - remove and recreate
                            self._connections.pop(url, None)
                            to_create.append(url)
            except Exception as e:
                logger.error(f"Verification error: {e}")
                to_create.extend(to_verify)

        # Create new connections in parallel
        if to_create:

            async def create_all():
                tasks = {url: self._create_connection(url) for url in to_create}
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                return dict(zip(tasks.keys(), results))

            try:
                create_results = self._run_async(create_all(), timeout=self.CONNECT_TIMEOUT + 5)
                with self._lock:
                    for url, conn_or_error in create_results.items():
                        if isinstance(conn_or_error, Exception):
                            logger.error(f"Failed to connect to {url}: {conn_or_error}")
                        else:
                            self._connections[url] = conn_or_error
                            self._update_tool_cache(conn_or_error)
                            result[url] = conn_or_error
                            logger.info(f"✓ Connected to {url} ({len(conn_or_error.tools)} tools)")
            except Exception as e:
                logger.error(f"Connection error: {e}")

        return result

    def get_tools_and_mapping(
        self, urls: List[str]
    ) -> Tuple[List[dict], Dict[str, MCPConnection], Dict[str, str]]:
        """Get tools, connections, and mapping."""
        connections = self.get_connections(urls)

        tools = []
        tool_to_server = {}
        seen = set()

        for url, conn in connections.items():
            for tool in conn.tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    tools.append(
                        self._all_tools.get(tool.name)
                        or {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema,
                            },
                        }
                    )
                    tool_to_server[tool.name] = url

        logger.info(f"Loaded {len(tools)} tools from {len(connections)} servers")
        return tools, connections, tool_to_server

    # ==================== Tool Execution ====================

    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Call a tool asynchronously."""
        logger.info(f"Calling tool {tool_name}")
        url = tool_to_server.get(tool_name) or self._tool_to_url.get(tool_name)
        if not url or url not in connections:
            raise ValueError(f"Tool '{tool_name}' not found")

        conn = connections[url]
        result = await asyncio.wait_for(
            conn.client.call_tool(tool_name, arguments=arguments), timeout=self.TOOL_CALL_TIMEOUT
        )
        conn.touch()
        return result

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Call a tool synchronously."""
        logger.info(f"Calling tool {tool_name}")
        return self._run_async(
            self.call_tool_async(tool_name, arguments, connections, tool_to_server),
            timeout=self.TOOL_CALL_TIMEOUT + 5,
        )

    async def call_tools_batch_async(
        self,
        calls: List[Tuple[str, str, Dict[str, Any]]],  # [(id, name, args), ...]
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> List[Tuple[str, Optional[Any], Optional[str]]]:
        """Call multiple tools in parallel. Returns [(id, result, error), ...]"""

        async def call_one(call_id: str, name: str, args: Dict):
            try:
                result = await self.call_tool_async(name, args, connections, tool_to_server)
                return (call_id, result, None)
            except Exception as e:
                return (call_id, None, str(e))

        tasks = [call_one(cid, name, args) for cid, name, args in calls]
        return await asyncio.gather(*tasks)

    def call_tools_batch(
        self,
        calls: List[Tuple[str, str, Dict[str, Any]]],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> List[Tuple[str, Optional[Any], Optional[str]]]:
        """Call multiple tools in parallel (sync)."""
        return self._run_async(
            self.call_tools_batch_async(calls, connections, tool_to_server),
            timeout=self.TOOL_CALL_TIMEOUT + 10,
        )


class AgenticModelClass(OpenAIModelClass):
    """Base class for wrapping OpenAI-compatible servers with MCP (Model Context Protocol) support.

    This class extends OpenAIModelClass to enable agentic behavior by integrating LLMs with MCP servers.
    It handles tool discovery, execution, and iterative tool calling for both chat completions and
    responses endpoints, supporting both streaming and non-streaming modes.

    MCP connections are maintained persistently across requests using a connection pool, which
    significantly improves performance by avoiding reconnection overhead.

    To use this class, create a subclass and set the following class attributes:
    - client: The OpenAI-compatible client instance
    - model: The name of the model to use with the client

    Example:
        class MyAgenticModel(AgenticModelClass):
            client = OpenAI(api_key="your-key")
            model = "gpt-4"
    """

    _pool: Optional[MCPConnectionPool] = None
    _pool_lock = threading.Lock()

    @classmethod
    def get_pool(cls) -> MCPConnectionPool:
        """Get shared connection pool."""
        if cls._pool is None:
            with cls._pool_lock:
                if cls._pool is None:
                    cls._pool = MCPConnectionPool()
        return cls._pool

    # === Token Tracking ===

    def _init_tokens(self):
        if not hasattr(self._thread_local, 'tokens'):
            self._thread_local.tokens = {'prompt': 0, 'completion': 0}

    def _add_tokens(self, resp):
        """Accumulate tokens from response."""
        usage = getattr(resp, 'usage', None) or (
            getattr(resp.response, 'usage', None) if hasattr(resp, 'response') else None
        )
        if usage:
            self._init_tokens()
            self._thread_local.tokens['prompt'] += (
                getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0) or 0
            )
            self._thread_local.tokens['completion'] += (
                getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0) or 0
            )

    def _finalize_tokens(self):
        """Send accumulated tokens to output context."""
        if hasattr(self._thread_local, 'tokens'):
            t = self._thread_local.tokens
            if t['prompt'] > 0 or t['completion'] > 0:
                self.set_output_context(
                    prompt_tokens=t['prompt'], completion_tokens=t['completion']
                )
            del self._thread_local.tokens

    def _set_usage(self, resp):
        self._add_tokens(resp)

    # === Tool Format Conversion ===

    def _to_response_api_tools(self, tools: List[dict]) -> List[dict]:
        """Convert chat completion tools to response API format."""
        result = []
        for t in tools:
            if "function" in t:
                f = t["function"]
                result.append(
                    {
                        "type": "function",
                        "name": f.get("name"),
                        "description": f.get("description", ""),
                        "parameters": f.get("parameters", {}),
                    }
                )
            elif "name" in t:
                result.append(t)
        return result

    def _to_dict(self, obj) -> dict:
        """Convert object to dict."""
        if isinstance(obj, dict):
            return obj
        for attr in ('model_dump', 'dict'):
            if hasattr(obj, attr):
                return getattr(obj, attr)()
        return getattr(obj, '__dict__', {})

    def _transform_mcp_server_url(self, url: str) -> str:
        """Transform MCP server URL from old format to new API format.

        Transforms: www.clarifai.com/USER_ID/APP_ID/models/MODEL_ID
        To: https://api.clarifai.com/v2/ext/mcp/v1/users/USER_ID/apps/APP_ID/models/MODEL_ID

        Args:
            url: The MCP server URL to transform

        Returns:
            The transformed URL, or the original URL if it doesn't match the Clarifai format
        """
        try:
            user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(url)
            return f"https://api.clarifai.com/v2/ext/mcp/v1/users/{user_id}/apps/{app_id}/models/{model_id}"
        except ValueError:
            # Not a Clarifai URL format, return as-is
            return url

    def _normalize_mcp_servers(self, mcp_servers):
        """Normalize MCP server URLs to the new API format.

        Args:
            mcp_servers: Single URL string or list of URL strings

        Returns:
            Transformed URL(s) in the same format as input
        """
        if mcp_servers is None:
            return None

        if isinstance(mcp_servers, str):
            return self._transform_mcp_server_url(mcp_servers)
        elif isinstance(mcp_servers, list):
            return [self._transform_mcp_server_url(url) for url in mcp_servers]

        return mcp_servers

    # === Tool Call Parsing ===

    def _parse_chat_tool_calls(self, tool_calls) -> List[Tuple[str, str, Dict]]:
        """Parse chat completion tool calls into (id, name, args) tuples."""
        result = []
        for tc in tool_calls:
            if hasattr(tc, 'function'):
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Malformed tool arguments for tool '{getattr(tc.function, 'name', None)}': {tc.function.arguments!r}"
                    )
                    args = {}
                result.append((tc.id, tc.function.name, args))
            else:
                try:
                    args = json.loads(tc['function']['arguments'])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Malformed tool arguments for tool '{tc['function'].get('name', None)}': {tc['function']['arguments']!r}"
                    )
                    args = {}
                result.append((tc['id'], tc['function']['name'], args))
        return result

    def _parse_response_tool_calls(self, items: List[dict]) -> List[Tuple[str, str, Dict]]:
        """Parse response API tool calls into (call_id, name, args) tuples."""
        result = []
        for item in items:
            d = self._to_dict(item)
            if d.get('type') in ('function_tool_call', 'function_call', 'function', 'tool_call'):
                status = d.get('status', '')
                if status in ('pending', 'in_progress', '') or d.get('output') is None:
                    call_id = d.get('call_id') or d.get('id')
                    name = d.get('name')
                    args_str = d.get('arguments', '{}')
                    if call_id and name:
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        result.append((call_id, name, args))
        return result

    # === Tool Execution ===

    def _execute_chat_tools(
        self,
        tool_calls,
        connections: Dict[str, MCPConnection],
        messages: List[dict],
        tool_to_server: Dict[str, str],
    ):
        """Execute chat completion tool calls and append results to messages."""
        pool = self.get_pool()
        parsed = self._parse_chat_tool_calls(tool_calls)
        results = pool.call_tools_batch(parsed, connections, tool_to_server)

        for call_id, result, error in results:
            if error:
                content = f"Error: {error}"
            elif (
                hasattr(result, 'content')
                and len(result.content) > 0
                and hasattr(result.content[0], 'text')
            ):
                content = result.content[0].text
            elif len(result) > 0 and hasattr(result[0], 'text') and result[0].text:
                content = result[0].text
            else:
                content = None
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    async def _execute_chat_tools_async(
        self,
        tool_calls,
        connections: Dict[str, MCPConnection],
        messages: List[dict],
        tool_to_server: Dict[str, str],
    ):
        """Async version of chat tool execution."""
        pool = self.get_pool()
        parsed = self._parse_chat_tool_calls(tool_calls)
        results = await pool.call_tools_batch_async(parsed, connections, tool_to_server)

        for call_id, result, error in results:
            if error:
                content = f"Error: {error}"
            elif (
                hasattr(result, 'content')
                and len(result.content) > 0
                and hasattr(result.content[0], 'text')
                and result.content[0].text
            ):
                content = result.content[0].text
            elif len(result) > 0 and hasattr(result[0], 'text') and result[0].text:
                content = result[0].text
            else:
                content = None
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def _execute_response_tools(
        self,
        tool_calls: List[Tuple[str, str, Dict]],
        connections: Dict[str, MCPConnection],
        input_items: List,
        tool_to_server: Dict[str, str],
    ):
        """Execute response API tool calls and append results to input_items."""
        pool = self.get_pool()
        results = pool.call_tools_batch(tool_calls, connections, tool_to_server)

        for call_id, result, error in results:
            if error:
                output = f"Error: {error}"
            elif (
                hasattr(result, 'content')
                and len(result.content) > 0
                and hasattr(result.content[0], 'text')
                and result.content[0].text
            ):
                output = result.content[0].text
            elif len(result) > 0 and hasattr(result[0], 'text') and result[0].text:
                output = result[0].text
            else:
                output = None
            input_items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )

    async def _execute_response_tools_async(
        self,
        tool_calls: List[Tuple[str, str, Dict]],
        connections: Dict[str, MCPConnection],
        input_items: List,
        tool_to_server: Dict[str, str],
    ):
        """Async version of response API tool execution."""
        pool = self.get_pool()
        results = await pool.call_tools_batch_async(tool_calls, connections, tool_to_server)

        for call_id, result, error in results:
            if error:
                output = f"Error: {error}"
            elif (
                hasattr(result, 'content')
                and len(result.content) > 0
                and hasattr(result.content[0], 'text')
                and result.content[0].text
            ):
                output = result.content[0].text
            elif len(result) > 0 and hasattr(result[0], 'text') and result[0].text:
                output = result[0].text
            else:
                output = None
            input_items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )

    # === Response Output Processing ===

    def _convert_output_to_input(self, output: List) -> List[dict]:
        """Convert response API output items to input items."""
        result = []
        for item in output:
            d = self._to_dict(item)
            t = d.get('type')
            if t in ('message', 'reasoning'):
                result.append(d)
            elif t in ('function_tool_call', 'function_call', 'function', 'tool_call'):
                if d.get('output') is not None or d.get('status') in ('completed', 'done'):
                    result.append(d)
        return result

    # === Request Handlers ===

    def _handle_chat_completions(
        self, request_data: Dict, mcp_servers=None, connections=None, tools=None
    ):
        if mcp_servers and tools:
            request_data = {
                **request_data,
                "tools": tools,
                "tool_choice": request_data.get("tool_choice", "auto"),
            }
        return super()._handle_chat_completions(request_data)

    def _handle_responses(
        self, request_data: Dict, mcp_servers=None, connections=None, tools=None
    ):
        if mcp_servers and tools:
            request_data = {
                **request_data,
                "tools": self._to_response_api_tools(tools),
                "tool_choice": request_data.get("tool_choice", "auto"),
            }
        return super()._handle_responses(request_data)

    def _route_request(
        self, endpoint: str, request_data: Dict, mcp_servers=None, connections=None, tools=None
    ):
        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            return self._handle_chat_completions(request_data, mcp_servers, connections, tools)
        if endpoint == self.ENDPOINT_RESPONSES:
            return self._handle_responses(request_data, mcp_servers, connections, tools)
        return super()._route_request(endpoint, request_data)

    # === Streaming Helpers ===

    def _accumulate_tool_delta(self, delta, accumulated: dict):
        """Accumulate streaming tool call deltas."""
        idx = delta.index
        if idx not in accumulated:
            accumulated[idx] = {
                "id": delta.id,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        if delta.id:
            accumulated[idx]["id"] = delta.id
        if delta.function:
            if delta.function.name:
                accumulated[idx]["function"]["name"] = delta.function.name
            if delta.function.arguments:
                accumulated[idx]["function"]["arguments"] += delta.function.arguments

    def _finalize_tool_calls(self, accumulated: dict) -> List[dict]:
        """Convert accumulated tool calls to list."""
        return [
            {"id": v["id"], "type": "function", "function": v["function"]}
            for v in (accumulated[k] for k in sorted(accumulated))
        ]

    def _create_stream_request(self, messages, tools, max_tokens, temperature, top_p):
        """Create streaming chat completion request."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _async_to_sync_generator(self, async_gen_fn):
        """Bridge async generator to sync generator."""
        pool = self.get_pool()
        loop = pool._loop
        queue = asyncio.Queue()
        error_holder = [None]

        async def producer():
            try:
                async for item in async_gen_fn():
                    await queue.put(item)
            except Exception as e:
                error_holder[0] = e
            finally:
                await queue.put(None)

        asyncio.run_coroutine_threadsafe(producer(), loop)

        while True:
            future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
            item = future.result(timeout=120.0)
            if item is None:
                if error_holder[0]:
                    raise error_holder[0]
                break
            yield item

    # === Streaming with MCP ===

    async def _stream_chat_with_tools(
        self, messages, tools, connections, tool_to_server, max_tokens, temperature, top_p
    ):
        """
        Stream chat completions with MCP tool support, recursively handling tool calls.
        This method streams chat completion chunks, accumulating any tool calls generated by the model.
        If tool calls are present after streaming, it executes those tools and recursively continues
        streaming with the updated messages (including tool call results). The recursion terminates
        when no further tool calls are generated in the streamed response.
        Args:
            messages: The list of chat messages so far.
            tools: The list of available tools.
            connections: MCP tool connections.
            tool_to_server: Mapping of tool names to server URLs.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
        Yields:
            JSON-serialized chat completion chunks.
        """
        accumulated_tools = {}
        assistant_content = ""

        for chunk in self._create_stream_request(messages, tools, max_tokens, temperature, top_p):
            self._set_usage(chunk)
            yield chunk.model_dump_json()

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        self._accumulate_tool_delta(tc, accumulated_tools)
                if delta.content:
                    assistant_content += delta.content

        if accumulated_tools:
            tool_calls = self._finalize_tool_calls(accumulated_tools)
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content or None,
                    "tool_calls": tool_calls,
                }
            )
            await self._execute_chat_tools_async(tool_calls, connections, messages, tool_to_server)

            async for chunk in self._stream_chat_with_tools(
                messages, tools, connections, tool_to_server, max_tokens, temperature, top_p
            ):
                yield chunk

    async def _stream_responses_with_tools(self, request_data, tools, connections, tool_to_server):
        """
        Streams responses for the API with MCP (Model Context Protocol) tool support.
        This method processes the incoming request data, which may include user messages or input items,
        and streams back responses that may involve multiple event types, such as user messages, assistant
        responses, and tool call events. It supports both single string input and a list of message objects.
        Event Handling Flow:
            - Parses the input data into a standardized list of message items.
            - Prepares response arguments, including tool definitions and tool choice if tools are provided.
            - Accumulates output chunks as they are generated, yielding each chunk as a JSON-encoded string.
            - Handles tool call events by invoking the appropriate tools via MCP connections, and streams
              the results back as part of the response.
            - May recursively invoke itself to handle follow-up tool calls or multi-turn interactions.
        Args:
            request_data (dict): The incoming request payload, including input messages.
            tools (list): List of tool definitions available for invocation.
            connections (dict): MCP connections for tool execution.
            tool_to_server (dict): Mapping from tool names to server endpoints.
        Yields:
            str: JSON-encoded response chunks, streamed as they become available.
        """
        input_data = request_data.get("input", "")
        input_items = (
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_data}],
                }
            ]
            if isinstance(input_data, str)
            else (input_data if isinstance(input_data, list) else [])
        )

        response_args = {**request_data, "model": self.model}
        if tools:
            response_args["tools"] = self._to_response_api_tools(tools)
            response_args["tool_choice"] = response_args.get("tool_choice", "auto")

        accumulated_output = []
        tool_calls_by_id = {}
        msg_index_map = {}

        for chunk in self.client.responses.create(**response_args):
            self._set_usage(chunk)
            chunk_type = getattr(chunk, 'type', '') or chunk.__class__.__name__

            # Track message indices for filtering
            if chunk_type in (
                'response.output_item.added',
                'ResponseOutputItemAddedEvent',
            ) and hasattr(chunk, 'item'):
                item_dict = self._to_dict(chunk.item)
                if item_dict.get('type') == 'message' and hasattr(chunk, 'output_index'):
                    msg_index_map[chunk.output_index] = len(msg_index_map)
                elif item_dict.get('type') in (
                    'function_tool_call',
                    'function_call',
                    'function',
                    'tool_call',
                ):
                    item_id = item_dict.get('id') or item_dict.get('call_id')
                    if item_id:
                        tool_calls_by_id[item_id] = {
                            'id': item_id,
                            'call_id': item_dict.get('call_id'),
                            'type': item_dict.get('type'),
                            'name': item_dict.get('name', ''),
                            'arguments': item_dict.get('arguments', ''),
                            'status': 'in_progress',
                        }

            # Accumulate arguments
            elif chunk_type in (
                'response.function_call_arguments.delta',
                'ResponseFunctionCallArgumentsDeltaEvent',
            ):
                item_id = getattr(chunk, 'item_id', None)
                if item_id and item_id in tool_calls_by_id:
                    tool_calls_by_id[item_id]['arguments'] += getattr(chunk, 'delta', '')

            elif chunk_type in (
                'response.function_call_arguments.done',
                'ResponseFunctionCallArgumentsDoneEvent',
            ):
                item_id = getattr(chunk, 'item_id', None)
                if item_id and item_id in tool_calls_by_id:
                    tool_calls_by_id[item_id]['arguments'] = getattr(chunk, 'arguments', '')

            # Mark tool call complete
            elif chunk_type in (
                'response.output_item.done',
                'ResponseOutputItemDoneEvent',
            ) and hasattr(chunk, 'item'):
                item_dict = self._to_dict(chunk.item)
                item_type = item_dict.get('type')
                if item_type in ('function_tool_call', 'function_call', 'function', 'tool_call'):
                    item_id = item_dict.get('id')
                    if item_id and item_id in tool_calls_by_id:
                        tool_calls_by_id[item_id]['status'] = 'completed'
                        if 'call_id' in item_dict:
                            tool_calls_by_id[item_id]['call_id'] = item_dict['call_id']
                        accumulated_output.append(tool_calls_by_id[item_id])
                else:
                    accumulated_output.append(item_dict)

            # Handle completed response - filter to messages only
            elif chunk_type in ('response.completed', 'ResponseCompletedEvent') and hasattr(
                chunk, 'response'
            ):
                resp = chunk.response
                if hasattr(resp, 'output') and resp.output:
                    filtered = [
                        self._to_dict(i)
                        for i in resp.output
                        if self._to_dict(i).get('type') == 'message'
                    ]
                    resp_dict = self._to_dict(resp)
                    resp_dict['output'] = filtered
                    yield json.dumps(
                        {
                            'type': 'response.completed',
                            'sequence_number': getattr(chunk, 'sequence_number', None),
                            'response': resp_dict,
                        }
                    )
                    continue

            # Yield message-related chunks with remapped indices
            should_yield = True
            if hasattr(chunk, 'output_index'):
                if chunk.output_index not in msg_index_map:
                    should_yield = False
                else:
                    chunk_dict = self._to_dict(chunk)
                    chunk_dict['output_index'] = msg_index_map[chunk.output_index]
                    yield json.dumps(chunk_dict)
                    continue

            if should_yield and chunk_type not in (
                'response.function_call_arguments.delta',
                'ResponseFunctionCallArgumentsDeltaEvent',
                'response.function_call_arguments.done',
                'ResponseFunctionCallArgumentsDoneEvent',
            ):
                item = getattr(chunk, 'item', None)
                if item:
                    if self._to_dict(item).get('type') not in (
                        'function_tool_call',
                        'function_call',
                        'function',
                        'tool_call',
                    ):
                        yield chunk.model_dump_json()
                else:
                    yield chunk.model_dump_json()

        # Add any remaining tool calls
        for tc in tool_calls_by_id.values():
            if tc.get('name') and tc['id'] not in {
                self._to_dict(o).get('id') for o in accumulated_output
            }:
                accumulated_output.append(tc)

        # Execute tool calls if any
        tool_calls = self._parse_response_tool_calls(accumulated_output)
        if tool_calls:
            input_items.extend(self._convert_output_to_input(accumulated_output))
            await self._execute_response_tools_async(
                tool_calls, connections, input_items, tool_to_server
            )
            request_data['input'] = input_items

            async for chunk in self._stream_responses_with_tools(
                request_data, tools, connections, tool_to_server
            ):
                yield chunk

    # === Main OpenAI Methods ===

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Handle non-streaming OpenAI requests."""
        try:
            data = from_json(msg)
            data = self._update_old_fields(data)
            mcp_servers = data.pop("mcp_servers", None)
            mcp_servers = self._normalize_mcp_servers(mcp_servers)
            endpoint = data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if mcp_servers and data.get("tools") is None:
                pool = self.get_pool()
                tools, connections, tool_to_server = pool.get_tools_and_mapping(mcp_servers)

                if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                    response = self._route_request(endpoint, data, mcp_servers, connections, tools)
                    while (
                        response.choices
                        and len(response.choices) > 0
                        and hasattr(response.choices[0], 'message')
                        and hasattr(response.choices[0].message, 'tool_calls')
                        and response.choices[0].message.tool_calls
                        and len(response.choices[0].message.tool_calls) > 0
                    ):
                        messages = data.get("messages", [])
                        messages.append(response.choices[0].message)
                        self._execute_chat_tools(
                            response.choices[0].message.tool_calls,
                            connections,
                            messages,
                            tool_to_server,
                        )
                        data["messages"] = messages
                        response = self._route_request(
                            endpoint, data, mcp_servers, connections, tools
                        )

                elif endpoint == self.ENDPOINT_RESPONSES:
                    response = self._route_request(endpoint, data, mcp_servers, connections, tools)

                    input_data = data.get("input", "")
                    input_items = (
                        [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": input_data}],
                            }
                        ]
                        if isinstance(input_data, str)
                        else (input_data if isinstance(input_data, list) else [])
                    )

                    output = response.output if hasattr(response, 'output') else []
                    tool_calls = self._parse_response_tool_calls(output)

                    while tool_calls:
                        input_items.extend(self._convert_output_to_input(output))
                        self._execute_response_tools(
                            tool_calls, connections, input_items, tool_to_server
                        )
                        data["input"] = input_items
                        response = self._route_request(
                            endpoint, data, mcp_servers, connections, tools
                        )
                        output = response.output if hasattr(response, 'output') else []
                        tool_calls = self._parse_response_tool_calls(output)
                else:
                    response = self._route_request(endpoint, data)
            else:
                response = self._route_request(endpoint, data)

            self._finalize_tokens()
            return response.model_dump_json()

        except Exception as e:
            logger.exception(e)
            return to_json(
                {
                    "code": status_code_pb2.MODEL_PREDICTION_FAILED,
                    "description": "Model prediction failed",
                    "details": str(e),
                }
            )

    @ModelClass.method
    def openai_stream_transport(self, msg: str) -> Iterator[str]:
        """Handle streaming OpenAI requests."""
        try:
            data = from_json(msg)
            data = self._update_old_fields(data)
            mcp_servers = data.pop("mcp_servers", None)
            mcp_servers = self._normalize_mcp_servers(mcp_servers)
            endpoint = data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if endpoint not in (self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES):
                raise ValueError("Streaming only for chat completions and responses")

            if mcp_servers and data.get("tools") is None:
                pool = self.get_pool()
                tools, connections, tool_to_server = pool.get_tools_and_mapping(mcp_servers)

                if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                    yield from self._async_to_sync_generator(
                        lambda: self._stream_chat_with_tools(
                            data.get("messages", []),
                            tools,
                            connections,
                            tool_to_server,
                            data.get("max_completion_tokens", 4096),
                            data.get("temperature", 1.0),
                            data.get("top_p", 1.0),
                        )
                    )
                else:
                    yield from self._async_to_sync_generator(
                        lambda: self._stream_responses_with_tools(
                            data, tools, connections, tool_to_server
                        )
                    )

                self._finalize_tokens()
                return

            # Non-MCP streaming
            if endpoint == self.ENDPOINT_RESPONSES:
                for chunk in self.client.responses.create(**{**data, "model": self.model}):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
            else:
                for chunk in self.client.chat.completions.create(
                    **self._create_completion_args(data)
                ):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()

            self._finalize_tokens()

        except Exception as e:
            logger.exception(e)
            yield to_json(
                {
                    "code": status_code_pb2.MODEL_PREDICTION_FAILED,
                    "description": "Model prediction failed",
                    "details": str(e),
                }
            )
