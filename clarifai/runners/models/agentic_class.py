"""Base class for creating OpenAI-compatible API server with MCP (Model Context Protocol) support."""

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger


@dataclass
class MCPConnection:
    """Represents a single MCP server connection with metadata."""

    client: Any
    tools: List[Any]
    tool_names: Set[str]  # For O(1) tool lookup
    last_used: float
    connected_at: float
    url: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    use_count: int = 0

    def mark_used(self):
        """Mark connection as recently used."""
        self.last_used = time.time()
        self.use_count += 1


class MCPConnectionPool:
    """Thread-safe connection pool for MCP servers with persistent connections."""

    _instance: Optional['MCPConnectionPool'] = None
    _lock = threading.Lock()

    # Pool configuration
    DEFAULT_MAX_IDLE_TIME = 600  # 10 minutes, if mcp server idle time > DEFAULT_MAX_IDLE_TIME it disconnect from server
    DEFAULT_CLEANUP_INTERVAL = 120  # 2 minute,
    DEFAULT_VERIFY_THRESHOLD = 60  # Only verify if idle > 60 seconds
    DEFAULT_CONNECT_TIMEOUT = 30  # Connection timeout
    DEFAULT_TOOL_CALL_TIMEOUT = 60  # Tool call timeout
    MAX_PARALLEL_CONNECTIONS = 10  # Max concurrent connection attempts

    def __new__(cls):
        """Singleton pattern to ensure one connection pool per process."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_idle_time: float = DEFAULT_MAX_IDLE_TIME,
        cleanup_interval: float = DEFAULT_CLEANUP_INTERVAL,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
    ):
        if self._initialized:
            return

        self._connections: Dict[str, MCPConnection] = {}
        self._global_lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        self._loop_ready = threading.Event()

        # Configuration
        self._max_idle_time = max_idle_time
        self._cleanup_interval = cleanup_interval
        self._verify_threshold = verify_threshold
        self._last_cleanup = time.time()

        # Tool name to URL cache for O(1) lookup
        self._tool_to_url_cache: Dict[str, str] = {}
        self._cache_lock = threading.RLock()

        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_PARALLEL_CONNECTIONS, thread_name_prefix="mcp_pool_"
        )

        # Start the background event loop immediately
        self._start_event_loop()

        self._initialized = True

    def _start_event_loop(self):
        """Start the persistent event loop in a background thread."""
        with self._loop_lock:
            if self._loop is not None and self._loop_thread is not None:
                if self._loop_thread.is_alive() and not self._loop.is_closed():
                    return

            self._loop_ready.clear()

            def run_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop_ready.set()
                self._loop.run_forever()

            self._loop_thread = threading.Thread(
                target=run_loop, daemon=True, name="mcp_event_loop"
            )
            self._loop_thread.start()

            if not self._loop_ready.wait(timeout=10.0):
                raise RuntimeError("Failed to start MCP event loop")

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the persistent event loop, starting it if necessary."""
        if self._loop is None or self._loop.is_closed():
            self._start_event_loop()
        return self._loop

    def _run_coroutine(self, coro, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> Any:
        """Run a coroutine in the persistent event loop with timeout."""
        loop = self._get_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def _run_coroutine_nowait(self, coro) -> asyncio.Future:
        """Schedule a coroutine without waiting for result."""
        loop = self._get_event_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def _create_client(self, url: str) -> Tuple[Any, List[Any]]:
        """Create and connect a single MCP client.

        Returns:
            Tuple of (client, tools)
        """
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError:
            raise ImportError(
                "fastmcp package is required to use MCP functionality. "
                "Install it with: pip install fastmcp"
            )

        transport = StreamableHttpTransport(
            url=url,
            headers={"Authorization": "Bearer " + os.environ.get("CLARIFAI_PAT", "")},
        )

        client = Client(transport)
        await client.__aenter__()
        tools = await client.list_tools()

        return client, tools

    async def _connect_single_server(
        self, url: str, max_retries: int = 2, retry_delay: float = 1.0
    ) -> Optional[MCPConnection]:
        """Connect to a single MCP server with retries."""
        last_error = None

        for attempt in range(max_retries):
            try:
                client, tools = await asyncio.wait_for(
                    self._create_client(url), timeout=self.DEFAULT_CONNECT_TIMEOUT
                )

                # Build tool name set for O(1) lookup
                tool_names = {tool.name for tool in tools}

                connection = MCPConnection(
                    client=client,
                    tools=tools,
                    tool_names=tool_names,
                    last_used=time.time(),
                    connected_at=time.time(),
                    url=url,
                )

                logger.info(f"✓ Connected to {url} with {len(tools)} tools")
                return connection

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Connection timeout after {self.DEFAULT_CONNECT_TIMEOUT}s"
                )
                logger.warning(
                    f"⚠ Timeout connecting to {url} (attempt {attempt + 1}/{max_retries})"
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠ Failed to connect to {url} (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)

        logger.error(f"❌ Failed to connect to {url} after {max_retries} attempts: {last_error}")
        return None

    async def _connect_servers_parallel(
        self, urls: List[str], max_retries: int = 2, retry_delay: float = 1.0
    ) -> Dict[str, MCPConnection]:
        """Connect to multiple servers in parallel."""
        if not urls:
            return {}

        tasks = [self._connect_single_server(url, max_retries, retry_delay) for url in urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        connections = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error connecting to {url}: {result}")
            elif result is not None:
                connections[url] = result

        return connections

    async def _verify_connection_async(self, connection: MCPConnection) -> bool:
        """Verify that a connection is still valid."""
        try:
            await asyncio.wait_for(connection.client.list_tools(), timeout=5.0)
            return True
        except Exception as e:
            logger.warning(f"⚠ Connection to {connection.url} is no longer valid: {e}")
            return False

    async def _disconnect_async(self, connection: MCPConnection):
        """Disconnect from a single MCP server."""
        try:
            client = connection.client
            if hasattr(client, 'close') and callable(getattr(client, 'close', None)):
                if asyncio.iscoroutinefunction(client.close):
                    await asyncio.wait_for(client.close(), timeout=5.0)
                else:
                    client.close()
            else:
                await asyncio.wait_for(client.__aexit__(None, None, None), timeout=5.0)
            logger.info(f"✓ Disconnected from {connection.url}")
        except Exception as e:
            logger.warning(f"⚠ Error disconnecting from {connection.url}: {e}")

    def _should_verify_connection(self, connection: MCPConnection) -> bool:
        """Check if connection needs verification based on idle time."""
        idle_time = time.time() - connection.last_used
        return idle_time > self._verify_threshold

    def _update_tool_cache(self, connections: Dict[str, MCPConnection]):
        """Update the tool-to-URL cache from connections."""
        with self._cache_lock:
            for url, conn in connections.items():
                for tool_name in conn.tool_names:
                    self._tool_to_url_cache[tool_name] = url

    def _invalidate_tool_cache(self, url: str):
        """Remove tools from cache when a connection is removed."""
        with self._cache_lock:
            self._tool_to_url_cache = {
                name: cached_url
                for name, cached_url in self._tool_to_url_cache.items()
                if cached_url != url
            }

    def get_connections(
        self, mcp_servers: List[str], max_retries: int = 2, retry_delay: float = 1.0
    ) -> Dict[str, MCPConnection]:
        """Get connections for the specified MCP servers.

        Uses lazy verification - only verifies connections that have been
        idle longer than the verification threshold.
        """
        self._maybe_cleanup_stale_connections()

        result = {}
        urls_to_connect = []
        urls_to_verify = []

        # First pass: categorize URLs
        with self._global_lock:
            for url in mcp_servers:
                if url in self._connections:
                    connection = self._connections[url]
                    if self._should_verify_connection(connection):
                        urls_to_verify.append(url)
                    else:
                        # Recently used, assume still valid
                        connection.mark_used()
                        result[url] = connection
                else:
                    urls_to_connect.append(url)

        # Verify stale connections in parallel
        if urls_to_verify:

            async def verify_all():
                tasks = []
                for url in urls_to_verify:
                    conn = self._connections.get(url)
                    if conn:
                        tasks.append(self._verify_connection_async(conn))
                    else:
                        tasks.append(asyncio.coroutine(lambda: False)())
                return await asyncio.gather(*tasks, return_exceptions=True)

            try:
                verify_results = self._run_coroutine(verify_all(), timeout=15.0)

                with self._global_lock:
                    for url, is_valid in zip(urls_to_verify, verify_results):
                        if isinstance(is_valid, Exception) or not is_valid:
                            # Connection is stale, need to reconnect
                            if url in self._connections:
                                self._invalidate_tool_cache(url)
                                del self._connections[url]
                            urls_to_connect.append(url)
                        else:
                            # Connection is valid
                            connection = self._connections[url]
                            connection.mark_used()
                            result[url] = connection
            except Exception as e:
                logger.error(f"❌ Error verifying connections: {e}")
                # On verification failure, try to reconnect all
                urls_to_connect.extend(urls_to_verify)

        # Connect to new servers in parallel
        if urls_to_connect:
            try:
                new_connections = self._run_coroutine(
                    self._connect_servers_parallel(urls_to_connect, max_retries, retry_delay),
                    timeout=self.DEFAULT_CONNECT_TIMEOUT * max_retries + 10,
                )

                with self._global_lock:
                    self._connections.update(new_connections)
                    result.update(new_connections)

                # Update tool cache
                self._update_tool_cache(new_connections)

            except Exception as e:
                logger.error(f"❌ Error connecting to MCP servers: {e}")

        return result

    def get_tools_and_mapping(
        self, mcp_servers: List[str]
    ) -> Tuple[List[dict], Dict[str, MCPConnection], Dict[str, str]]:
        """Get tools and server mapping for the specified MCP servers.

        Returns:
            Tuple of (tools in OpenAI format, connections dict, tool_to_server mapping)
        """
        connections = self.get_connections(mcp_servers)

        all_tools = []
        tool_to_server = {}
        seen_tools = set()  # Avoid duplicate tools

        for url, conn in connections.items():
            for tool in conn.tools:
                if tool.name not in seen_tools:
                    seen_tools.add(tool.name)
                    all_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema,
                            },
                        }
                    )
                    tool_to_server[tool.name] = url

        logger.info(f"Access to {len(all_tools)} tools from {len(connections)} servers")
        return all_tools, connections, tool_to_server

    def get_cached_tool_url(self, tool_name: str) -> Optional[str]:
        """Get URL for a tool from cache. O(1) lookup."""
        with self._cache_lock:
            return self._tool_to_url_cache.get(tool_name)

    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Call a tool on the appropriate MCP server."""
        # Try cached/mapped server first
        server_url = tool_to_server.get(tool_name) or self.get_cached_tool_url(tool_name)

        if server_url and server_url in connections:
            conn = connections[server_url]
            with conn.lock:
                try:
                    logger.info(f"Calling tool {tool_name} on {server_url}")
                    result = await asyncio.wait_for(
                        conn.client.call_tool(tool_name, arguments=arguments),
                        timeout=self.DEFAULT_TOOL_CALL_TIMEOUT,
                    )
                    conn.mark_used()
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout calling tool {tool_name}")
                    raise
                except Exception as e:
                    logger.error(f"❌ Error calling tool {tool_name}: {e}")
                    # Fall through to try other servers

        # Fallback: find server with this tool
        for url, conn in connections.items():
            if url == server_url:
                continue  # Already tried
            if tool_name in conn.tool_names:
                with conn.lock:
                    try:
                        logger.info(f"Calling tool {tool_name} on {url} (fallback)")
                        result = await asyncio.wait_for(
                            conn.client.call_tool(tool_name, arguments=arguments),
                            timeout=self.DEFAULT_TOOL_CALL_TIMEOUT,
                        )
                        conn.mark_used()
                        # Update cache for future lookups
                        with self._cache_lock:
                            self._tool_to_url_cache[tool_name] = url
                        return result
                    except Exception as e:
                        logger.error(f"❌ Error calling tool {tool_name} on {url}: {e}")
                        continue

        raise Exception(f"Tool {tool_name} not found on any connected server")

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Synchronous wrapper for call_tool_async."""
        return self._run_coroutine(
            self.call_tool_async(tool_name, arguments, connections, tool_to_server),
            timeout=self.DEFAULT_TOOL_CALL_TIMEOUT + 5,
        )

    async def call_tools_parallel(
        self,
        tool_calls: List[Tuple[str, Dict[str, Any]]],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ) -> List[Tuple[str, Any, Optional[Exception]]]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            connections: Connection dictionary
            tool_to_server: Tool to server mapping

        Returns:
            List of (tool_name, result, exception) tuples
        """

        async def call_single(tool_name: str, args: Dict[str, Any]):
            try:
                result = await self.call_tool_async(tool_name, args, connections, tool_to_server)
                return (tool_name, result, None)
            except Exception as e:
                return (tool_name, None, e)

        tasks = [call_single(name, args) for name, args in tool_calls]
        return await asyncio.gather(*tasks)

    def _maybe_cleanup_stale_connections(self):
        """Clean up connections that have been idle for too long."""
        current_time = time.time()

        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time

        with self._global_lock:
            urls_to_remove = [
                url
                for url, conn in self._connections.items()
                if current_time - conn.last_used > self._max_idle_time
            ]

        for url in urls_to_remove:
            self._disconnect_url(url)

    def _disconnect_url(self, url: str):
        """Disconnect from a specific URL."""
        with self._global_lock:
            connection = self._connections.pop(url, None)

        if connection:
            self._invalidate_tool_cache(url)
            try:
                self._run_coroutine(self._disconnect_async(connection), timeout=10.0)
            except Exception as e:
                logger.warning(f"⚠ Error during disconnect from {url}: {e}")

    def disconnect(self, url: str):
        """Public method to disconnect from a specific MCP server."""
        self._disconnect_url(url)

    def disconnect_all(self):
        """Disconnect from all MCP servers and cleanup."""
        with self._global_lock:
            urls = list(self._connections.keys())

        # Disconnect all in parallel
        async def disconnect_all_async():
            tasks = []
            for url in urls:
                conn = self._connections.get(url)
                if conn:
                    tasks.append(self._disconnect_async(conn))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            self._run_coroutine(disconnect_all_async(), timeout=30.0)
        except Exception as e:
            logger.warning(f"⚠ Error during bulk disconnect: {e}")

        with self._global_lock:
            self._connections.clear()

        with self._cache_lock:
            self._tool_to_url_cache.clear()

        # Stop the event loop
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread is not None:
                    self._loop_thread.join(timeout=5.0)
                self._loop = None
                self._loop_thread = None

        # Shutdown executor
        self._executor.shutdown(wait=False)

    def warm_up(self, mcp_servers: List[str]):
        """Pre-establish connections to servers.

        Call this during initialization to avoid connection latency on first request.
        """
        logger.info(f"Warming up connections to {len(mcp_servers)} MCP servers")
        self.get_connections(mcp_servers)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring."""
        with self._global_lock:
            connections_info = []
            for url, conn in self._connections.items():
                connections_info.append(
                    {
                        "url": url,
                        "tools_count": len(conn.tools),
                        "use_count": conn.use_count,
                        "idle_seconds": time.time() - conn.last_used,
                        "connected_seconds": time.time() - conn.connected_at,
                    }
                )

            return {
                "total_connections": len(self._connections),
                "cached_tools": len(self._tool_to_url_cache),
                "connections": connections_info,
            }


class AgenticModelClass(OpenAIModelClass):
    """Base class for wrapping OpenAI-compatible servers with MCP support.

    Optimizations over base implementation:
    - Persistent connection pool across requests
    - Parallel tool execution
    - Tool name caching for O(1) lookup
    - Lazy connection verification
    - Efficient streaming with queue-based async bridging
    """

    _mcp_pool: Optional[MCPConnectionPool] = None
    _pool_lock = threading.Lock()

    @classmethod
    def _get_mcp_pool(cls) -> MCPConnectionPool:
        """Get or create the MCP connection pool singleton."""
        if cls._mcp_pool is None:
            with cls._pool_lock:
                if cls._mcp_pool is None:
                    cls._mcp_pool = MCPConnectionPool()
        return cls._mcp_pool

    @classmethod
    def warm_up_mcp(cls, mcp_servers: List[str]):
        """Pre-establish MCP connections during model initialization."""
        pool = cls._get_mcp_pool()
        pool.warm_up(mcp_servers)

    def _get_mcp_tools_and_clients(
        self, mcp_servers: List[str]
    ) -> Tuple[List[dict], Dict[str, MCPConnection], Dict[str, str]]:
        """Get available tools and clients from MCP servers."""
        pool = self._get_mcp_pool()
        return pool.get_tools_and_mapping(mcp_servers)

    def _init_token_accumulation(self):
        """Initialize token accumulation for a new request."""
        if not hasattr(self._thread_local, 'accumulated_tokens'):
            self._thread_local.accumulated_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

    def _accumulate_usage(self, resp):
        """Accumulate token usage from response object."""
        has_usage = getattr(resp, "usage", None)
        has_response_usage = getattr(resp, "response", None) and getattr(
            resp.response, "usage", None
        )

        if has_response_usage or has_usage:
            prompt_tokens = 0
            completion_tokens = 0
            if has_usage:
                prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) or getattr(
                    resp.usage, "input_tokens", 0
                )
                completion_tokens = getattr(resp.usage, "completion_tokens", 0) or getattr(
                    resp.usage, "output_tokens", 0
                )
            else:
                prompt_tokens = getattr(resp.response.usage, "input_tokens", 0)
                completion_tokens = getattr(resp.response.usage, "output_tokens", 0)

            prompt_tokens = prompt_tokens or 0
            completion_tokens = completion_tokens or 0

            if prompt_tokens > 0 or completion_tokens > 0:
                self._init_token_accumulation()
                self._thread_local.accumulated_tokens['prompt_tokens'] += prompt_tokens
                self._thread_local.accumulated_tokens['completion_tokens'] += completion_tokens

    def _finalize_token_usage(self):
        """Finalize token accumulation and set the total in output context."""
        if hasattr(self._thread_local, 'accumulated_tokens'):
            prompt_tokens = self._thread_local.accumulated_tokens['prompt_tokens']
            completion_tokens = self._thread_local.accumulated_tokens['completion_tokens']

            if prompt_tokens > 0 or completion_tokens > 0:
                self.set_output_context(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            del self._thread_local.accumulated_tokens

    def _set_usage(self, resp):
        """Override _set_usage to accumulate tokens across multiple API calls."""
        self._accumulate_usage(resp)

    def _handle_chat_completions(
        self,
        request_data: Dict[str, Any],
        mcp_servers: List[str] = None,
        mcp_clients: dict = None,
        tools: List[dict] = None,
    ):
        """Handle chat completion requests with optional MCP tool support."""
        if mcp_servers and tools:
            request_data = request_data.copy()
            request_data["tools"] = tools
            request_data["tool_choice"] = request_data.get("tool_choice", "auto")
        return super()._handle_chat_completions(request_data)

    def _convert_tools_to_response_api_format(self, tools: List[dict]) -> List[dict]:
        """Convert tools from chat completion format to response API format."""
        response_api_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "function")
                if "function" in tool:
                    func = tool["function"]
                    response_api_tools.append(
                        {
                            "type": tool_type,
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    )
                elif "name" in tool:
                    response_api_tools.append(tool)
        return response_api_tools

    def _handle_responses(
        self,
        request_data: Dict[str, Any],
        mcp_servers: List[str] = None,
        mcp_clients: dict = None,
        tools: List[dict] = None,
    ):
        """Handle response API requests with optional MCP tool support."""
        if mcp_servers and tools:
            request_data = request_data.copy()
            response_api_tools = self._convert_tools_to_response_api_format(tools)
            request_data["tools"] = response_api_tools
            request_data["tool_choice"] = request_data.get("tool_choice", "auto")
        return super()._handle_responses(request_data)

    def _route_request(
        self,
        endpoint: str,
        request_data: Dict[str, Any],
        mcp_servers: List[str] = None,
        mcp_clients: dict = None,
        tools: List[dict] = None,
    ):
        """Route the request to appropriate handler based on endpoint."""
        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            return self._handle_chat_completions(request_data, mcp_servers, mcp_clients, tools)
        if endpoint == self.ENDPOINT_RESPONSES:
            return self._handle_responses(request_data, mcp_servers, mcp_clients, tools)
        return super()._route_request(endpoint, request_data)

    def _execute_tool_calls(
        self,
        tool_calls: List[Any],
        connections: Dict[str, MCPConnection],
        messages: List[dict],
        tool_to_server: Dict[str, str],
    ):
        """Execute tool calls from chat completion and add results to messages."""
        pool = self._get_mcp_pool()

        # Prepare tool calls for potential parallel execution
        parsed_calls = []
        for tool_call in tool_calls:
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            else:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                tool_id = tool_call['id']
            parsed_calls.append((tool_id, tool_name, tool_args))

        # Execute tools (could be parallelized for independent tools)
        for tool_id, tool_name, tool_args in parsed_calls:
            try:
                result = pool.call_tool(tool_name, tool_args, connections, tool_to_server)
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            except Exception as e:
                content = f"Error: {str(e)}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content,
                }
            )

    async def _execute_tool_calls_async(
        self,
        tool_calls: List[Any],
        connections: Dict[str, MCPConnection],
        messages: List[dict],
        tool_to_server: Dict[str, str],
    ):
        """Async version with parallel tool execution support."""
        pool = self._get_mcp_pool()

        # Parse all tool calls
        parsed_calls = []
        for tool_call in tool_calls:
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            else:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                tool_id = tool_call['id']
            parsed_calls.append((tool_id, tool_name, tool_args))

        # Execute all tools in parallel
        tool_inputs = [(name, args) for _, name, args in parsed_calls]
        results = await pool.call_tools_parallel(tool_inputs, connections, tool_to_server)

        # Map results back to tool IDs
        for (tool_id, tool_name, _), (_, result, error) in zip(parsed_calls, results):
            if error:
                content = f"Error: {str(error)}"
            else:
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content,
                }
            )

    def _execute_response_api_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        connections: Dict[str, MCPConnection],
        input_items: List[Any],
        tool_to_server: Dict[str, str],
    ):
        """Execute tool calls from response API and add results to input items."""
        pool = self._get_mcp_pool()

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args_str = tool_call.get("arguments", "{}")
            tool_id = tool_call.get("id")
            call_id = tool_call.get("call_id")

            try:
                tool_args = (
                    json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                )
            except json.JSONDecodeError:
                tool_args = {}

            try:
                result = pool.call_tool(tool_name, tool_args, connections, tool_to_server)
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            except Exception as e:
                content = f"Error: {str(e)}"

            output_call_id = call_id or tool_id
            if not output_call_id:
                logger.warning(f"⚠ No call_id or id found for tool {tool_name}, skipping")
                continue

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": output_call_id,
                    "output": content,
                }
            )

    async def _execute_response_api_tool_calls_async(
        self,
        tool_calls: List[Dict[str, Any]],
        connections: Dict[str, MCPConnection],
        input_items: List[Any],
        tool_to_server: Dict[str, str],
    ):
        """Async version with parallel tool execution support."""
        pool = self._get_mcp_pool()

        # Parse all tool calls
        parsed_calls = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args_str = tool_call.get("arguments", "{}")
            tool_id = tool_call.get("id")
            call_id = tool_call.get("call_id")

            try:
                tool_args = (
                    json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                )
            except json.JSONDecodeError:
                tool_args = {}

            parsed_calls.append((tool_name, tool_args, tool_id, call_id))

        # Execute all tools in parallel
        tool_inputs = [(name, args) for name, args, _, _ in parsed_calls]
        results = await pool.call_tools_parallel(tool_inputs, connections, tool_to_server)

        # Map results back
        for (tool_name, _, tool_id, call_id), (_, result, error) in zip(parsed_calls, results):
            if error:
                content = f"Error: {str(error)}"
            else:
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )

            output_call_id = call_id or tool_id
            if not output_call_id:
                logger.warning(f"⚠ No call_id or id found for tool {tool_name}, skipping")
                continue

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": output_call_id,
                    "output": content,
                }
            )

    def _extract_tool_calls_from_response_output(
        self, response_output: List[Any]
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from response API output array."""
        tool_calls = []
        for item in response_output:
            if not isinstance(item, dict):
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                elif hasattr(item, '__dict__'):
                    item = item.__dict__
                else:
                    continue

            item_type = item.get("type")
            if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                status = item.get("status", "")
                output = item.get("output")
                if status in ["pending", "in_progress", ""] or output is None:
                    tool_calls.append(item)
        return tool_calls

    def _convert_output_items_to_input_items(
        self, response_output: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert response API output items to input items format."""
        input_items = []
        for item in response_output:
            if not isinstance(item, dict):
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                elif hasattr(item, '__dict__'):
                    item = item.__dict__
                else:
                    continue

            item_type = item.get("type")
            if item_type in ["message", "reasoning"]:
                input_items.append(item)
            elif item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                status = item.get("status", "")
                output = item.get("output")
                if output is not None or status in ["completed", "done"]:
                    input_items.append(item)
        return input_items

    def _accumulate_tool_call_delta(self, tool_call_delta, tool_calls_accumulated: dict):
        """Accumulate tool call data from a streaming delta."""
        index = tool_call_delta.index
        if index not in tool_calls_accumulated:
            tool_calls_accumulated[index] = {
                "id": tool_call_delta.id,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        if tool_call_delta.id:
            tool_calls_accumulated[index]["id"] = tool_call_delta.id
        if tool_call_delta.function:
            if tool_call_delta.function.name:
                tool_calls_accumulated[index]["function"]["name"] = tool_call_delta.function.name
            if tool_call_delta.function.arguments:
                tool_calls_accumulated[index]["function"]["arguments"] += (
                    tool_call_delta.function.arguments
                )

    def _convert_accumulated_tool_calls(self, tool_calls_accumulated: dict) -> List[dict]:
        """Convert accumulated tool calls dictionary to list format."""
        return [
            {
                "id": tc["id"],
                "type": tc["type"],
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for tc in (
                tool_calls_accumulated[idx] for idx in sorted(tool_calls_accumulated.keys())
            )
        ]

    def _create_completion_request(
        self,
        messages: List[dict],
        tools: List[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool = False,
    ):
        """Create a completion request with common parameters."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return self.client.chat.completions.create(**kwargs)

    def _bridge_async_generator(self, async_gen_func):
        """Bridge an async generator to a sync generator using efficient queue-based approach."""
        pool = self._get_mcp_pool()
        loop = pool._get_event_loop()

        # Use a bounded queue to apply backpressure
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        done = threading.Event()
        exception_holder: List[Optional[Exception]] = [None]

        async def producer():
            try:
                async for item in async_gen_func():
                    await queue.put(item)
            except Exception as e:
                exception_holder[0] = e
            finally:
                await queue.put(None)  # Sentinel
                done.set()

        # Start producer
        asyncio.run_coroutine_threadsafe(producer(), loop)

        try:
            while True:
                # Get from queue with timeout
                try:
                    get_future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
                    item = get_future.result(timeout=60.0)
                except Exception:
                    if exception_holder[0]:
                        raise exception_holder[0]
                    raise

                if item is None:
                    break
                yield item

            if exception_holder[0]:
                raise exception_holder[0]
        finally:
            done.wait(timeout=1.0)

    async def _stream_with_mcp_tools_json(
        self,
        openai_messages: List[dict],
        tools: List[dict],
        connections: Dict[str, MCPConnection],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tool_to_server: Dict[str, str],
    ):
        """Async generator for streaming chat completions with MCP tools."""
        tool_calls_accumulated = {}
        streaming_response = ""

        stream = self._create_completion_request(
            openai_messages, tools, max_tokens, temperature, top_p, stream=True
        )

        for chunk in stream:
            self._set_usage(chunk)
            yield chunk.model_dump_json()

            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        self._accumulate_tool_call_delta(tool_call_delta, tool_calls_accumulated)
                if delta.content:
                    streaming_response += delta.content

        if tool_calls_accumulated:
            tool_calls_list = self._convert_accumulated_tool_calls(tool_calls_accumulated)
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": streaming_response or None,
                    "tool_calls": tool_calls_list,
                }
            )

            # Execute tools in parallel
            await self._execute_tool_calls_async(
                tool_calls_list, connections, openai_messages, tool_to_server
            )

            async for chunk_json in self._stream_with_mcp_tools_json(
                openai_messages, tools, connections, max_tokens, temperature, top_p, tool_to_server
            ):
                yield chunk_json

    async def _stream_responses_with_mcp_tools_json(
        self,
        request_data: Dict[str, Any],
        tools: List[dict],
        connections: Dict[str, MCPConnection],
        tool_to_server: Dict[str, str],
    ):
        """Async generator for streaming response API with MCP tools."""
        input_data = request_data.get("input", "")
        if isinstance(input_data, str):
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_data}],
                }
            ]
        else:
            input_items = input_data if isinstance(input_data, list) else []

        response_args = {**request_data, "model": self.model}
        if tools:
            response_api_tools = self._convert_tools_to_response_api_format(tools)
            response_args["tools"] = response_api_tools
            response_args["tool_choice"] = response_args.get("tool_choice", "auto")

        stream = self.client.responses.create(**response_args)
        accumulated_output = []
        tool_calls_accumulated = {}
        original_to_filtered_index_map = {}

        for chunk in stream:
            self._set_usage(chunk)
            chunk_type = getattr(chunk, 'type', None) or chunk.__class__.__name__

            should_yield = True
            item_to_check = None

            # Process chunk based on type (condensed for brevity - same logic as before)
            if chunk_type in (
                'response.output_item.added',
                'ResponseOutputItemAddedEvent',
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item
                if hasattr(chunk, 'output_index'):
                    item_dict = self._to_dict(item_to_check)
                    if item_dict.get("type") == "message":
                        original_to_filtered_index_map[chunk.output_index] = len(
                            original_to_filtered_index_map
                        )

                    # Track tool calls
                    if item_dict.get("type") in [
                        "function_tool_call",
                        "function_call",
                        "function",
                        "tool_call",
                    ]:
                        item_id = item_dict.get("id") or item_dict.get("call_id")
                        if item_id:
                            tool_calls_accumulated[item_id] = {
                                "id": item_id,
                                "call_id": item_dict.get("call_id"),
                                "type": item_dict.get("type"),
                                "name": item_dict.get("name", ""),
                                "arguments": item_dict.get("arguments", ""),
                                "status": item_dict.get("status", "in_progress"),
                            }

            elif chunk_type in (
                'response.output_item.done',
                'ResponseOutputItemDoneEvent',
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item

            elif hasattr(chunk, 'output_index'):
                if chunk.output_index not in original_to_filtered_index_map:
                    should_yield = False

            # Filter non-message items
            if item_to_check:
                item_dict = self._to_dict(item_to_check)
                if item_dict.get("type") != "message":
                    should_yield = False

            # Handle response.completed
            if chunk_type in ('response.completed', 'ResponseCompletedEvent') and hasattr(
                chunk, 'response'
            ):
                response = chunk.response
                if hasattr(response, 'output') and response.output:
                    filtered_output, accumulated_output = self._process_response_output(
                        response.output, accumulated_output, tool_calls_accumulated
                    )

                    response_dict = self._to_dict(response)
                    response_dict["output"] = filtered_output

                    yield json.dumps(
                        {
                            "type": "response.completed",
                            "sequence_number": getattr(chunk, 'sequence_number', None),
                            "response": response_dict,
                        }
                    )
                else:
                    yield chunk.model_dump_json()
            elif should_yield:
                if (
                    hasattr(chunk, 'output_index')
                    and chunk.output_index in original_to_filtered_index_map
                ):
                    chunk_dict = self._to_dict(chunk)
                    chunk_dict["output_index"] = original_to_filtered_index_map[chunk.output_index]
                    yield json.dumps(chunk_dict)
                elif not hasattr(chunk, 'output_index'):
                    yield chunk.model_dump_json()

            # Handle argument deltas
            if chunk_type in (
                'response.function_call_arguments.delta',
                'ResponseFunctionCallArgumentsDeltaEvent',
            ):
                item_id = getattr(chunk, 'item_id', None)
                if item_id and item_id in tool_calls_accumulated:
                    tool_calls_accumulated[item_id]["arguments"] += getattr(chunk, 'delta', '')

            elif chunk_type in (
                'response.function_call_arguments.done',
                'ResponseFunctionCallArgumentsDoneEvent',
            ):
                item_id = getattr(chunk, 'item_id', None)
                if item_id and item_id in tool_calls_accumulated:
                    tool_calls_accumulated[item_id]["arguments"] = getattr(chunk, 'arguments', '')

            # Handle output item done
            elif chunk_type in (
                'response.output_item.done',
                'ResponseOutputItemDoneEvent',
            ) and hasattr(chunk, 'item'):
                item_dict = self._to_dict(chunk.item)
                item_type = item_dict.get("type")

                if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                    item_id = item_dict.get("id")
                    if item_id and item_id in tool_calls_accumulated:
                        tool_calls_accumulated[item_id]["status"] = item_dict.get(
                            "status", "completed"
                        )
                        if "call_id" in item_dict:
                            tool_calls_accumulated[item_id]["call_id"] = item_dict.get("call_id")
                        accumulated_output.append(tool_calls_accumulated[item_id])
                    else:
                        accumulated_output.append(item_dict)
                else:
                    accumulated_output.append(item_dict)

        # Add remaining accumulated tool calls
        for call_id, call_data in tool_calls_accumulated.items():
            if call_data.get("name"):
                existing_ids = {self._get_id(i) for i in accumulated_output}
                if call_id not in existing_ids:
                    accumulated_output.append(call_data)

        # Execute tool calls if any
        tool_calls = self._extract_tool_calls_from_response_output(accumulated_output)
        if tool_calls:
            model_output_items = self._convert_output_items_to_input_items(accumulated_output)
            input_items.extend(model_output_items)

            # Execute tools in parallel
            await self._execute_response_api_tool_calls_async(
                tool_calls, connections, input_items, tool_to_server
            )

            request_data["input"] = input_items

            async for chunk_json in self._stream_responses_with_mcp_tools_json(
                request_data, tools, connections, tool_to_server
            ):
                yield chunk_json

    def _to_dict(self, obj: Any) -> dict:
        """Convert object to dictionary."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return {}

    def _get_id(self, item: Any) -> Optional[str]:
        """Get ID from an item."""
        if isinstance(item, dict):
            return item.get("id")
        return getattr(item, "id", None)

    def _process_response_output(
        self,
        output: List[Any],
        accumulated_output: List[Dict],
        tool_calls_accumulated: Dict[str, Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process response output, filtering messages and accumulating tool calls."""
        filtered_output = []

        for item in output:
            item_dict = self._to_dict(item)
            item_type = item_dict.get("type")

            if item_type == "message":
                filtered_output.append(item_dict)
            elif item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                item_id = item_dict.get("id")
                existing_ids = {self._get_id(i) for i in accumulated_output}
                if not item_id or item_id not in existing_ids:
                    accumulated_output.append(item_dict)
            else:
                item_id = item_dict.get("id")
                existing_ids = {self._get_id(i) for i in accumulated_output}
                if not item_id or item_id not in existing_ids:
                    accumulated_output.append(item_dict)

        return filtered_output, accumulated_output

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Process an OpenAI-compatible request."""
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            tools = request_data.get("tools")

            if mcp_servers and len(mcp_servers) > 0 and tools is None:
                logger.info(f"Getting tools for MCP servers: {mcp_servers}")
                tools_local, connections, tool_to_server = self._get_mcp_tools_and_clients(
                    mcp_servers
                )

                if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                    response = self._route_request(
                        endpoint, request_data, mcp_servers, connections, tools_local
                    )

                    while response.choices and response.choices[0].message.tool_calls:
                        messages = request_data.get("messages", [])
                        messages.append(response.choices[0].message)
                        self._execute_tool_calls(
                            response.choices[0].message.tool_calls,
                            connections,
                            messages,
                            tool_to_server,
                        )
                        request_data["messages"] = messages
                        response = self._route_request(
                            endpoint, request_data, mcp_servers, connections, tools_local
                        )

                elif endpoint == self.ENDPOINT_RESPONSES:
                    response = self._route_request(
                        endpoint, request_data, mcp_servers, connections, tools_local
                    )

                    input_data = request_data.get("input", "")
                    if isinstance(input_data, str):
                        input_items = [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": input_data}],
                            }
                        ]
                    else:
                        input_items = input_data if isinstance(input_data, list) else []

                    response_output = response.output if hasattr(response, 'output') else []
                    tool_calls = self._extract_tool_calls_from_response_output(response_output)

                    while tool_calls:
                        model_output_items = self._convert_output_items_to_input_items(
                            response_output
                        )
                        input_items.extend(model_output_items)
                        self._execute_response_api_tool_calls(
                            tool_calls, connections, input_items, tool_to_server
                        )
                        request_data["input"] = input_items
                        response = self._route_request(
                            endpoint, request_data, mcp_servers, connections, tools_local
                        )
                        response_output = response.output if hasattr(response, 'output') else []
                        tool_calls = self._extract_tool_calls_from_response_output(response_output)
                else:
                    response = self._route_request(endpoint, request_data)
            else:
                response = self._route_request(endpoint, request_data)

            self._finalize_token_usage()
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
        """Process an OpenAI-compatible request with streaming."""
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming only supported for chat completions and responses.")

            if mcp_servers and len(mcp_servers) > 0 and request_data.get("tools") is None:
                logger.info(f"Getting tools for MCP servers: {mcp_servers}")
                tools_local, connections, tool_to_server = self._get_mcp_tools_and_clients(
                    mcp_servers
                )

                async def stream_generator():
                    if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                        messages = request_data.get("messages", [])
                        async for chunk_json in self._stream_with_mcp_tools_json(
                            messages,
                            tools_local,
                            connections,
                            request_data.get("max_completion_tokens", 4096),
                            request_data.get("temperature", 1.0),
                            request_data.get("top_p", 1.0),
                            tool_to_server,
                        ):
                            yield chunk_json
                        self._finalize_token_usage()
                    elif endpoint == self.ENDPOINT_RESPONSES:
                        async for chunk_json in self._stream_responses_with_mcp_tools_json(
                            request_data, tools_local, connections, tool_to_server
                        ):
                            yield chunk_json
                        self._finalize_token_usage()

                yield from self._bridge_async_generator(stream_generator)
                return

            # Non-MCP path
            if endpoint == self.ENDPOINT_RESPONSES:
                response_args = {**request_data, "model": self.model}
                for chunk in self.client.responses.create(**response_args):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
            else:
                completion_args = self._create_completion_args(request_data)
                for chunk in self.client.chat.completions.create(**completion_args):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()

            self._finalize_token_usage()

        except Exception as e:
            logger.exception(e)
            yield to_json(
                {
                    "code": status_code_pb2.MODEL_PREDICTION_FAILED,
                    "description": "Model prediction failed",
                    "details": str(e),
                }
            )
