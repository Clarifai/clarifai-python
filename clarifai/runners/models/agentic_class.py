"""Base class for creating OpenAI-compatible API server with MCP (Model Context Protocol) support."""

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger


class MCPConnectionPool:
    """Thread-safe connection pool for MCP servers with persistent connections.

    This class manages MCP client connections across multiple requests,
    maintaining persistent connections and handling reconnection when needed.
    """

    _instance: Optional['MCPConnectionPool'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure one connection pool per process."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._connections: Dict[
            str, Dict[str, Any]
        ] = {}  # url -> {client, tools, loop, last_used, lock}
        self._connection_locks: Dict[str, threading.Lock] = {}  # url -> lock for that connection
        self._global_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        self._max_idle_time = 300  # 5 minutes idle timeout
        self._cleanup_interval = 60  # Check for stale connections every minute
        self._last_cleanup = time.time()
        self._initialized = True

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create a persistent event loop running in a background thread.

        This ensures MCP connections persist across request boundaries even when
        the request's event loop is closed.
        """
        with self._loop_lock:
            # Check if we have a running loop
            if self._loop is not None and self._loop_thread is not None:
                if self._loop_thread.is_alive() and not self._loop.is_closed():
                    return self._loop

            # Create a new event loop in a background thread
            loop_ready = threading.Event()

            def run_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                loop_ready.set()
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()
            loop_ready.wait(timeout=5.0)  # Wait for loop to be ready

            if self._loop is None:
                raise RuntimeError("Failed to create event loop for MCP connections")

            return self._loop

    def _run_coroutine(self, coro) -> Any:
        """Run a coroutine in the persistent event loop.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        loop = self._get_or_create_event_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=30.0)  # 30 second timeout for operations

    def _get_connection_lock(self, url: str) -> threading.Lock:
        """Get or create a lock for a specific URL."""
        with self._global_lock:
            if url not in self._connection_locks:
                self._connection_locks[url] = threading.Lock()
            return self._connection_locks[url]

    async def _connect_single_server(
        self, url: str, max_retries: int = 2, retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Connect to a single MCP server with retries.

        Args:
            url: MCP server URL
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Dictionary with client and tools, or None if connection failed
        """
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError:
            raise ImportError(
                "fastmcp package is required to use MCP functionality. "
                "Install it with: pip install fastmcp"
            )

        last_error = None

        for attempt in range(max_retries):
            try:
                transport = StreamableHttpTransport(
                    url=url,
                    headers={"Authorization": "Bearer " + os.environ.get("CLARIFAI_PAT", "")},
                )

                client = Client(transport)
                await client.__aenter__()

                # List available tools
                tools_result = await client.list_tools()

                logger.info(f"✓ Connected to {url} with {len(tools_result)} tools")

                return {
                    "client": client,
                    "tools": tools_result,
                    "last_used": time.time(),
                    "connected_at": time.time(),
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠ Failed to connect to {url} (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"❌ Failed to connect to {url} after {max_retries} attempts: {e}"
                    )

        return None

    async def _verify_connection(self, url: str, connection_info: Dict[str, Any]) -> bool:
        """Verify that a connection is still valid.

        Args:
            url: MCP server URL
            connection_info: Connection info dictionary

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            client = connection_info["client"]
            # Try to list tools as a health check
            await asyncio.wait_for(client.list_tools(), timeout=5.0)
            return True
        except Exception as e:
            logger.warning(f"⚠ Connection to {url} is no longer valid: {e}")
            return False

    async def _disconnect_single_server(self, url: str, connection_info: Dict[str, Any]):
        """Disconnect from a single MCP server.

        Args:
            url: MCP server URL
            connection_info: Connection info dictionary
        """
        try:
            client = connection_info["client"]
            if hasattr(client, 'close') and callable(getattr(client, 'close', None)):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
            else:
                await client.__aexit__(None, None, None)
            logger.info(f"✓ Disconnected from {url}")
        except Exception as e:
            logger.warning(f"⚠ Error disconnecting from {url}: {e}")

    def get_connections(
        self, mcp_servers: List[str], max_retries: int = 2, retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """Get connections for the specified MCP servers.

        This method reuses existing connections when possible and creates
        new ones as needed. Thread-safe.

        Args:
            mcp_servers: List of MCP server URLs
            max_retries: Maximum retry attempts for new connections
            retry_delay: Delay between retries

        Returns:
            Dictionary mapping server URLs to client info and tools
        """
        # Periodic cleanup of stale connections
        self._maybe_cleanup_stale_connections()

        result = {}
        urls_to_connect = []

        # First pass: get existing valid connections
        for url in mcp_servers:
            lock = self._get_connection_lock(url)
            with lock:
                if url in self._connections:
                    connection_info = self._connections[url]
                    # Check if connection is still valid
                    try:
                        is_valid = self._run_coroutine(
                            self._verify_connection(url, connection_info)
                        )
                        if is_valid:
                            connection_info["last_used"] = time.time()
                            result[url] = connection_info
                            logger.debug(f"Reusing existing connection to {url}")
                            continue
                        else:
                            # Connection is stale, remove it
                            del self._connections[url]
                    except Exception as e:
                        logger.warning(f"⚠ Error verifying connection to {url}: {e}")
                        # Remove potentially stale connection
                        if url in self._connections:
                            del self._connections[url]

                urls_to_connect.append(url)

        # Second pass: connect to servers that need new connections
        if urls_to_connect:

            async def connect_servers():
                tasks = []
                for url in urls_to_connect:
                    tasks.append(self._connect_single_server(url, max_retries, retry_delay))
                return await asyncio.gather(*tasks, return_exceptions=True)

            try:
                results = self._run_coroutine(connect_servers())

                for url, connection_result in zip(urls_to_connect, results):
                    if isinstance(connection_result, Exception):
                        logger.error(f"❌ Failed to connect to {url}: {connection_result}")
                        continue
                    if connection_result is not None:
                        lock = self._get_connection_lock(url)
                        with lock:
                            self._connections[url] = connection_result
                            result[url] = connection_result
            except Exception as e:
                logger.error(f"❌ Error connecting to MCP servers: {e}")

        return result

    def get_tools_and_mapping(
        self, mcp_servers: List[str]
    ) -> tuple[List[dict], Dict[str, Any], Dict[str, str]]:
        """Get tools and server mapping for the specified MCP servers.

        Args:
            mcp_servers: List of MCP server URLs

        Returns:
            Tuple of (tools in OpenAI format, mcp_clients dictionary, tool_to_server mapping)
        """
        mcp_clients = self.get_connections(mcp_servers)

        all_tools = []
        tool_to_server = {}

        for mcp_url, server_info in mcp_clients.items():
            tools = server_info.get("tools", [])
            for tool in tools:
                tool_name = tool.name
                all_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": f"{tool.description}",
                            "parameters": tool.inputSchema,
                        },
                    }
                )
                tool_to_server[tool_name] = mcp_url

        logger.info(f"Access to {len(all_tools)} tools from {len(mcp_clients)} servers")
        return all_tools, mcp_clients, tool_to_server

    def _maybe_cleanup_stale_connections(self):
        """Clean up connections that have been idle for too long."""
        current_time = time.time()

        # Only run cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = current_time
        urls_to_remove = []

        with self._global_lock:
            for url, connection_info in self._connections.items():
                last_used = connection_info.get("last_used", 0)
                if current_time - last_used > self._max_idle_time:
                    urls_to_remove.append(url)

        for url in urls_to_remove:
            self.disconnect(url)

    def disconnect(self, url: str):
        """Disconnect from a specific MCP server.

        Args:
            url: MCP server URL to disconnect from
        """
        lock = self._get_connection_lock(url)
        with lock:
            if url in self._connections:
                connection_info = self._connections.pop(url)
                try:
                    self._run_coroutine(self._disconnect_single_server(url, connection_info))
                except Exception as e:
                    logger.warning(f"⚠ Error during disconnect from {url}: {e}")

    def disconnect_all(self):
        """Disconnect from all MCP servers."""
        with self._global_lock:
            urls = list(self._connections.keys())

        for url in urls:
            self.disconnect(url)

        # Stop the event loop
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread is not None:
                    self._loop_thread.join(timeout=5.0)
                self._loop = None
                self._loop_thread = None

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        mcp_clients: Dict[str, Any],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Call a tool on the appropriate MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            mcp_clients: Dictionary of MCP clients
            tool_to_server: Mapping of tool names to server URLs

        Returns:
            Tool call result
        """

        async def _call_tool():
            result = None
            error_msg = None

            # Try the mapped server first
            if tool_to_server and tool_name in tool_to_server:
                server_url = tool_to_server[tool_name]
                if server_url in mcp_clients:
                    try:
                        logger.info(f"Calling tool {tool_name} with arguments {arguments}")
                        result = await mcp_clients[server_url]["client"].call_tool(
                            tool_name, arguments=arguments
                        )
                        return result
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ Error calling tool {tool_name}: {e}")

            # Fallback: try all servers
            for server_url, server_info in mcp_clients.items():
                if tool_to_server and tool_name in tool_to_server:
                    if tool_to_server[tool_name] == server_url:
                        continue  # Already tried this one
                try:
                    logger.info(f"Calling tool {tool_name} with arguments {arguments}")
                    result = await server_info["client"].call_tool(tool_name, arguments=arguments)
                    return result
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"❌ Error calling tool {tool_name}: {e}")
                    continue

            raise Exception(
                f"Failed to execute tool {tool_name}. "
                f"{error_msg if error_msg else 'Tool not found on any server.'}"
            )

        return self._run_coroutine(_call_tool())

    async def call_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        mcp_clients: Dict[str, Any],
        tool_to_server: Dict[str, str],
    ) -> Any:
        """Async version of call_tool for use within async contexts.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            mcp_clients: Dictionary of MCP clients
            tool_to_server: Mapping of tool names to server URLs

        Returns:
            Tool call result
        """
        result = None
        error_msg = None

        # Try the mapped server first
        if tool_to_server and tool_name in tool_to_server:
            server_url = tool_to_server[tool_name]
            if server_url in mcp_clients:
                try:
                    logger.info(f"Calling tool {tool_name} with arguments {arguments}")
                    result = await mcp_clients[server_url]["client"].call_tool(
                        tool_name, arguments=arguments
                    )
                    return result
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"❌ Error calling tool {tool_name}: {e}")

        # Fallback: try all servers
        for server_url, server_info in mcp_clients.items():
            if tool_to_server and tool_name in tool_to_server:
                if tool_to_server[tool_name] == server_url:
                    continue
            try:
                logger.info(f"Calling tool {tool_name} with arguments {arguments}")
                result = await server_info["client"].call_tool(tool_name, arguments=arguments)
                return result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error calling tool {tool_name}: {e}")
                continue

        raise Exception(
            f"Failed to execute tool {tool_name}. "
            f"{error_msg if error_msg else 'Tool not found on any server.'}"
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

    # Singleton connection pool shared across all instances
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

    def _get_mcp_tools_and_clients(self, mcp_servers: List[str]) -> tuple[List[dict], dict, dict]:
        """Get available tools and clients from all connected MCP servers.

        This method uses the connection pool to reuse existing connections
        when possible, significantly improving performance.

        Args:
            mcp_servers: List of MCP server URLs

        Returns:
            A tuple of (tools in OpenAI format, mcp_clients dictionary, tool_to_server mapping).
        """
        pool = self._get_mcp_pool()
        return pool.get_tools_and_mapping(mcp_servers)

    def _init_token_accumulation(self):
        """Initialize token accumulation for a new request."""
        if not hasattr(self._thread_local, 'accumulated_tokens'):
            self._thread_local.accumulated_tokens = {'prompt_tokens': 0, 'completion_tokens': 0}

    def _accumulate_usage(self, resp):
        """Accumulate token usage from response object without calling set_output_context.

        This method extracts tokens from the response and adds them to the accumulated total.
        It should be called for each API response in a multi-call request flow.

        Args:
            resp: Response object with usage information
        """
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

            if prompt_tokens is None:
                prompt_tokens = 0
            if completion_tokens is None:
                completion_tokens = 0

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
        mcp_clients: dict,
        messages: List[dict],
        tool_to_server: dict = None,
    ):
        """Execute tool calls from chat completion and add results to messages.

        Uses the connection pool for tool execution.
        """
        pool = self._get_mcp_pool()

        for tool_call in tool_calls:
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            else:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                tool_id = tool_call['id']

            try:
                result = pool.call_tool(tool_name, tool_args, mcp_clients, tool_to_server)
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
        mcp_clients: dict,
        messages: List[dict],
        tool_to_server: dict = None,
    ):
        """Async version of _execute_tool_calls for streaming contexts."""
        pool = self._get_mcp_pool()

        for tool_call in tool_calls:
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            else:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                tool_id = tool_call['id']

            try:
                result = await pool.call_tool_async(
                    tool_name, tool_args, mcp_clients, tool_to_server
                )
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

    def _execute_response_api_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        mcp_clients: dict,
        input_items: List[Any],
        tool_to_server: dict = None,
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
                result = pool.call_tool(tool_name, tool_args, mcp_clients, tool_to_server)
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            except Exception as e:
                content = f"Error: {str(e)}"

            output_call_id = call_id if call_id else tool_id
            if not output_call_id:
                logger.warning(f"⚠ No call_id or id found for tool {tool_name}, skipping output")
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
        mcp_clients: dict,
        input_items: List[Any],
        tool_to_server: dict = None,
    ):
        """Async version of _execute_response_api_tool_calls for streaming contexts."""
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
                result = await pool.call_tool_async(
                    tool_name, tool_args, mcp_clients, tool_to_server
                )
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            except Exception as e:
                content = f"Error: {str(e)}"

            output_call_id = call_id if call_id else tool_id
            if not output_call_id:
                logger.warning(f"⚠ No call_id or id found for tool {tool_name}, skipping output")
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
        tool_calls_list = []
        for idx in sorted(tool_calls_accumulated.keys()):
            tc = tool_calls_accumulated[idx]
            tool_calls_list.append(
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
            )
        return tool_calls_list

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
        """Bridge an async generator to a sync generator using the pool's event loop."""
        pool = self._get_mcp_pool()
        loop = pool._get_or_create_event_loop()

        # Create a queue for communication between async generator and sync iteration
        queue = asyncio.Queue()
        done_event = threading.Event()
        exception_holder = [None]

        async def producer():
            try:
                async for item in async_gen_func():
                    await queue.put(item)
            except Exception as e:
                exception_holder[0] = e
            finally:
                await queue.put(None)  # Sentinel to signal completion
                done_event.set()

        # Start the producer in the background loop
        future = asyncio.run_coroutine_threadsafe(producer(), loop)

        try:
            while True:
                # Get from queue with timeout to allow checking for exceptions
                get_future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
                try:
                    item = get_future.result(timeout=30.0)
                except Exception as e:
                    if exception_holder[0]:
                        raise exception_holder[0]
                    raise

                if item is None:  # Sentinel
                    break
                yield item

            # Check if there was an exception in the producer
            if exception_holder[0]:
                raise exception_holder[0]
        finally:
            # Ensure the future is done
            try:
                future.result(timeout=1.0)
            except:
                pass

    async def _stream_with_mcp_tools_json(
        self,
        openai_messages: List[dict],
        tools: List[dict],
        mcp_clients: dict,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tool_to_server: dict = None,
    ):
        """Async generator to handle MCP tool calls with streaming support for chat completions."""
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
                    "content": streaming_response if streaming_response else None,
                    "tool_calls": tool_calls_list,
                }
            )
            await self._execute_tool_calls_async(
                tool_calls_list, mcp_clients, openai_messages, tool_to_server
            )

            async for chunk_json in self._stream_with_mcp_tools_json(
                openai_messages, tools, mcp_clients, max_tokens, temperature, top_p, tool_to_server
            ):
                yield chunk_json

    async def _stream_responses_with_mcp_tools_json(
        self,
        request_data: Dict[str, Any],
        tools: List[dict],
        mcp_clients: dict,
        tool_to_server: dict = None,
    ):
        """Async generator to handle MCP tool calls with streaming support for response API."""
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

            if (
                chunk_type == 'response.output_item.added'
                or chunk_type == 'ResponseOutputItemAddedEvent'
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item
                if hasattr(chunk, 'output_index'):
                    item_dict = (
                        item_to_check
                        if isinstance(item_to_check, dict)
                        else (
                            item_to_check.model_dump()
                            if hasattr(item_to_check, 'model_dump')
                            else item_to_check.dict()
                            if hasattr(item_to_check, 'dict')
                            else {}
                        )
                    )
                    item_type = item_dict.get("type")
                    if item_type == "message":
                        original_index = chunk.output_index
                        original_to_filtered_index_map[original_index] = len(
                            original_to_filtered_index_map
                        )
            elif (
                chunk_type == 'response.output_item.done'
                or chunk_type == 'ResponseOutputItemDoneEvent'
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item
            elif hasattr(chunk, 'output_index'):
                original_index = chunk.output_index
                if original_index not in original_to_filtered_index_map:
                    should_yield = False

            if item_to_check:
                item_dict = (
                    item_to_check
                    if isinstance(item_to_check, dict)
                    else (
                        item_to_check.model_dump()
                        if hasattr(item_to_check, 'model_dump')
                        else item_to_check.dict()
                        if hasattr(item_to_check, 'dict')
                        else {}
                    )
                )
                item_type = item_dict.get("type")
                if item_type != "message":
                    should_yield = False

            if (
                chunk_type == 'response.completed' or chunk_type == 'ResponseCompletedEvent'
            ) and hasattr(chunk, 'response'):
                response = chunk.response
                if hasattr(response, 'output') and response.output:
                    filtered_output = []
                    filtered_index = 0
                    original_to_filtered_index_map.clear()

                    for original_index, item in enumerate(response.output):
                        item_dict = (
                            item
                            if isinstance(item, dict)
                            else (
                                item.model_dump()
                                if hasattr(item, 'model_dump')
                                else item.dict()
                                if hasattr(item, 'dict')
                                else {}
                            )
                        )
                        item_type = item_dict.get("type")
                        if item_type == "message":
                            filtered_output.append(item_dict)
                            original_to_filtered_index_map[original_index] = filtered_index
                            filtered_index += 1
                        elif item_type in [
                            "function_tool_call",
                            "function_call",
                            "function",
                            "tool_call",
                        ]:
                            item_id = item_dict.get("id")
                            if item_id:
                                existing_ids = [
                                    i.get("id")
                                    if isinstance(i, dict)
                                    else (getattr(i, "id", None) if hasattr(i, "id") else None)
                                    for i in accumulated_output
                                ]
                                if item_id not in existing_ids:
                                    accumulated_output.append(item_dict)
                            else:
                                accumulated_output.append(item_dict)
                        else:
                            item_id = item_dict.get("id")
                            if item_id:
                                existing_ids = [
                                    i.get("id")
                                    if isinstance(i, dict)
                                    else (getattr(i, "id", None) if hasattr(i, "id") else None)
                                    for i in accumulated_output
                                ]
                                if item_id not in existing_ids:
                                    accumulated_output.append(item_dict)
                            else:
                                accumulated_output.append(item_dict)

                    response_dict = (
                        response.model_dump()
                        if hasattr(response, 'model_dump')
                        else response.dict()
                        if hasattr(response, 'dict')
                        else {}
                    )
                    response_dict["output"] = filtered_output

                    modified_chunk_dict = {
                        "type": "response.completed",
                        "sequence_number": getattr(chunk, 'sequence_number', None),
                        "response": response_dict,
                    }
                    yield json.dumps(modified_chunk_dict)
                else:
                    yield chunk.model_dump_json()
            elif should_yield:
                if hasattr(chunk, 'output_index'):
                    original_index = chunk.output_index
                    if original_index in original_to_filtered_index_map:
                        chunk_dict = (
                            chunk.model_dump()
                            if hasattr(chunk, 'model_dump')
                            else (
                                chunk.dict()
                                if hasattr(chunk, 'dict')
                                else json.loads(chunk.model_dump_json())
                                if hasattr(chunk, 'model_dump_json')
                                else {}
                            )
                        )
                        chunk_dict["output_index"] = original_to_filtered_index_map[original_index]
                        yield json.dumps(chunk_dict)
                else:
                    yield chunk.model_dump_json()

            if (
                chunk_type == 'response.output_item.added'
                or chunk_type == 'ResponseOutputItemAddedEvent'
            ) and hasattr(chunk, 'item'):
                item = chunk.item
                item_dict = (
                    item
                    if isinstance(item, dict)
                    else (
                        item.model_dump()
                        if hasattr(item, 'model_dump')
                        else item.dict()
                        if hasattr(item, 'dict')
                        else {}
                    )
                )
                item_type = item_dict.get("type")

                if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                    item_id = item_dict.get("id") or item_dict.get("call_id")
                    call_id = item_dict.get("call_id")
                    if item_id:
                        tool_calls_accumulated[item_id] = {
                            "id": item_id,
                            "call_id": call_id,
                            "type": item_type,
                            "name": item_dict.get("name", ""),
                            "arguments": item_dict.get("arguments", ""),
                            "status": item_dict.get("status", "in_progress"),
                        }

            elif (
                chunk_type == 'response.function_call_arguments.delta'
                or chunk_type == 'ResponseFunctionCallArgumentsDeltaEvent'
            ):
                item_id = getattr(chunk, 'item_id', None)
                delta = getattr(chunk, 'delta', '')

                if item_id and item_id in tool_calls_accumulated:
                    tool_calls_accumulated[item_id]["arguments"] += delta

            elif (
                chunk_type == 'response.function_call_arguments.done'
                or chunk_type == 'ResponseFunctionCallArgumentsDoneEvent'
            ):
                item_id = getattr(chunk, 'item_id', None)
                arguments = getattr(chunk, 'arguments', '')

                if item_id and item_id in tool_calls_accumulated:
                    tool_calls_accumulated[item_id]["arguments"] = arguments

            elif (
                chunk_type == 'response.output_item.done'
                or chunk_type == 'ResponseOutputItemDoneEvent'
            ) and hasattr(chunk, 'item'):
                item = chunk.item
                item_dict = (
                    item
                    if isinstance(item, dict)
                    else (
                        item.model_dump()
                        if hasattr(item, 'model_dump')
                        else item.dict()
                        if hasattr(item, 'dict')
                        else {}
                    )
                )
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

            elif hasattr(chunk, 'output') and chunk.output:
                for item in chunk.output:
                    item_dict = (
                        item
                        if isinstance(item, dict)
                        else (
                            item.model_dump()
                            if hasattr(item, 'model_dump')
                            else item.dict()
                            if hasattr(item, 'dict')
                            else {}
                        )
                    )
                    item_id = item_dict.get("id")
                    if item_id:
                        existing_ids = [
                            i.get("id")
                            if isinstance(i, dict)
                            else (getattr(i, "id", None) if hasattr(i, "id") else None)
                            for i in accumulated_output
                        ]
                        if item_id not in existing_ids:
                            accumulated_output.append(item_dict)
                    else:
                        accumulated_output.append(item_dict)

        for call_id, call_data in tool_calls_accumulated.items():
            if call_data.get("name"):
                existing_ids = [
                    i.get("id")
                    if isinstance(i, dict)
                    else (getattr(i, "id", None) if hasattr(i, "id") else None)
                    for i in accumulated_output
                ]
                if call_id not in existing_ids:
                    accumulated_output.append(call_data)

        tool_calls = self._extract_tool_calls_from_response_output(accumulated_output)
        if tool_calls:
            model_output_items = self._convert_output_items_to_input_items(accumulated_output)
            input_items.extend(model_output_items)

            await self._execute_response_api_tool_calls_async(
                tool_calls, mcp_clients, input_items, tool_to_server
            )

            request_data["input"] = input_items

            async for chunk_json in self._stream_responses_with_mcp_tools_json(
                request_data, tools, mcp_clients, tool_to_server
            ):
                yield chunk_json

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Process an OpenAI-compatible request and send it to the appropriate OpenAI endpoint."""
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            tools = request_data.get("tools")

            if mcp_servers and len(mcp_servers) > 0 and tools is None:
                logger.info(f"Getting tools and clients for MCP servers: {mcp_servers}")
                tools_local, mcp_clients_local, tool_to_server_local = (
                    self._get_mcp_tools_and_clients(mcp_servers)
                )

                # Note: No cleanup needed - connections are maintained in the pool

                if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                    response = self._route_request(
                        endpoint, request_data, mcp_servers, mcp_clients_local, tools_local
                    )

                    while response.choices and response.choices[0].message.tool_calls:
                        messages = request_data.get("messages", [])
                        messages.append(response.choices[0].message)
                        self._execute_tool_calls(
                            response.choices[0].message.tool_calls,
                            mcp_clients_local,
                            messages,
                            tool_to_server_local,
                        )
                        request_data["messages"] = messages
                        response = self._route_request(
                            endpoint,
                            request_data,
                            mcp_servers,
                            mcp_clients_local,
                            tools_local,
                        )

                elif endpoint == self.ENDPOINT_RESPONSES:
                    response = self._route_request(
                        endpoint, request_data, mcp_servers, mcp_clients_local, tools_local
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
                            tool_calls,
                            mcp_clients_local,
                            input_items,
                            tool_to_server_local,
                        )
                        request_data["input"] = input_items

                        response = self._route_request(
                            endpoint,
                            request_data,
                            mcp_servers,
                            mcp_clients_local,
                            tools_local,
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
        """Process an OpenAI-compatible request and return a streaming response iterator."""
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming is only supported for chat completions and responses.")

            if mcp_servers and len(mcp_servers) > 0 and request_data.get("tools") is None:
                logger.info(f"Getting tools and clients for MCP servers: {mcp_servers}")
                tools_local, mcp_clients_local, tool_to_server_local = (
                    self._get_mcp_tools_and_clients(mcp_servers)
                )

                # Note: No cleanup needed - connections are maintained in the pool

                async def stream_generator():
                    if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                        messages = request_data.get("messages", [])
                        async for chunk_json in self._stream_with_mcp_tools_json(
                            messages,
                            tools_local,
                            mcp_clients_local,
                            request_data.get("max_completion_tokens", 4096),
                            request_data.get("temperature", 1.0),
                            request_data.get("top_p", 1.0),
                            tool_to_server_local,
                        ):
                            yield chunk_json
                        self._finalize_token_usage()
                    elif endpoint == self.ENDPOINT_RESPONSES:
                        async for chunk_json in self._stream_responses_with_mcp_tools_json(
                            request_data, tools_local, mcp_clients_local, tool_to_server_local
                        ):
                            yield chunk_json
                        self._finalize_token_usage()
                    else:
                        response_args = {**request_data, "model": self.model}
                        for chunk in self.client.responses.create(**response_args):
                            self._set_usage(chunk)
                            yield chunk.model_dump_json()
                        self._finalize_token_usage()

                yield from self._bridge_async_generator(stream_generator)
                return

            if endpoint == self.ENDPOINT_RESPONSES:
                response_args = {**request_data, "model": self.model}
                for chunk in self.client.responses.create(**response_args):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
                self._finalize_token_usage()
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
