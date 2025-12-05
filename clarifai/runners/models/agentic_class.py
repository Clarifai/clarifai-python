"""Base class for creating OpenAI-compatible API server with MCP (Model Context Protocol) support."""

import asyncio
import json
import os
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api.status import status_code_pb2
from pydantic_core import from_json, to_json

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger


class AgenticModelClass(OpenAIModelClass):
    """Base class for wrapping OpenAI-compatible servers with MCP (Model Context Protocol) support.

    This class extends OpenAIModelClass to enable agentic behavior by integrating LLMs with MCP servers.
    It handles tool discovery, execution, and iterative tool calling for both chat completions and
    responses endpoints, supporting both streaming and non-streaming modes.

    To use this class, create a subclass and set the following class attributes:
    - client: The OpenAI-compatible client instance
    - model: The name of the model to use with the client

    Example:
        class MyAgenticModel(AgenticModelClass):
            client = OpenAI(api_key="your-key")
            model = "gpt-4"
    """

    async def _connect_to_servers(
        self, mcp_servers: List[str], max_retries: int = 2, retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """Connect to all configured Clarifai MCP servers.

        Args:
            mcp_servers: List of MCP server URLs to connect to
            max_retries: Maximum number of retry attempts per server (default: 2)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)

        Returns:
            Dictionary mapping server URLs to client info and tools
        """
        try:
            from fastmcp import Client
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError:
            raise ImportError(
                "fastmcp package is required to use MCP functionality. "
                "Install it with: pip install fastmcp"
            )

        mcp_clients = {}

        for mcp_url in mcp_servers:
            last_error = None
            connected = False

            for attempt in range(max_retries):
                try:
                    # Create transport for this server
                    transport = StreamableHttpTransport(
                        url=mcp_url,
                        headers={"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]},
                    )

                    # Create and connect client
                    client = Client(transport)
                    await client.__aenter__()

                    # Store client with server info
                    mcp_clients[mcp_url] = {"client": client, "tools": []}

                    # List available tools
                    tools_result = await client.list_tools()
                    mcp_clients[mcp_url]["tools"] = tools_result

                    logger.info(f"✓ Connected to {mcp_url} with {len(tools_result)} tools")
                    connected = True
                    break  # Success, exit retry loop

                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠ Failed to connect to {mcp_url} (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"❌ Failed to connect to {mcp_url} after {max_retries} attempts: {e}"
                        )

            if not connected:
                # Log final failure if all retries exhausted
                logger.error(
                    f"❌ Could not connect to {mcp_url} after {max_retries} attempts. "
                    f"Last error: {last_error}"
                )
                # Continue with other servers even if one fails

        return mcp_clients

    async def _get_mcp_tools_and_clients(
        self, mcp_servers: List[str]
    ) -> tuple[List[dict], dict, dict]:
        """Get available tools and clients from all connected MCP servers.

        Args:
            mcp_servers: List of MCP server URLs

        Returns:
            A tuple of (tools in OpenAI format, mcp_clients dictionary, tool_to_server mapping).
        """
        mcp_clients = await self._connect_to_servers(mcp_servers)

        all_tools = []
        tool_to_server = {}  # Map tool name to server URL

        for mcp_url, server_info in mcp_clients.items():
            tools = server_info["tools"]
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
                # Map tool name to its server URL
                tool_to_server[tool_name] = mcp_url

        logger.info(f"Access to the {len(all_tools)} tools")
        return all_tools, mcp_clients, tool_to_server

    async def _cleanup(self, mcp_clients: dict):
        """Clean up MCP client resources.

        Args:
            mcp_clients: Dictionary of MCP clients to clean up
        """
        logger.info("Cleaning up MCP connections...")
        for mcp_url, server_info in mcp_clients.items():
            try:
                client = server_info["client"]
                # Try to close the client properly
                if hasattr(client, 'close') and callable(getattr(client, 'close', None)):
                    if asyncio.iscoroutinefunction(client.close):
                        await client.close()
                    else:
                        client.close()
                else:
                    await client.__aexit__(None, None, None)
                logger.info(f"✓ Disconnected from {mcp_url}")
            except Exception as e:
                # Log other errors but don't fail cleanup
                logger.warning(f"⚠ Error disconnecting from {mcp_url}: {e} (continuing cleanup)")

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
        # Extract usage from response (same logic as base _set_usage)
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

            # Only accumulate if we have valid tokens
            if prompt_tokens > 0 or completion_tokens > 0:
                self._init_token_accumulation()
                self._thread_local.accumulated_tokens['prompt_tokens'] += prompt_tokens
                self._thread_local.accumulated_tokens['completion_tokens'] += completion_tokens

    def _finalize_token_usage(self):
        """Finalize token accumulation and set the total in output context.

        This should be called once at the end of a request that may have multiple API calls.
        """
        if hasattr(self._thread_local, 'accumulated_tokens'):
            prompt_tokens = self._thread_local.accumulated_tokens['prompt_tokens']
            completion_tokens = self._thread_local.accumulated_tokens['completion_tokens']

            if prompt_tokens > 0 or completion_tokens > 0:
                self.set_output_context(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            # Clean up
            del self._thread_local.accumulated_tokens

    def _set_usage(self, resp):
        """Override _set_usage to accumulate tokens across multiple API calls.

        In agentic flows, multiple OpenAI API calls are made (initial + recursive calls after tool execution).
        This method accumulates tokens from all calls and only sets the final total once.

        Args:
            resp: Response object with usage information
        """
        # Accumulate tokens instead of immediately setting them
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

        # Use base class implementation
        return super()._handle_chat_completions(request_data)

    def _convert_tools_to_response_api_format(self, tools: List[dict]) -> List[dict]:
        """Convert tools from chat completion format to response API format.

        Chat completion format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Response API format: {"type": "function", "name": ..., "description": ..., "parameters": ...}

        Args:
            tools: List of tools in chat completion format

        Returns:
            List of tools in response API format
        """
        response_api_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type", "function")
                # Check if it's in chat completion format (has nested "function")
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
                # Already in response API format
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
        # If we have MCP tools, convert them to response API format and add them to the request
        if mcp_servers and tools:
            request_data = request_data.copy()  # Don't modify original
            # Convert tools from chat completion format to response API format
            response_api_tools = self._convert_tools_to_response_api_format(tools)
            request_data["tools"] = response_api_tools
            request_data["tool_choice"] = request_data.get("tool_choice", "auto")

        # Use base class implementation
        return super()._handle_responses(request_data)

    def _route_request(
        self,
        endpoint: str,
        request_data: Dict[str, Any],
        mcp_servers: List[str] = None,
        mcp_clients: dict = None,
        tools: List[dict] = None,
    ):
        """Route the request to appropriate handler based on endpoint, with optional MCP support."""
        # For chat completions, pass MCP parameters
        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
            return self._handle_chat_completions(request_data, mcp_servers, mcp_clients, tools)

        # For responses endpoint, pass MCP parameters
        if endpoint == self.ENDPOINT_RESPONSES:
            return self._handle_responses(request_data, mcp_servers, mcp_clients, tools)

        # For other endpoints, use base class implementation
        return super()._route_request(endpoint, request_data)

    async def _execute_tool_calls(
        self,
        tool_calls: List[Any],
        mcp_clients: dict,
        messages: List[dict],
        tool_to_server: dict = None,
    ):
        """Execute tool calls from chat completion and add results to messages. Handles both OpenAI tool_call objects and dict format."""
        for tool_call in tool_calls:
            # Handle both OpenAI tool_call objects and dict format
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            else:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                tool_id = tool_call['id']

            result = None
            error_msg = None

            # If we have tool-to-server mapping, try the correct server first
            if tool_to_server and tool_name in tool_to_server:
                server_url = tool_to_server[tool_name]
                if server_url in mcp_clients:
                    try:
                        logger.info(f"Calling tool {tool_name} with arguments {tool_args}")
                        result = await mcp_clients[server_url]["client"].call_tool(
                            tool_name, arguments=tool_args
                        )
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ Error calling tool {tool_name}: {e}")

            # If not found or failed, try all servers as fallback
            if result is None:
                for server_url, server_info in mcp_clients.items():
                    # Skip if we already tried this server
                    if (
                        tool_to_server
                        and tool_name in tool_to_server
                        and tool_to_server[tool_name] == server_url
                    ):
                        continue
                    try:
                        logger.info(f"Calling tool {tool_name} with arguments {tool_args}")
                        result = await server_info["client"].call_tool(
                            tool_name, arguments=tool_args
                        )
                        break
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ Error calling tool {tool_name}: {e}")
                        continue

            if result:
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            else:
                content = f"Error: Failed to execute tool {tool_name}. {error_msg if error_msg else 'Tool not found on any server.'}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content,
                }
            )

    async def _execute_response_api_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        mcp_clients: dict,
        input_items: List[Any],
        tool_to_server: dict = None,
    ):
        """Execute tool calls from response API and add results to input items.

        Args:
            tool_calls: List of tool call dicts from response API output
            mcp_clients: Dictionary of MCP clients
            input_items: List of input items (can be modified in place)
            tool_to_server: Mapping of tool names to server URLs
        """
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args_str = tool_call.get("arguments", "{}")
            tool_id = tool_call.get("id")
            call_id = tool_call.get("call_id")

            # Parse arguments
            try:
                tool_args = (
                    json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                )
            except json.JSONDecodeError:
                tool_args = {}

            result = None
            error_msg = None

            # If we have tool-to-server mapping, try the correct server first
            if tool_to_server and tool_name in tool_to_server:
                server_url = tool_to_server[tool_name]
                if server_url in mcp_clients:
                    try:
                        logger.info(f"Calling tool {tool_name} with arguments {tool_args}")
                        result = await mcp_clients[server_url]["client"].call_tool(
                            tool_name, arguments=tool_args
                        )
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ Error calling tool {tool_name}: {e}")

            # If not found or failed, try all servers as fallback
            if result is None:
                for server_url, server_info in mcp_clients.items():
                    # Skip if we already tried this server
                    if (
                        tool_to_server
                        and tool_name in tool_to_server
                        and tool_to_server[tool_name] == server_url
                    ):
                        continue
                    try:
                        logger.info(f"Calling tool {tool_name} with arguments {tool_args}")
                        result = await server_info["client"].call_tool(
                            tool_name, arguments=tool_args
                        )
                        break
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"❌ Error calling tool {tool_name} : {e}")
                        continue

            # Get tool output
            if result:
                content = (
                    result.content[0].text if hasattr(result, 'content') else str(result[0].text)
                )
            else:
                content = f"Error: Failed to execute tool {tool_name}. {error_msg if error_msg else 'Tool not found on any server.'}"

            # Use call_id if available, otherwise use id (call_id is required for function_call_output)
            output_call_id = call_id if call_id else tool_id
            if not output_call_id:
                # If neither is available, skip this tool call
                logger.warning(
                    f"⚠ Warning: No call_id or id found for tool {tool_name}, skipping output"
                )
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
        """Extract tool calls from response API output array.

        Args:
            response_output: List of output items from response API

        Returns:
            List of tool call dictionaries that need to be executed
        """
        tool_calls = []
        for item in response_output:
            # Convert item to dict if it's a Pydantic model
            if not isinstance(item, dict):
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                elif hasattr(item, '__dict__'):
                    item = item.__dict__
                else:
                    continue

            # Check if item is a function_tool_call that needs execution
            item_type = item.get("type")
            if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                # Only execute if status indicates it needs execution (not already completed)
                status = item.get("status", "")
                output = item.get("output")
                # Execute if status is pending/in_progress/empty or if output is missing
                if status in ["pending", "in_progress", ""] or output is None:
                    tool_calls.append(item)
        return tool_calls

    def _convert_output_items_to_input_items(
        self, response_output: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert response API output items to input items format.

        This includes messages, reasoning, and completed tool calls (with outputs).
        Excludes tool calls that are pending or in progress.

        Args:
            response_output: List of output items from response API

        Returns:
            List of input items in the format expected by response API
        """
        input_items = []
        for item in response_output:
            # Convert item to dict if it's a Pydantic model
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

            # Include messages and reasoning as-is
            if item_type in ["message", "reasoning"]:
                input_items.append(item)
            # Include completed tool calls (with output) as function_tool_call items
            elif item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                status = item.get("status", "")
                output = item.get("output")
                # Only include if it's completed (has output)
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
        """Convert accumulated tool calls dictionary to list format in chat completion format."""
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

    def _accumulate_response_tool_call_delta(
        self, delta_item: Dict[str, Any], tool_calls_accumulated: Dict[str, Dict[str, Any]]
    ):
        """Accumulate tool call data from a streaming delta in response API format.

        Args:
            delta_item: A delta item from response API streaming (type="function_tool_call")
            tool_calls_accumulated: Dictionary mapping call_id to accumulated tool call data
        """
        # Get call_id or generate one if not present
        call_id = delta_item.get("call_id") or delta_item.get("id")
        if not call_id:
            # Use a temporary ID based on output_index if available
            output_index = delta_item.get("output_index", 0)
            call_id = f"temp_{output_index}"

        if call_id not in tool_calls_accumulated:
            tool_calls_accumulated[call_id] = {
                "id": call_id,
                "type": "function_tool_call",
                "name": "",
                "arguments": "",
                "status": "in_progress",
            }

        # Accumulate name (may come incrementally)
        if "name" in delta_item and delta_item["name"]:
            tool_calls_accumulated[call_id]["name"] = delta_item["name"]

        # Accumulate arguments (may come incrementally as string)
        if "arguments" in delta_item and delta_item["arguments"]:
            tool_calls_accumulated[call_id]["arguments"] += delta_item["arguments"]

        # Update status if present
        if "status" in delta_item:
            tool_calls_accumulated[call_id]["status"] = delta_item["status"]

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
        """Bridge an async generator to a sync generator."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            gen = async_gen_func()
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

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
        """Async generator to handle MCP tool calls with streaming support for chat completions, yielding JSON chunks."""
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

        # Execute tool calls if any were accumulated
        if tool_calls_accumulated:
            tool_calls_list = self._convert_accumulated_tool_calls(tool_calls_accumulated)
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": streaming_response if streaming_response else None,
                    "tool_calls": tool_calls_list,
                }
            )
            await self._execute_tool_calls(
                tool_calls_list, mcp_clients, openai_messages, tool_to_server
            )

            # Continue streaming with tool results (recursive call - don't finalize here)
            async for chunk_json in self._stream_with_mcp_tools_json(
                openai_messages, tools, mcp_clients, max_tokens, temperature, top_p, tool_to_server
            ):
                yield chunk_json
        # Note: Finalization happens at the top level in openai_stream_transport

    async def _stream_responses_with_mcp_tools_json(
        self,
        request_data: Dict[str, Any],
        tools: List[dict],
        mcp_clients: dict,
        tool_to_server: dict = None,
    ):
        """Async generator to handle MCP tool calls with streaming support for response API, yielding JSON chunks."""
        # Get input items
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

        # Create request with tools (convert to response API format)
        response_args = {**request_data, "model": self.model}
        if tools:
            # Convert tools from chat completion format to response API format
            response_api_tools = self._convert_tools_to_response_api_format(tools)
            response_args["tools"] = response_api_tools
            response_args["tool_choice"] = response_args.get("tool_choice", "auto")

        # Stream the response and accumulate output
        stream = self.client.responses.create(**response_args)
        accumulated_output = []
        tool_calls_accumulated = {}  # Track tool calls incrementally
        original_to_filtered_index_map = {}  # Map original output indices to filtered indices (for messages only)

        for chunk in stream:
            self._set_usage(chunk)

            # Handle different event types from response API streaming
            chunk_type = getattr(chunk, 'type', None) or chunk.__class__.__name__

            # Check if this event contains a non-message item that should be filtered
            should_yield = True
            item_to_check = None

            # Check response.output_item.added events
            if (
                chunk_type == 'response.output_item.added'
                or chunk_type == 'ResponseOutputItemAddedEvent'
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item
                # Build index mapping for messages as we see them
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
                        # This is a message, map original index to filtered index
                        original_index = chunk.output_index
                        # The filtered index is just the count of messages we've seen so far
                        original_to_filtered_index_map[original_index] = len(
                            original_to_filtered_index_map
                        )
            # Check response.output_item.done events
            elif (
                chunk_type == 'response.output_item.done'
                or chunk_type == 'ResponseOutputItemDoneEvent'
            ) and hasattr(chunk, 'item'):
                item_to_check = chunk.item
            # Check events with output_index (like response.output_item.delta)
            elif hasattr(chunk, 'output_index'):
                original_index = chunk.output_index
                # Only yield if this index maps to a message
                if original_index not in original_to_filtered_index_map:
                    should_yield = False

            # If we have an item to check, verify it's a message type
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
                # Only yield if it's a message, otherwise skip (but still process internally)
                if item_type != "message":
                    should_yield = False

            # For response.completed events, filter output to only include messages before yielding
            if (
                chunk_type == 'response.completed' or chunk_type == 'ResponseCompletedEvent'
            ) and hasattr(chunk, 'response'):
                response = chunk.response
                if hasattr(response, 'output') and response.output:
                    # Filter output to only include message items and build index mapping
                    filtered_output = []
                    filtered_index = 0
                    original_to_filtered_index_map.clear()  # Reset mapping for this response

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
                        # Only include message items in the filtered output (as dicts for JSON serialization)
                        if item_type == "message":
                            filtered_output.append(item_dict)
                            original_to_filtered_index_map[original_index] = filtered_index
                            filtered_index += 1
                        # Still accumulate tool calls for internal processing
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
                            # For other types, still accumulate but don't include in filtered output
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

                    # Create a modified response with filtered output
                    response_dict = (
                        response.model_dump()
                        if hasattr(response, 'model_dump')
                        else response.dict()
                        if hasattr(response, 'dict')
                        else {}
                    )
                    response_dict["output"] = filtered_output

                    # Create modified chunk with filtered response
                    modified_chunk_dict = {
                        "type": "response.completed",
                        "sequence_number": getattr(chunk, 'sequence_number', None),
                        "response": response_dict,
                    }
                    yield json.dumps(modified_chunk_dict)
                else:
                    # No output to filter, yield as-is
                    yield chunk.model_dump_json()
            elif should_yield:
                # For events with output_index, remap to filtered index if it's a message index
                if hasattr(chunk, 'output_index'):
                    original_index = chunk.output_index
                    if original_index in original_to_filtered_index_map:
                        # Remap the output_index to the filtered index
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
                    # else: already filtered out by should_yield = False above
                else:
                    # For all other chunk types, yield as-is (if not filtered out)
                    yield chunk.model_dump_json()

            # Handle ResponseOutputItemAddedEvent - initial tool call item
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

                # If it's a tool call, start accumulating it
                if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                    item_id = item_dict.get("id") or item_dict.get("call_id")
                    call_id = item_dict.get("call_id")
                    if item_id:
                        tool_calls_accumulated[item_id] = {
                            "id": item_id,
                            "call_id": call_id,  # Preserve call_id for function_call_output
                            "type": item_type,
                            "name": item_dict.get("name", ""),
                            "arguments": item_dict.get("arguments", ""),
                            "status": item_dict.get("status", "in_progress"),
                        }

            # Handle ResponseFunctionCallArgumentsDeltaEvent - incremental argument updates
            elif (
                chunk_type == 'response.function_call_arguments.delta'
                or chunk_type == 'ResponseFunctionCallArgumentsDeltaEvent'
            ):
                item_id = getattr(chunk, 'item_id', None)
                delta = getattr(chunk, 'delta', '')

                if item_id and item_id in tool_calls_accumulated:
                    # Accumulate the delta arguments
                    tool_calls_accumulated[item_id]["arguments"] += delta

            # Handle ResponseFunctionCallArgumentsDoneEvent - arguments complete
            elif (
                chunk_type == 'response.function_call_arguments.done'
                or chunk_type == 'ResponseFunctionCallArgumentsDoneEvent'
            ):
                item_id = getattr(chunk, 'item_id', None)
                arguments = getattr(chunk, 'arguments', '')

                if item_id and item_id in tool_calls_accumulated:
                    # Set final arguments
                    tool_calls_accumulated[item_id]["arguments"] = arguments

            # Handle ResponseOutputItemDoneEvent - tool call item completed
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

                # If it's a completed tool call, add to accumulated output
                if item_type in ["function_tool_call", "function_call", "function", "tool_call"]:
                    item_id = item_dict.get("id")
                    if item_id and item_id in tool_calls_accumulated:
                        # Update with final status and preserve call_id if present
                        tool_calls_accumulated[item_id]["status"] = item_dict.get(
                            "status", "completed"
                        )
                        if "call_id" in item_dict:
                            tool_calls_accumulated[item_id]["call_id"] = item_dict.get("call_id")
                        # Add to accumulated output
                        accumulated_output.append(tool_calls_accumulated[item_id])
                    else:
                        # Not in accumulated, add directly
                        accumulated_output.append(item_dict)
                else:
                    # Non-tool-call item
                    accumulated_output.append(item_dict)

            # Handle standard response objects with output (fallback)
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

        # After streaming completes, add any remaining accumulated tool calls
        for call_id, call_data in tool_calls_accumulated.items():
            # Only add if it has a name and is not already in accumulated_output
            if call_data.get("name"):
                existing_ids = [
                    i.get("id")
                    if isinstance(i, dict)
                    else (getattr(i, "id", None) if hasattr(i, "id") else None)
                    for i in accumulated_output
                ]
                if call_id not in existing_ids:
                    accumulated_output.append(call_data)

        # Check for tool calls in accumulated output
        tool_calls = self._extract_tool_calls_from_response_output(accumulated_output)
        # Execute tool calls if any
        if tool_calls:
            # Convert model's output (messages, reasoning, completed tool calls) to input items
            model_output_items = self._convert_output_items_to_input_items(accumulated_output)
            input_items.extend(model_output_items)

            # Execute tool calls and add results to input
            await self._execute_response_api_tool_calls(
                tool_calls, mcp_clients, input_items, tool_to_server
            )

            # Update request with new input including model output and tool results
            request_data["input"] = input_items

            # Continue streaming with tool results (recursive call - don't finalize here)
            async for chunk_json in self._stream_responses_with_mcp_tools_json(
                request_data, tools, mcp_clients, tool_to_server
            ):
                yield chunk_json
        # Note: Finalization happens at the top level in openai_stream_transport

    @ModelClass.method
    def openai_transport(self, msg: str) -> str:
        """Process an OpenAI-compatible request and send it to the appropriate OpenAI endpoint.

        Args:
            msg: JSON string containing the request parameters including 'openai_endpoint'

        Returns:
            JSON string containing the response or error
        """
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)
            tools = request_data.get("tools")

            if mcp_servers and len(mcp_servers) > 0 and tools is None:

                async def run_with_mcp():
                    logger.info(f"Getting tools and clients for MCP servers: {mcp_servers}")
                    (
                        tools_local,
                        mcp_clients_local,
                        tool_to_server_local,
                    ) = await self._get_mcp_tools_and_clients(mcp_servers)
                    try:
                        if endpoint == self.ENDPOINT_CHAT_COMPLETIONS:
                            response = self._route_request(
                                endpoint, request_data, mcp_servers, mcp_clients_local, tools_local
                            )

                            # Handle tool calls iteratively for chat completions
                            while response.choices and response.choices[0].message.tool_calls:
                                messages = request_data.get("messages", [])
                                messages.append(response.choices[0].message)
                                await self._execute_tool_calls(
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

                            return response
                        elif endpoint == self.ENDPOINT_RESPONSES:
                            response = self._route_request(
                                endpoint, request_data, mcp_servers, mcp_clients_local, tools_local
                            )

                            # Handle tool calls iteratively for response API
                            # Get input items (can be string or list)
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

                            # Extract tool calls from response output
                            response_output = (
                                response.output if hasattr(response, 'output') else []
                            )
                            tool_calls = self._extract_tool_calls_from_response_output(
                                response_output
                            )

                            while tool_calls:
                                # Convert model's output (messages, reasoning, completed tool calls) to input items
                                model_output_items = self._convert_output_items_to_input_items(
                                    response_output
                                )
                                input_items.extend(model_output_items)

                                # Execute tool calls and add results to input
                                await self._execute_response_api_tool_calls(
                                    tool_calls,
                                    mcp_clients_local,
                                    input_items,
                                    tool_to_server_local,
                                )
                                # Update request with new input including model output and tool results
                                request_data["input"] = input_items

                                # Make new request with tool results
                                response = self._route_request(
                                    endpoint,
                                    request_data,
                                    mcp_servers,
                                    mcp_clients_local,
                                    tools_local,
                                )

                                # Check for more tool calls
                                response_output = (
                                    response.output if hasattr(response, 'output') else []
                                )
                                tool_calls = self._extract_tool_calls_from_response_output(
                                    response_output
                                )

                            return response
                        else:
                            return self._route_request(endpoint, request_data)
                    finally:
                        await self._cleanup(mcp_clients_local)

                response = asyncio.run(run_with_mcp())
            else:
                response = self._route_request(endpoint, request_data)

            # Finalize token usage accumulation (sum of all API calls)
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
        """Process an OpenAI-compatible request and return a streaming response iterator.

        This method is used when stream=True and returns an iterator of strings directly,
        without converting to a list or JSON serializing. Supports chat completions and responses endpoints.

        Args:
            msg: The request as a JSON string.

        Returns:
            Iterator[str]: An iterator yielding text chunks from the streaming response.
        """
        try:
            request_data = from_json(msg)
            request_data = self._update_old_fields(request_data)
            mcp_servers = request_data.pop("mcp_servers", None)
            endpoint = request_data.pop("openai_endpoint", self.DEFAULT_ENDPOINT)

            if endpoint not in [self.ENDPOINT_CHAT_COMPLETIONS, self.ENDPOINT_RESPONSES]:
                raise ValueError("Streaming is only supported for chat completions and responses.")

            if mcp_servers and len(mcp_servers) > 0 and request_data.get("tools") is None:

                async def run_with_mcp():
                    logger.info(f"Getting tools and clients for MCP servers: {mcp_servers}")
                    (
                        tools_local,
                        mcp_clients_local,
                        tool_to_server_local,
                    ) = await self._get_mcp_tools_and_clients(mcp_servers)
                    try:
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
                            # Finalize token usage accumulation after streaming completes
                            self._finalize_token_usage()
                        elif endpoint == self.ENDPOINT_RESPONSES:
                            async for chunk_json in self._stream_responses_with_mcp_tools_json(
                                request_data, tools_local, mcp_clients_local, tool_to_server_local
                            ):
                                yield chunk_json
                            # Finalize token usage accumulation after streaming completes
                            self._finalize_token_usage()
                        else:
                            # Fallback for other endpoints
                            response_args = {**request_data, "model": self.model}
                            for chunk in self.client.responses.create(**response_args):
                                self._set_usage(chunk)
                                yield chunk.model_dump_json()
                            # Finalize token usage accumulation after streaming completes
                            self._finalize_token_usage()
                    finally:
                        await self._cleanup(mcp_clients_local)

                yield from self._bridge_async_generator(run_with_mcp)
                return

            # Non-MCP path or responses endpoint
            if endpoint == self.ENDPOINT_RESPONSES:
                response_args = {**request_data, "model": self.model}
                for chunk in self.client.responses.create(**response_args):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
                # Finalize token usage accumulation after streaming completes
                self._finalize_token_usage()
            else:
                completion_args = self._create_completion_args(request_data)
                for chunk in self.client.chat.completions.create(**completion_args):
                    self._set_usage(chunk)
                    yield chunk.model_dump_json()
                # Finalize token usage accumulation after streaming completes
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
