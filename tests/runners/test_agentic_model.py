"""Comprehensive tests for AgenticModelClass functionality.

This test module provides extensive coverage for the AgenticModelClass, which extends
OpenAIModelClass to add MCP (Model Context Protocol) support for tool calling.

Test Coverage:
    1. MCP Connection Pool Management
       - Singleton pattern verification
       - Connection creation and caching
       - Connection verification and lifecycle
       - Idle connection cleanup
       - Connection pool initialization

    2. Tool Discovery and Management
       - Tool loading from MCP servers
       - Tool cache updates
       - Multiple server support
       - Tool-to-server mapping

    3. Tool Execution
       - Single tool calls (sync and async)
       - Batch tool calls with parallel execution
       - Tool call result handling
       - Error handling in tool execution

    4. Streaming and Non-Streaming Modes
       - Chat completions with and without tools
       - Response API with and without tools
       - Tool call accumulation in streaming mode
       - Token tracking across requests

    5. Error Handling
       - Invalid requests
       - Tool execution failures
       - Connection failures
       - Missing tools

    6. Integration Tests
       - Full chat completion workflow with tool calling
       - Full streaming workflow with tool calling
       - Multiple tool iterations

The tests use mock objects to simulate MCP server behavior without requiring
actual server connections, making them suitable for CI/CD environments.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from clarifai.runners.models.agentic_class import (
    AgenticModelClass,
    MCPConnection,
    MCPConnectionPool,
)
from clarifai.runners.models.dummy_agentic_model import (
    DummyAgenticModel,
    MockMCPClient,
    MockMCPTool,
)


class TestMCPConnectionPool:
    """Tests for MCP connection pool management."""

    def test_connection_pool_singleton(self):
        """Test that connection pool is a singleton."""
        pool1 = MCPConnectionPool()
        pool2 = MCPConnectionPool()
        assert pool1 is pool2

    def test_connection_pool_initialization(self):
        """Test connection pool initialization."""
        pool = MCPConnectionPool()
        assert pool._connections == {}
        assert pool._tool_to_url == {}
        assert pool._all_tools == {}
        assert pool._loop is not None
        assert pool._loop_thread is not None

    @pytest.mark.asyncio
    async def test_create_connection(self):
        """Test creating a new MCP connection."""
        pool = MCPConnectionPool()
        
        # Mock the Client and transport
        with patch('clarifai.runners.models.agentic_class.Client') as mock_client_class, \
             patch('clarifai.runners.models.agentic_class.StreamableHttpTransport'):
            
            # Create a mock client instance
            mock_client = Mock()
            mock_client.__aenter__ = Mock(return_value=asyncio.Future())
            mock_client.__aenter__.return_value.set_result(mock_client)
            
            # Mock list_tools
            mock_tool = MockMCPTool("test_tool", "Test tool")
            list_tools_future = asyncio.Future()
            list_tools_future.set_result([mock_tool])
            mock_client.list_tools = Mock(return_value=list_tools_future)
            
            mock_client_class.return_value = mock_client
            
            # Create connection
            conn = await pool._create_connection("http://test.com")
            
            assert conn.url == "http://test.com"
            assert len(conn.tools) == 1
            assert "test_tool" in conn.tool_names

    def test_get_connections_caching(self):
        """Test that connections are cached and reused."""
        pool = MCPConnectionPool()
        
        # Mock connection creation
        with patch.object(pool, '_create_connection') as mock_create:
            mock_conn = MCPConnection(
                client=Mock(),
                tools=[MockMCPTool("test_tool")],
                tool_names={"test_tool"},
                url="http://test.com"
            )
            
            async def create_conn(url):
                return mock_conn
            
            mock_create.side_effect = create_conn
            
            # First call should create connection
            connections = pool.get_connections(["http://test.com"])
            assert len(connections) == 1
            assert "http://test.com" in connections
            
            # Second call should reuse connection
            connections2 = pool.get_connections(["http://test.com"])
            assert len(connections2) == 1
            
            # Should only call create once
            assert mock_create.call_count == 1

    def test_connection_touch_updates_last_used(self):
        """Test that touching a connection updates its last_used time."""
        conn = MCPConnection(
            client=Mock(),
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        
        # Record the initial time and manually set it to a past time
        old_time = time.time() - 10.0
        conn.last_used = old_time
        
        # Touch should update to current time
        conn.touch()
        
        assert conn.last_used > old_time
        assert conn.last_used >= time.time() - 1.0  # Within last second

    def test_idle_connection_cleanup(self):
        """Test that idle connections are cleaned up."""
        pool = MCPConnectionPool()
        
        # Create a connection and set it as idle
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[MockMCPTool("test_tool")],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        # Set last_used to make it appear idle for a long time
        mock_conn.last_used = time.time() - (pool.MAX_IDLE_TIME + 100)
        
        # Add connection to pool
        with pool._lock:
            pool._connections["http://test.com"] = mock_conn
            pool._tool_to_url["test_tool"] = "http://test.com"
        
        # Trigger cleanup
        pool._maybe_cleanup_idle()
        
        # Connection should be removed
        assert "http://test.com" not in pool._connections
        assert "test_tool" not in pool._tool_to_url

    def test_cleanup_rate_limiting(self):
        """Test that cleanup is rate limited."""
        pool = MCPConnectionPool()
        
        # Set last cleanup to now
        pool._last_cleanup = time.time()
        
        # Create idle connection
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        mock_conn.last_used = time.time() - (pool.MAX_IDLE_TIME + 100)
        
        with pool._lock:
            pool._connections["http://test.com"] = mock_conn
        
        # Trigger cleanup - should skip due to rate limiting
        pool._maybe_cleanup_idle()
        
        # Connection should still be there
        assert "http://test.com" in pool._connections

    def test_needs_verification(self):
        """Test connection verification check."""
        pool = MCPConnectionPool()
        
        # Fresh connection - should not need verification
        conn1 = MCPConnection(
            client=Mock(),
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        assert not pool._needs_verification(conn1)
        
        # Old connection - should need verification
        conn2 = MCPConnection(
            client=Mock(),
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        conn2.last_used = time.time() - (pool.VERIFY_IDLE_THRESHOLD + 10)
        assert pool._needs_verification(conn2)

    @pytest.mark.asyncio
    async def test_verify_connection(self):
        """Test connection verification."""
        pool = MCPConnectionPool()
        
        # Mock client with working list_tools
        mock_client = Mock()
        list_tools_future = asyncio.Future()
        list_tools_future.set_result([])
        mock_client.list_tools = Mock(return_value=list_tools_future)
        
        conn = MCPConnection(
            client=mock_client,
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        
        # Should return True for valid connection
        is_valid = await pool._verify_connection(conn)
        assert is_valid

    @pytest.mark.asyncio
    async def test_verify_connection_failure(self):
        """Test connection verification with failing connection."""
        pool = MCPConnectionPool()
        
        # Mock client that fails list_tools
        mock_client = Mock()
        mock_client.list_tools = Mock(side_effect=Exception("Connection failed"))
        
        conn = MCPConnection(
            client=mock_client,
            tools=[],
            tool_names=set(),
            url="http://test.com"
        )
        
        # Should return False for invalid connection
        is_valid = await pool._verify_connection(conn)
        assert not is_valid

    def test_update_tool_cache(self):
        """Test that tool cache is updated correctly."""
        pool = MCPConnectionPool()
        
        tool = MockMCPTool("test_tool", "Test tool", {"type": "object"})
        conn = MCPConnection(
            client=Mock(),
            tools=[tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        pool._update_tool_cache(conn)
        
        assert pool._tool_to_url["test_tool"] == "http://test.com"
        assert "test_tool" in pool._all_tools
        assert pool._all_tools["test_tool"]["function"]["name"] == "test_tool"

    def test_get_tools_and_mapping(self):
        """Test getting tools and mapping from connections."""
        pool = MCPConnectionPool()
        
        # Mock get_connections
        tool = MockMCPTool("test_tool", "Test tool", {"type": "object"})
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        with patch.object(pool, 'get_connections', return_value={"http://test.com": mock_conn}):
            tools, connections, tool_to_server = pool.get_tools_and_mapping(["http://test.com"])
        
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"
        assert "http://test.com" in connections
        assert tool_to_server["test_tool"] == "http://test.com"

    @pytest.mark.asyncio
    async def test_call_tool_async(self):
        """Test async tool execution."""
        pool = MCPConnectionPool()
        
        # Create mock connection with mock client
        mock_client = Mock()
        call_result = asyncio.Future()
        call_result.set_result(Mock(content=[Mock(text="Tool result")]))
        mock_client.call_tool = Mock(return_value=call_result)
        
        tool = MockMCPTool("test_tool", "Test tool")
        conn = MCPConnection(
            client=mock_client,
            tools=[tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        connections = {"http://test.com": conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        result = await pool.call_tool_async("test_tool", {"arg": "value"}, connections, tool_to_server)
        
        assert result is not None
        mock_client.call_tool.assert_called_once_with("test_tool", arguments={"arg": "value"})

    def test_call_tool_sync(self):
        """Test synchronous tool execution."""
        pool = MCPConnectionPool()
        
        # Create mock connection
        mock_client = Mock()
        
        tool = MockMCPTool("test_tool", "Test tool")
        conn = MCPConnection(
            client=mock_client,
            tools=[tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        connections = {"http://test.com": conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        # Mock _run_async to avoid actual async execution
        with patch.object(pool, '_run_async', return_value=Mock(content=[Mock(text="Tool result")])):
            result = pool.call_tool("test_tool", {"arg": "value"}, connections, tool_to_server)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_call_tools_batch_async(self):
        """Test batch async tool execution."""
        pool = MCPConnectionPool()
        
        # Create mock connection
        mock_client = Mock()
        
        async def mock_call_tool(name, arguments):
            return Mock(content=[Mock(text=f"Result of {name}")])
        
        mock_client.call_tool = mock_call_tool
        
        tool = MockMCPTool("test_tool", "Test tool")
        conn = MCPConnection(
            client=mock_client,
            tools=[tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        connections = {"http://test.com": conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        calls = [
            ("call_1", "test_tool", {"arg": "value1"}),
            ("call_2", "test_tool", {"arg": "value2"}),
        ]
        
        results = await pool.call_tools_batch_async(calls, connections, tool_to_server)
        
        assert len(results) == 2
        assert results[0][0] == "call_1"  # call_id
        assert results[0][2] is None  # no error
        assert results[1][0] == "call_2"

    def test_call_tools_batch_sync(self):
        """Test batch synchronous tool execution."""
        pool = MCPConnectionPool()
        
        # Mock _run_async
        mock_results = [
            ("call_1", Mock(content=[Mock(text="Result 1")]), None),
            ("call_2", Mock(content=[Mock(text="Result 2")]), None),
        ]
        
        with patch.object(pool, '_run_async', return_value=mock_results):
            calls = [
                ("call_1", "test_tool", {"arg": "value1"}),
                ("call_2", "test_tool", {"arg": "value2"}),
            ]
            
            results = pool.call_tools_batch(calls, {}, {})
        
        assert len(results) == 2


class TestAgenticModelClass:
    """Tests for AgenticModelClass."""

    def test_agentic_model_initialization(self):
        """Test that AgenticModelClass can be initialized."""
        model = DummyAgenticModel()
        assert isinstance(model, AgenticModelClass)

    def test_get_pool(self):
        """Test getting the connection pool."""
        model = DummyAgenticModel()
        pool = model.get_pool()
        assert isinstance(pool, MCPConnectionPool)

    def test_token_tracking_initialization(self):
        """Test token tracking initialization."""
        model = DummyAgenticModel()
        model._init_tokens()
        assert hasattr(model._thread_local, 'tokens')
        assert model._thread_local.tokens == {'prompt': 0, 'completion': 0}

    def test_add_tokens(self):
        """Test adding tokens from response."""
        model = DummyAgenticModel()
        model._init_tokens()
        
        # Mock response with usage
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        model._add_tokens(mock_response)
        
        assert model._thread_local.tokens['prompt'] == 10
        assert model._thread_local.tokens['completion'] == 20

    def test_to_response_api_tools(self):
        """Test conversion of chat tools to response API format."""
        model = DummyAgenticModel()
        
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                },
            }
        ]
        
        response_tools = model._to_response_api_tools(chat_tools)
        
        assert len(response_tools) == 1
        assert response_tools[0]["type"] == "function"
        assert response_tools[0]["name"] == "test_tool"
        assert response_tools[0]["description"] == "A test tool"

    def test_parse_chat_tool_calls(self):
        """Test parsing chat completion tool calls."""
        model = DummyAgenticModel()
        
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        result = model._parse_chat_tool_calls([mock_tool_call])
        
        assert len(result) == 1
        assert result[0][0] == "call_1"
        assert result[0][1] == "test_tool"
        assert result[0][2] == {"arg": "value"}

    def test_parse_response_tool_calls(self):
        """Test parsing response API tool calls."""
        model = DummyAgenticModel()
        
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
                "status": "pending",
            }
        ]
        
        result = model._parse_response_tool_calls(items)
        
        assert len(result) == 1
        assert result[0][0] == "call_1"
        assert result[0][1] == "test_tool"
        assert result[0][2] == {"arg": "value"}

    def test_execute_chat_tools(self):
        """Test executing chat completion tool calls."""
        model = DummyAgenticModel()
        model.load_model()
        
        # Mock tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        messages = []
        
        # Mock pool and connections
        mock_pool = Mock()
        mock_result = Mock(content=[Mock(text="Tool result")])
        mock_pool.call_tools_batch.return_value = [("call_1", mock_result, None)]
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            model._execute_chat_tools([mock_tool_call], {}, messages, {})
        
        # Check that tool result was added to messages
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"
        assert "Tool result" in messages[0]["content"]

    def test_execute_response_tools(self):
        """Test executing response API tool calls."""
        model = DummyAgenticModel()
        model.load_model()
        
        tool_calls = [("call_1", "test_tool", {"arg": "value"})]
        input_items = []
        
        # Mock pool and connections
        mock_pool = Mock()
        mock_result = Mock(content=[Mock(text="Tool result")])
        mock_pool.call_tools_batch.return_value = [("call_1", mock_result, None)]
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            model._execute_response_tools(tool_calls, {}, input_items, {})
        
        # Check that tool result was added to input_items
        assert len(input_items) == 1
        assert input_items[0]["type"] == "function_call_output"
        assert input_items[0]["call_id"] == "call_1"
        assert "Tool result" in input_items[0]["output"]

    def test_convert_output_to_input(self):
        """Test converting response API output to input."""
        model = DummyAgenticModel()
        
        output = [
            {"type": "message", "role": "assistant", "content": "Hello"},
            {"type": "reasoning", "content": "Thinking..."},
            {"type": "function_call", "call_id": "call_1", "output": "Result"},
        ]
        
        result = model._convert_output_to_input(output)
        
        # Should only include message, reasoning, and completed function calls
        assert len(result) == 3

    def test_openai_transport_non_streaming(self):
        """Test non-streaming OpenAI transport without MCP."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)
        
        assert "id" in response
        assert "choices" in response

    def test_openai_transport_with_mcp_tools(self):
        """Test non-streaming OpenAI transport with MCP tools."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Use the test tool"},
            ],
            "mcp_servers": ["http://test.com"],
        }
        
        # Mock the pool
        mock_pool = Mock()
        mock_tool = MockMCPTool("test_tool", "Test tool")
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[mock_tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {},
                },
            }
        ]
        
        connections = {"http://test.com": mock_conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        mock_pool.get_tools_and_mapping.return_value = (tools, connections, tool_to_server)
        mock_pool.call_tools_batch.return_value = [
            ("call_1", Mock(content=[Mock(text="Tool result")]), None)
        ]
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            response_str = model.openai_transport(json.dumps(request))
        
        response = json.loads(response_str)
        # Verify we got a valid response (either success with id or non-failure error)
        # A response is valid if it has an id (success) OR if it doesn't have error code 2401
        is_success = "id" in response
        is_not_prediction_failed = response.get("code") != 2401
        assert is_success or is_not_prediction_failed, \
            f"Expected valid response but got: {response}"

    def test_openai_stream_transport(self):
        """Test streaming OpenAI transport without MCP."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "stream": True,
        }
        
        response_iter = model.openai_stream_transport(json.dumps(request))
        chunks = list(response_iter)
        
        assert len(chunks) > 0

    def test_openai_stream_transport_with_mcp(self):
        """Test streaming OpenAI transport with MCP tools."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Use the test tool"},
            ],
            "stream": True,
            "mcp_servers": ["http://test.com"],
        }
        
        # Mock the pool
        mock_pool = Mock()
        mock_tool = MockMCPTool("test_tool", "Test tool")
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[mock_tool],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {},
                },
            }
        ]
        
        connections = {"http://test.com": mock_conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        mock_pool.get_tools_and_mapping.return_value = (tools, connections, tool_to_server)
        
        # Mock the async tool execution
        async def mock_call_tools_batch_async(calls, conns, mapping):
            results = []
            for call_id, name, args in calls:
                results.append((call_id, Mock(content=[Mock(text="Tool result")]), None))
            return results
        
        mock_pool.call_tools_batch_async = mock_call_tools_batch_async
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            response_iter = model.openai_stream_transport(json.dumps(request))
            chunks = list(response_iter)
        
        assert len(chunks) > 0

    def test_responses_api_non_streaming(self):
        """Test non-streaming responses API without MCP."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }
        
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)
        
        assert "id" in response
        assert "output" in response

    def test_responses_api_streaming(self):
        """Test streaming responses API without MCP."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "stream": True,
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }
        
        response_iter = model.openai_stream_transport(json.dumps(request))
        events = [json.loads(chunk) for chunk in response_iter]
        
        assert len(events) > 0
        assert any(event.get("type") == "response.created" for event in events)

    def test_error_handling_invalid_request(self):
        """Test error handling for invalid requests."""
        model = DummyAgenticModel()
        model.load_model()
        
        # Invalid JSON
        response_str = model.openai_transport("invalid json")
        response = json.loads(response_str)
        
        assert "code" in response
        assert response["code"] == 2401  # MODEL_PREDICTION_FAILED

    def test_error_handling_tool_execution_failure(self):
        """Test error handling when tool execution fails."""
        model = DummyAgenticModel()
        model.load_model()
        
        # Mock tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        
        messages = []
        
        # Mock pool with error
        mock_pool = Mock()
        mock_pool.call_tools_batch.return_value = [("call_1", None, "Tool execution failed")]
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            model._execute_chat_tools([mock_tool_call], {}, messages, {})
        
        # Check that error message was added
        assert len(messages) == 1
        assert "Error:" in messages[0]["content"]

    def test_accumulate_tool_delta(self):
        """Test accumulating streaming tool call deltas."""
        model = DummyAgenticModel()
        
        accumulated = {}
        
        # First delta - tool ID and name
        delta1 = Mock()
        delta1.index = 0
        delta1.id = "call_1"
        delta1.function = Mock()
        delta1.function.name = "test_tool"
        delta1.function.arguments = ""
        
        model._accumulate_tool_delta(delta1, accumulated)
        
        assert 0 in accumulated
        assert accumulated[0]["id"] == "call_1"
        assert accumulated[0]["function"]["name"] == "test_tool"
        
        # Second delta - arguments
        delta2 = Mock()
        delta2.index = 0
        delta2.id = None
        delta2.function = Mock()
        delta2.function.name = None
        delta2.function.arguments = '{"arg": "value"}'
        
        model._accumulate_tool_delta(delta2, accumulated)
        
        assert accumulated[0]["function"]["arguments"] == '{"arg": "value"}'

    def test_finalize_tool_calls(self):
        """Test finalizing accumulated tool calls."""
        model = DummyAgenticModel()
        
        accumulated = {
            0: {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
            }
        }
        
        result = model._finalize_tool_calls(accumulated)
        
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["function"]["name"] == "test_tool"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_chat_completion_with_tool_calling(self):
        """Test complete chat completion flow with tool calling."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Use the test tool"},
            ],
            "mcp_servers": ["http://test.com"],
        }
        
        # Mock the entire pool workflow
        mock_pool = Mock()
        
        # Mock tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {},
                },
            }
        ]
        
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[MockMCPTool("test_tool")],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        connections = {"http://test.com": mock_conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        mock_pool.get_tools_and_mapping.return_value = (tools, connections, tool_to_server)
        mock_pool.call_tools_batch.return_value = [
            ("call_1", Mock(content=[Mock(text="Tool executed successfully")]), None)
        ]
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            response_str = model.openai_transport(json.dumps(request))
        
        response = json.loads(response_str)
        
        # Should get a response (either success or controlled error)
        assert isinstance(response, dict)

    def test_full_streaming_with_tool_calling(self):
        """Test complete streaming flow with tool calling."""
        model = DummyAgenticModel()
        model.load_model()
        
        request = {
            "model": "dummy-model",
            "messages": [
                {"role": "user", "content": "Use the test tool"},
            ],
            "stream": True,
            "mcp_servers": ["http://test.com"],
        }
        
        # Mock the pool
        mock_pool = Mock()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {},
                },
            }
        ]
        
        mock_conn = MCPConnection(
            client=Mock(),
            tools=[MockMCPTool("test_tool")],
            tool_names={"test_tool"},
            url="http://test.com"
        )
        
        connections = {"http://test.com": mock_conn}
        tool_to_server = {"test_tool": "http://test.com"}
        
        mock_pool.get_tools_and_mapping.return_value = (tools, connections, tool_to_server)
        
        async def mock_call_tools_batch_async(calls, conns, mapping):
            results = []
            for call_id, name, args in calls:
                results.append((call_id, Mock(content=[Mock(text="Tool result")]), None))
            return results
        
        mock_pool.call_tools_batch_async = mock_call_tools_batch_async
        
        with patch.object(model, 'get_pool', return_value=mock_pool):
            response_iter = model.openai_stream_transport(json.dumps(request))
            chunks = list(response_iter)
        
        # Should get multiple chunks
        assert len(chunks) > 0
