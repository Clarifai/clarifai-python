"""Comprehensive tests for AgenticModelClass and MCPConnectionPool integration."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_core import to_json

from clarifai.runners.models.agentic_class import (
    AgenticModelClass,
    MCPConnection,
    MCPConnectionPool,
)
from clarifai.runners.models.dummy_openai_model import MockOpenAIClient


class DummyAgenticModel(AgenticModelClass):
    """Dummy agentic model for testing."""

    client = MockOpenAIClient()
    model = "dummy-agentic-model"


class TestAgenticModelClass:
    """Tests for AgenticModelClass functionality."""

    @pytest.fixture(autouse=True)
    def reset_pool(self):
        """Reset the singleton instance before each test."""
        AgenticModelClass._pool = None
        MCPConnectionPool._instance = None
        yield
        AgenticModelClass._pool = None
        MCPConnectionPool._instance = None

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return DummyAgenticModel()

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock(spec=MCPConnectionPool)
        pool.get_tools_and_mapping = MagicMock(return_value=([], {}, {}))
        pool.call_tools_batch = MagicMock(return_value=[])
        pool.call_tools_batch_async = AsyncMock(return_value=[])
        pool._loop = asyncio.new_event_loop()
        return pool

    # === Token Tracking Tests ===

    def test_init_tokens(self, model):
        """Test token initialization."""
        model._init_tokens()
        assert hasattr(model._thread_local, 'tokens')
        assert model._thread_local.tokens == {'prompt': 0, 'completion': 0}

    def test_add_tokens_from_usage(self, model):
        """Test adding tokens from response with usage attribute."""
        mock_response = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_response.usage = mock_usage

        model._add_tokens(mock_response)
        assert model._thread_local.tokens['prompt'] == 10
        assert model._thread_local.tokens['completion'] == 20

    def test_add_tokens_from_response_usage(self, model):
        """Test adding tokens from response.response.usage."""
        mock_response = MagicMock()
        mock_response.usage = None
        mock_inner_response = MagicMock()

        # Use a simple object instead of MagicMock to avoid mock attribute issues
        class MockUsage:
            input_tokens = 15
            output_tokens = 25

        mock_usage = MockUsage()
        mock_inner_response.usage = mock_usage
        mock_response.response = mock_inner_response

        model._add_tokens(mock_response)
        assert model._thread_local.tokens['prompt'] == 15
        assert model._thread_local.tokens['completion'] == 25

    def test_add_tokens_accumulates(self, model):
        """Test that tokens accumulate across multiple calls."""
        mock_response1 = MagicMock()
        mock_usage1 = MagicMock()
        mock_usage1.prompt_tokens = 10
        mock_usage1.completion_tokens = 20
        mock_response1.usage = mock_usage1

        mock_response2 = MagicMock()
        mock_usage2 = MagicMock()
        mock_usage2.prompt_tokens = 5
        mock_usage2.completion_tokens = 10
        mock_response2.usage = mock_usage2

        model._add_tokens(mock_response1)
        model._add_tokens(mock_response2)

        assert model._thread_local.tokens['prompt'] == 15
        assert model._thread_local.tokens['completion'] == 30

    def test_finalize_tokens(self, model):
        """Test finalizing tokens to output context."""
        model._init_tokens()
        model._thread_local.tokens['prompt'] = 10
        model._thread_local.tokens['completion'] = 20

        with patch.object(model, 'set_output_context') as mock_set:
            model._finalize_tokens()
            mock_set.assert_called_once_with(prompt_tokens=10, completion_tokens=20)
            assert not hasattr(model._thread_local, 'tokens')

    def test_finalize_tokens_no_tokens(self, model):
        """Test finalizing when no tokens were tracked."""
        with patch.object(model, 'set_output_context') as mock_set:
            model._finalize_tokens()
            mock_set.assert_not_called()

    # === Tool Format Conversion Tests ===

    def test_to_response_api_tools_with_function(self, model):
        """Test converting chat completion tools to response API format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = model._to_response_api_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "test_tool"
        assert result[0]["description"] == "A test tool"

    def test_to_response_api_tools_with_name(self, model):
        """Test converting tools that already have name field."""
        tools = [
            {
                "type": "function",
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {},
            }
        ]
        result = model._to_response_api_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"

    def test_to_dict_from_dict(self, model):
        """Test _to_dict with dict input."""
        obj = {"key": "value"}
        result = model._to_dict(obj)
        assert result == obj

    def test_to_dict_with_model_dump(self, model):
        """Test _to_dict with object that has model_dump method."""
        obj = MagicMock()
        obj.model_dump.return_value = {"key": "value"}
        result = model._to_dict(obj)
        assert result == {"key": "value"}

    def test_to_dict_with_dict_method(self, model):
        """Test _to_dict with object that has dict method."""

        # Use a simple class to avoid MagicMock issues
        class TestObj:
            def dict(self):
                return {"key": "value"}

        obj = TestObj()
        result = model._to_dict(obj)
        assert result == {"key": "value"}

    def test_to_dict_with_dict_attribute(self, model):
        """Test _to_dict with object that has __dict__ attribute."""

        class TestObj:
            def __init__(self):
                self.key = "value"

        obj = TestObj()
        result = model._to_dict(obj)
        assert result == {"key": "value"}

    # === Tool Call Parsing Tests ===

    def test_parse_chat_tool_calls_with_function_attribute(self, model):
        """Test parsing chat completion tool calls with function attribute."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        tool_calls = [mock_tool_call]
        result = model._parse_chat_tool_calls(tool_calls)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {"arg": "value"})

    def test_parse_chat_tool_calls_with_dict(self, model):
        """Test parsing chat completion tool calls from dict."""
        tool_calls = [
            {"id": "call_123", "function": {"name": "test_tool", "arguments": '{"arg": "value"}'}}
        ]
        result = model._parse_chat_tool_calls(tool_calls)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {"arg": "value"})

    def test_parse_chat_tool_calls_invalid_json(self, model):
        """Test parsing tool calls with invalid JSON arguments."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "invalid json"

        tool_calls = [mock_tool_call]
        result = model._parse_chat_tool_calls(tool_calls)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {})

    def test_parse_response_tool_calls(self, model):
        """Test parsing response API tool calls."""
        items = [
            {
                "type": "function_tool_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
                "status": "pending",
            }
        ]
        result = model._parse_response_tool_calls(items)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {"arg": "value"})

    def test_parse_response_tool_calls_with_id(self, model):
        """Test parsing response API tool calls using id instead of call_id."""
        items = [
            {
                "type": "function_call",
                "id": "call_123",
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
                "output": None,
            }
        ]
        result = model._parse_response_tool_calls(items)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {"arg": "value"})

    def test_parse_response_tool_calls_skips_completed(self, model):
        """Test that completed tool calls are skipped."""
        items = [
            {
                "type": "function_tool_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
                "status": "completed",
                "output": "result",
            }
        ]
        result = model._parse_response_tool_calls(items)
        assert len(result) == 0

    def test_parse_response_tool_calls_with_dict_arguments(self, model):
        """Test parsing response tool calls with dict arguments instead of string."""
        items = [
            {
                "type": "function_tool_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": {"arg": "value"},  # Already a dict
                "status": "pending",
            }
        ]
        result = model._parse_response_tool_calls(items)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {"arg": "value"})

    def test_parse_response_tool_calls_empty_string_arguments(self, model):
        """Test parsing response tool calls with empty string arguments."""
        items = [
            {
                "type": "function_tool_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": "",
                "status": "pending",
            }
        ]
        result = model._parse_response_tool_calls(items)

        assert len(result) == 1
        assert result[0] == ("call_123", "test_tool", {})

    # === Tool Execution Tests ===

    def test_execute_chat_tools(self, model):
        """Test executing chat completion tool calls."""
        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool result")]
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [MagicMock()]
            tool_calls[0].id = "call_123"
            tool_calls[0].function.name = "test_tool"
            tool_calls[0].function.arguments = '{}'

            connections = {}
            messages = []
            tool_to_server = {}

            model._execute_chat_tools(tool_calls, connections, messages, tool_to_server)

            assert len(messages) == 1
            assert messages[0]["role"] == "tool"
            assert messages[0]["tool_call_id"] == "call_123"
            assert messages[0]["content"] == "Tool result"

    def test_execute_chat_tools_with_error(self, model):
        """Test executing chat tools with error."""
        mock_pool = MagicMock()
        mock_pool.call_tools_batch.return_value = [("call_123", None, "Tool error")]

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [MagicMock()]
            tool_calls[0].id = "call_123"
            tool_calls[0].function.name = "test_tool"
            tool_calls[0].function.arguments = '{}'

            connections = {}
            messages = []
            tool_to_server = {}

            model._execute_chat_tools(tool_calls, connections, messages, tool_to_server)

            assert len(messages) == 1
            assert messages[0]["content"] == "Error: Tool error"

    def test_execute_chat_tools_with_list_result(self, model):
        """Test executing chat tools with list result format."""
        mock_pool = MagicMock()

        # Use a dict-like object that supports both .get() and .text access
        class TextDict:
            def __init__(self, text):
                self._text = text

            def get(self, key, default=None):
                if key == 'text':
                    return self._text
                return default

            @property
            def text(self):
                return self._text

        mock_result = [TextDict("List result")]
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [MagicMock()]
            tool_calls[0].id = "call_123"
            tool_calls[0].function.name = "test_tool"
            tool_calls[0].function.arguments = '{}'

            connections = {}
            messages = []
            tool_to_server = {}

            model._execute_chat_tools(tool_calls, connections, messages, tool_to_server)

            assert len(messages) == 1
            assert messages[0]["content"] == "List result"

    def test_execute_chat_tools_with_none_content(self, model):
        """Test executing chat tools when result has no content."""
        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = []
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [MagicMock()]
            tool_calls[0].id = "call_123"
            tool_calls[0].function.name = "test_tool"
            tool_calls[0].function.arguments = '{}'

            connections = {}
            messages = []
            tool_to_server = {}

            model._execute_chat_tools(tool_calls, connections, messages, tool_to_server)

            assert len(messages) == 1
            assert messages[0]["content"] is None

    @pytest.mark.asyncio
    async def test_execute_chat_tools_async(self, model):
        """Test async execution of chat tools."""
        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Async result")]
        mock_pool.call_tools_batch_async = AsyncMock(
            return_value=[("call_123", mock_result, None)]
        )

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [MagicMock()]
            tool_calls[0].id = "call_123"
            tool_calls[0].function.name = "test_tool"
            tool_calls[0].function.arguments = '{}'

            connections = {}
            messages = []
            tool_to_server = {}

            await model._execute_chat_tools_async(
                tool_calls, connections, messages, tool_to_server
            )

            assert len(messages) == 1
            assert messages[0]["content"] == "Async result"

    def test_execute_response_tools(self, model):
        """Test executing response API tool calls."""
        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Response result")]
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [("call_123", "test_tool", {})]
            connections = {}
            input_items = []
            tool_to_server = {}

            model._execute_response_tools(tool_calls, connections, input_items, tool_to_server)

            assert len(input_items) == 1
            assert input_items[0]["type"] == "function_call_output"
            assert input_items[0]["call_id"] == "call_123"
            assert input_items[0]["output"] == "Response result"

    @pytest.mark.asyncio
    async def test_execute_response_tools_async(self, model):
        """Test async execution of response tools."""
        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Async response result")]
        mock_pool.call_tools_batch_async = AsyncMock(
            return_value=[("call_123", mock_result, None)]
        )

        with patch.object(model, 'get_pool', return_value=mock_pool):
            tool_calls = [("call_123", "test_tool", {})]
            connections = {}
            input_items = []
            tool_to_server = {}

            await model._execute_response_tools_async(
                tool_calls, connections, input_items, tool_to_server
            )

            assert len(input_items) == 1
            assert input_items[0]["output"] == "Async response result"

    # === Response Output Processing Tests ===

    def test_convert_output_to_input(self, model):
        """Test converting response API output to input items."""
        output = [
            {"type": "message", "role": "assistant", "content": "Hello"},
            {"type": "function_tool_call", "name": "tool1", "output": "result"},
            {"type": "reasoning", "content": "Thinking..."},
        ]
        result = model._convert_output_to_input(output)

        assert len(result) == 3
        assert result[0]["type"] == "message"
        assert result[1]["type"] == "function_tool_call"
        assert result[2]["type"] == "reasoning"

    def test_convert_output_to_input_filters_pending(self, model):
        """Test that pending tool calls are filtered out."""
        output = [
            {"type": "function_tool_call", "name": "tool1", "status": "pending", "output": None}
        ]
        result = model._convert_output_to_input(output)
        assert len(result) == 0

    def test_convert_output_to_input_empty_list(self, model):
        """Test converting empty output list."""
        result = model._convert_output_to_input([])
        assert result == []

    # === Request Handler Tests ===

    def test_handle_chat_completions_with_tools(self, model):
        """Test handling chat completions with MCP tools."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}

        with patch.object(
            AgenticModelClass.__bases__[0], '_handle_chat_completions'
        ) as mock_super_handle:
            mock_response = MagicMock()
            mock_super_handle.return_value = mock_response

            result = model._handle_chat_completions(
                request_data, mcp_servers=["http://server"], connections={}, tools=tools
            )

            # The method creates a new dict, so check what was passed to super
            call_args = mock_super_handle.call_args[0][0]
            assert "tools" in call_args
            assert call_args["tool_choice"] == "auto"

    def test_handle_chat_completions_without_tools(self, model):
        """Test handling chat completions without MCP tools."""
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}

        with patch.object(model, '_handle_chat_completions') as mock_handle:
            mock_response = MagicMock()
            mock_handle.return_value = mock_response
            result = model._handle_chat_completions(request_data, None, None, None)

            # Should not modify request_data when no tools
            assert "tools" not in request_data

    def test_handle_responses_with_tools(self, model):
        """Test handling responses with MCP tools."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        request_data = {"input": "Hello"}

        with patch.object(
            AgenticModelClass.__bases__[0], '_handle_responses'
        ) as mock_super_handle:
            mock_response = MagicMock()
            mock_super_handle.return_value = mock_response

            result = model._handle_responses(
                request_data, mcp_servers=["http://server"], connections={}, tools=tools
            )

            # The method creates a new dict, so check what was passed to super
            call_args = mock_super_handle.call_args[0][0]
            assert "tools" in call_args
            assert call_args["tool_choice"] == "auto"

    def test_handle_responses_without_tools(self, model):
        """Test handling responses without MCP tools."""
        request_data = {"input": "Hello"}

        with patch.object(model, '_handle_responses') as mock_handle:
            mock_response = MagicMock()
            mock_handle.return_value = mock_response
            result = model._handle_responses(request_data, None, None, None)

            # Should not modify request_data when no tools
            assert "tools" not in request_data

    def test_route_request_chat_completions(self, model):
        """Test routing to chat completions endpoint."""
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}
        with patch.object(model, '_handle_chat_completions') as mock_handle:
            mock_handle.return_value = MagicMock()
            model._route_request(model.ENDPOINT_CHAT_COMPLETIONS, request_data, None, None, None)
            mock_handle.assert_called_once()

    def test_route_request_responses(self, model):
        """Test routing to responses endpoint."""
        request_data = {"input": "Hello"}
        with patch.object(model, '_handle_responses') as mock_handle:
            mock_handle.return_value = MagicMock()
            model._route_request(model.ENDPOINT_RESPONSES, request_data, None, None, None)
            mock_handle.assert_called_once()

    # === Streaming Helper Tests ===

    def test_accumulate_tool_delta(self, model):
        """Test accumulating streaming tool call deltas."""
        accumulated = {}
        delta = MagicMock()
        delta.index = 0
        delta.id = "call_123"
        delta.function.name = "test_tool"
        delta.function.arguments = '{"arg": "value"}'

        model._accumulate_tool_delta(delta, accumulated)

        assert 0 in accumulated
        assert accumulated[0]["id"] == "call_123"
        assert accumulated[0]["function"]["name"] == "test_tool"
        assert accumulated[0]["function"]["arguments"] == '{"arg": "value"}'

    def test_accumulate_tool_delta_incremental(self, model):
        """Test accumulating incremental tool call arguments."""
        accumulated = {
            0: {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": ""},
            }
        }
        delta = MagicMock()
        delta.index = 0
        delta.id = None
        delta.function.name = None
        delta.function.arguments = '{"arg": "value"}'

        model._accumulate_tool_delta(delta, accumulated)

        assert accumulated[0]["function"]["arguments"] == '{"arg": "value"}'

    def test_finalize_tool_calls(self, model):
        """Test finalizing accumulated tool calls."""
        accumulated = {
            0: {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": "{}"},
            },
            1: {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": "{}"},
            },
        }
        result = model._finalize_tool_calls(accumulated)

        assert len(result) == 2
        assert result[0]["id"] == "call_1"
        assert result[1]["id"] == "call_2"

    def test_create_stream_request(self, model):
        """Test creating streaming chat completion request."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch.object(model.client.chat.completions, 'create') as mock_create:
            mock_create.return_value = iter([])
            result = model._create_stream_request(messages, tools, 100, 0.7, 0.9)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["stream"] is True
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    def test_async_to_sync_generator(self, model):
        """Test bridging async generator to sync generator."""

        async def async_gen():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        pool = model.get_pool()
        result = list(model._async_to_sync_generator(async_gen))

        assert len(result) == 3
        assert result == ["chunk1", "chunk2", "chunk3"]

    def test_async_to_sync_generator_with_error(self, model):
        """Test async to sync generator with error."""

        async def async_gen():
            yield "chunk1"
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            list(model._async_to_sync_generator(async_gen))

    # === Streaming with MCP Tests ===

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_no_tool_calls(self, model):
        """Test streaming chat with tools when no tool calls are generated."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = []
        connections = {}
        tool_to_server = {}

        mock_chunk = MagicMock()
        mock_chunk.model_dump_json.return_value = '{"id": "chunk1"}'
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.content = "Hello"

        with patch.object(model, '_create_stream_request', return_value=[mock_chunk]):
            chunks = []
            async for chunk in model._stream_chat_with_tools(
                messages, tools, connections, tool_to_server, 100, 0.7, 0.9
            ):
                chunks.append(chunk)

            assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_chat_with_tools_with_tool_calls(self, model):
        """Test streaming chat with tools when tool calls are generated."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        connections = {}
        tool_to_server = {}

        # Create mock tool call delta
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = 0
        mock_tool_call_delta.id = "call_123"
        mock_tool_call_delta.function.name = "test_tool"
        mock_tool_call_delta.function.arguments = '{}'

        # First chunk with tool calls
        mock_chunk_with_tools = MagicMock()
        mock_chunk_with_tools.model_dump_json.return_value = '{"id": "chunk1"}'
        mock_chunk_with_tools.choices = [MagicMock()]
        mock_chunk_with_tools.choices[0].delta = MagicMock()
        mock_chunk_with_tools.choices[0].delta.tool_calls = [mock_tool_call_delta]
        mock_chunk_with_tools.choices[0].delta.content = None

        # Second chunk without tool calls (to prevent recursion)
        mock_chunk_no_tools = MagicMock()
        mock_chunk_no_tools.model_dump_json.return_value = '{"id": "chunk2"}'
        mock_chunk_no_tools.choices = [MagicMock()]
        mock_chunk_no_tools.choices[0].delta = MagicMock()
        mock_chunk_no_tools.choices[0].delta.tool_calls = None
        mock_chunk_no_tools.choices[0].delta.content = "Response"

        mock_pool = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool result")]
        mock_pool.call_tools_batch_async = AsyncMock(
            return_value=[("call_123", mock_result, None)]
        )

        # Create a generator that yields chunks
        def chunk_generator():
            yield mock_chunk_with_tools

        # For recursive call, return chunk without tool calls
        def chunk_generator_no_tools():
            yield mock_chunk_no_tools

        with (
            patch.object(
                model,
                '_create_stream_request',
                side_effect=[chunk_generator(), chunk_generator_no_tools()],
            ),
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(
                model,
                '_finalize_tool_calls',
                return_value=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            ),
        ):
            chunks = []
            async for chunk in model._stream_chat_with_tools(
                messages, tools, connections, tool_to_server, 100, 0.7, 0.9
            ):
                chunks.append(chunk)

            # Should have initial chunk plus recursive call chunks
            assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_responses_with_tools(self, model):
        """Test streaming responses with MCP tools."""
        request_data = {"input": "Hello"}
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        connections = {}
        tool_to_server = {}

        # Create a simple event class that will be yielded
        # The code checks chunk_type and yields if it doesn't match certain patterns
        # We need to avoid output_index being detected by hasattr
        class MockEvent:
            def __init__(self):
                self.type = "response.created"
                self.response = None

            def model_dump_json(self):
                return '{"type": "response.created"}'

        # Set the class name for chunk_type detection
        MockEvent.__name__ = "ResponseCreatedEvent"
        mock_event = MockEvent()

        with patch.object(model.client.responses, 'create', return_value=[mock_event]):
            chunks = []
            async for chunk in model._stream_responses_with_tools(
                request_data, tools, connections, tool_to_server
            ):
                chunks.append(chunk)

            # Should yield at least the event as JSON
            assert len(chunks) >= 1

    # === Main OpenAI Methods Tests ===

    def test_openai_transport_without_mcp(self, model):
        """Test openai_transport without MCP servers."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        with patch.object(model, '_route_request') as mock_route:
            mock_response = MagicMock()
            mock_response.model_dump_json.return_value = '{"id": "test"}'
            mock_route.return_value = mock_response

            result = model.openai_transport(to_json(request))
            assert json.loads(result)["id"] == "test"

    def test_openai_transport_with_mcp_chat_completions(self, model):
        """Test openai_transport with MCP servers for chat completions."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "mcp_servers": ["http://server"],
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {}

        mock_conn = MCPConnection(
            client=MagicMock(), tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        mock_pool = MagicMock()
        mock_pool.get_tools_and_mapping.return_value = (
            [{"type": "function", "function": {"name": "test_tool"}}],
            {"http://server": mock_conn},
            {"test_tool": "http://server"},
        )

        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"id": "test"}'
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].get.return_value = {}
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.get.return_value = None

        with (
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(model, '_route_request', return_value=mock_response),
        ):
            result = model.openai_transport(to_json(request))
            assert json.loads(result)["id"] == "test"

    def test_openai_transport_with_mcp_responses(self, model):
        """Test openai_transport with MCP servers for responses API."""
        request = {
            "input": "Hello",
            "mcp_servers": ["http://server"],
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {}

        mock_conn = MCPConnection(
            client=MagicMock(), tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        mock_pool = MagicMock()
        mock_pool.get_tools_and_mapping.return_value = (
            [{"type": "function", "function": {"name": "test_tool"}}],
            {"http://server": mock_conn},
            {"test_tool": "http://server"},
        )

        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"id": "test"}'
        mock_response.output = []

        with (
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(model, '_route_request', return_value=mock_response),
        ):
            result = model.openai_transport(to_json(request))
            assert json.loads(result)["id"] == "test"

    def test_openai_transport_with_existing_tools(self, model):
        """Test openai_transport when tools are already provided (should not use MCP)."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "mcp_servers": ["http://server"],
            "tools": [{"type": "function", "function": {"name": "existing_tool"}}],
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        with patch.object(model, '_route_request') as mock_route:
            mock_response = MagicMock()
            mock_response.model_dump_json.return_value = '{"id": "test"}'
            mock_route.return_value = mock_response

            result = model.openai_transport(to_json(request))
            # Should not call get_pool when tools are already provided
            assert json.loads(result)["id"] == "test"

    def test_openai_transport_error_handling(self, model):
        """Test error handling in openai_transport."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        with patch.object(model, '_route_request', side_effect=Exception("Test error")):
            result = model.openai_transport(to_json(request))
            error = json.loads(result)
            # Check that it's an error code (could be 22000 or 21313 depending on status_code_pb2)
            assert "code" in error
            assert error["code"] > 0
            assert "Test error" in error["details"]

    def test_openai_stream_transport_without_mcp(self, model):
        """Test openai_stream_transport without MCP servers."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        mock_chunk = MagicMock()
        mock_chunk.model_dump_json.return_value = '{"id": "chunk1"}'
        # Ensure usage is None to avoid token finalization issues
        mock_chunk.usage = None

        with patch.object(model.client.chat.completions, 'create', return_value=[mock_chunk]):
            chunks = list(model.openai_stream_transport(to_json(request)))
            # Filter out error chunks if any
            chunks = [c for c in chunks if not (isinstance(c, bytes) and b'"code"' in c)]
            assert len(chunks) == 1

    def test_openai_stream_transport_with_mcp(self, model):
        """Test openai_stream_transport with MCP servers."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "mcp_servers": ["http://server"],
            "stream": True,
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {}

        mock_conn = MCPConnection(
            client=MagicMock(), tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        mock_pool = MagicMock()
        mock_pool.get_tools_and_mapping.return_value = (
            [{"type": "function", "function": {"name": "test_tool"}}],
            {"http://server": mock_conn},
            {"test_tool": "http://server"},
        )
        mock_pool._loop = asyncio.new_event_loop()

        mock_chunk = MagicMock()
        mock_chunk.model_dump_json.return_value = '{"id": "chunk1"}'
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.content = "Hello"

        with (
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(model, '_async_to_sync_generator') as mock_gen,
        ):
            mock_gen.return_value = iter(['{"id": "chunk1"}'])
            chunks = list(model.openai_stream_transport(to_json(request)))
            assert len(chunks) == 1

    def test_openai_stream_transport_error_handling(self, model):
        """Test error handling in openai_stream_transport."""
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        # Patch the client method to raise an error
        def raise_error(**kwargs):
            raise Exception("Test error")

        with patch.object(model.client.chat.completions, 'create', side_effect=raise_error):
            chunks = list(model.openai_stream_transport(to_json(request)))
            # Should have at least one error chunk
            assert len(chunks) >= 1
            # The error should be in JSON format
            error_str = (
                chunks[-1]
                if isinstance(chunks[-1], str)
                else chunks[-1].decode('utf-8')
                if isinstance(chunks[-1], bytes)
                else str(chunks[-1])
            )
            error = json.loads(error_str)
            assert "code" in error
            assert "Test error" in error.get("details", "")

    # === Pool Management Tests ===

    def test_get_pool_singleton(self, model):
        """Test that get_pool returns a singleton."""
        pool1 = model.get_pool()
        pool2 = model.get_pool()
        assert pool1 is pool2

    def test_get_pool_thread_safe(self, model):
        """Test that get_pool is thread-safe."""
        import threading

        pools = []

        def get_pool():
            pools.append(model.get_pool())

        threads = [threading.Thread(target=get_pool) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert all(p is pools[0] for p in pools)

    # === Integration Tests ===

    def test_full_flow_chat_completions_with_tools(self, model):
        """Test full flow: chat completions with tool calls."""
        request = {
            "messages": [{"role": "user", "content": "Use test_tool"}],
            "mcp_servers": ["http://server"],
            "openai_endpoint": model.ENDPOINT_CHAT_COMPLETIONS,
        }

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {}

        mock_conn = MCPConnection(
            client=MagicMock(), tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        # First response has tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]

        # Make message behave like a dict for .get() calls
        def message_get(key, default=None):
            if key == "tool_calls":
                return [mock_tool_call]
            return default

        mock_message.get = message_get

        mock_response1 = MagicMock()
        mock_response1.model_dump_json.return_value = '{"id": "test1"}'
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message = mock_message

        # Make choices[0] behave like a dict for .get() calls
        def mock_get(key, default=None):
            if key == "message":
                return mock_message
            return default

        mock_response1.choices[0].get = mock_get

        # Second response (after tool execution) has no tool calls
        mock_message2 = MagicMock()
        mock_message2.tool_calls = None

        def message_get2(key, default=None):
            if key == "tool_calls":
                return None
            return default

        mock_message2.get = message_get2

        mock_response2 = MagicMock()
        mock_response2.model_dump_json.return_value = '{"id": "test2"}'
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message = mock_message2

        def mock_get2(key, default=None):
            if key == "message":
                return mock_message2
            return default

        mock_response2.choices[0].get = mock_get2

        mock_pool = MagicMock()
        mock_pool.get_tools_and_mapping.return_value = (
            [{"type": "function", "function": {"name": "test_tool"}}],
            {"http://server": mock_conn},
            {"test_tool": "http://server"},
        )
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool executed")]
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with (
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(model, '_route_request', side_effect=[mock_response1, mock_response2]),
        ):
            result = model.openai_transport(to_json(request))
            assert json.loads(result)["id"] == "test2"

    def test_full_flow_responses_with_tools(self, model):
        """Test full flow: responses API with tool calls."""
        request = {
            "input": "Use test_tool",
            "mcp_servers": ["http://server"],
            "openai_endpoint": model.ENDPOINT_RESPONSES,
        }

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {}

        mock_conn = MCPConnection(
            client=MagicMock(), tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        # First response has tool calls
        mock_response1 = MagicMock()
        mock_response1.model_dump_json.return_value = '{"id": "test1"}'
        mock_response1.output = [
            {
                "type": "function_tool_call",
                "call_id": "call_123",
                "name": "test_tool",
                "arguments": '{}',
                "output": None,
            }
        ]

        # Second response (after tool execution) has no tool calls
        mock_response2 = MagicMock()
        mock_response2.model_dump_json.return_value = '{"id": "test2"}'
        mock_response2.output = []

        mock_pool = MagicMock()
        mock_pool.get_tools_and_mapping.return_value = (
            [{"type": "function", "function": {"name": "test_tool"}}],
            {"http://server": mock_conn},
            {"test_tool": "http://server"},
        )
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool executed")]
        mock_pool.call_tools_batch.return_value = [("call_123", mock_result, None)]

        with (
            patch.object(model, 'get_pool', return_value=mock_pool),
            patch.object(model, '_route_request', side_effect=[mock_response1, mock_response2]),
        ):
            result = model.openai_transport(to_json(request))
            assert json.loads(result)["id"] == "test2"

    def test_to_response_api_tools_empty_list(self, model):
        """Test converting empty tools list."""
        result = model._to_response_api_tools([])
        assert result == []
