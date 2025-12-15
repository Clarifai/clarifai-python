"""Dummy Agentic model implementation for testing.

This module provides mock implementations of MCP (Model Context Protocol) components
and an AgenticModelClass for testing purposes. The mocks simulate the behavior of
actual MCP servers and tool execution without requiring real network connections.

Key Components:
    - MockMCPTool: Simulates MCP tool definitions
    - MockMCPToolResult: Simulates tool execution results
    - MockMCPClient: Simulates MCP client connections
    - MockOpenAIClientWithTools: Extended OpenAI client that supports tool calls
    - MockCompletionWithTools: Simulates chat completions with tool calling
    - MockCompletionStreamWithTools: Simulates streaming chat with tool calls
    - MockResponseWithTools: Simulates response API with tool calling
    - MockResponseStreamWithTools: Simulates streaming responses with tool calls
    - DummyAgenticModel: Test implementation of AgenticModelClass

The mock implementations are designed to work with the test suite in
tests/runners/test_agentic_model.py and simulate realistic tool calling scenarios
including:
    - Tool discovery and selection
    - Tool call execution
    - Streaming and non-streaming modes
    - Multiple tool iterations
    - Error scenarios
"""

import asyncio
import json
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock

from clarifai.runners.models.agentic_class import AgenticModelClass, MCPConnection
from clarifai.runners.models.dummy_openai_model import (
    MockCompletion,
    MockCompletionStream,
    MockOpenAIClient,
    MockResponse,
    MockResponseStream,
)


class MockMCPTool:
    """Mock MCP tool for testing."""

    def __init__(self, name: str, description: str = "", parameters: dict = None):
        self.name = name
        self.description = description
        self.inputSchema = parameters or {}


class MockMCPToolResult:
    """Mock MCP tool call result."""

    class Content:
        def __init__(self, text: str):
            self.text = text

    def __init__(self, text: str):
        self.content = [self.Content(text)]


class MockMCPClient:
    """Mock MCP client for testing."""

    def __init__(self, url: str, tools: List[MockMCPTool] = None):
        self.url = url
        self._tools = tools or [
            MockMCPTool(
                "test_tool",
                "A test tool",
                {"type": "object", "properties": {"arg1": {"type": "string"}}},
            )
        ]
        self._is_open = False

    async def __aenter__(self):
        self._is_open = True
        return self

    async def __aexit__(self, *args):
        self._is_open = False

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        """Simulate tool execution."""
        return MockMCPToolResult(f"Result of {name}: {json.dumps(arguments)}")

    async def close(self):
        self._is_open = False


class MockOpenAIClientWithTools(MockOpenAIClient):
    """Extended mock client that supports tool calls."""

    class CompletionsWithTools:
        def create(self, **kwargs):
            """Mock create method with tool call support."""
            if kwargs.get("stream", False):
                return MockCompletionStreamWithTools(**kwargs)
            else:
                return MockCompletionWithTools(**kwargs)

    class ResponsesWithTools:
        def create(self, **kwargs):
            """Mock create method for responses API with tools."""
            if kwargs.get("stream", False):
                return MockResponseStreamWithTools(**kwargs)
            else:
                return MockResponseWithTools(**kwargs)

    def __init__(self):
        super().__init__()
        self.completions = self.CompletionsWithTools()
        self.responses = self.ResponsesWithTools()


class MockCompletionWithTools(MockCompletion):
    """Mock completion with tool calls."""

    class ToolCall:
        class Function:
            def __init__(self, name: str, arguments: str):
                self.name = name
                self.arguments = arguments

        def __init__(self, tool_id: str, name: str, arguments: dict):
            self.id = tool_id
            self.type = "function"
            self.function = self.Function(name, json.dumps(arguments))

    class ChoiceWithTools(MockCompletion.Choice):
        def __init__(self, content, tool_calls=None):
            super().__init__(content)
            self.message.tool_calls = tool_calls

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Check if tools are provided and should be called
        tools = kwargs.get("tools", [])
        messages = kwargs.get("messages", [])
        
        if tools and not any(msg.get("role") == "tool" for msg in messages):
            # First call - trigger tool use
            tool_calls = [
                self.ToolCall("call_1", "test_tool", {"arg1": "test_value"})
            ]
            self.choices = [self.ChoiceWithTools("", tool_calls)]
        else:
            # After tool results - normal response
            self.choices = [self.Choice("Response after tool call")]


class MockCompletionStreamWithTools(MockCompletionStream):
    """Mock streaming completion with tool calls."""

    class ChunkWithTools(MockCompletionStream.Chunk):
        class ChoiceWithTools(MockCompletionStream.Chunk.Choice):
            class DeltaWithTools(MockCompletionStream.Chunk.Choice.Delta):
                class ToolCallDelta:
                    class FunctionDelta:
                        def __init__(self, name: str = "", arguments: str = ""):
                            self.name = name
                            self.arguments = arguments

                    def __init__(self, index: int, tool_id: str = "", name: str = "", arguments: str = ""):
                        self.index = index
                        self.id = tool_id
                        self.type = "function" if tool_id else None
                        self.function = self.FunctionDelta(name, arguments) if (name or arguments) else None

                def __init__(self, content=None, tool_calls=None):
                    super().__init__(content)
                    self.tool_calls = tool_calls

            def __init__(self, content=None, tool_calls=None, include_usage=False):
                self.delta = self.DeltaWithTools(content, tool_calls)
                self.finish_reason = None if (content or tool_calls) else "stop"
                self.index = 0
                self.usage = (
                    self.Usage(**{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                    if include_usage
                    else self.Usage(None, None, None)
                )

        def __init__(self, content=None, tool_calls=None, include_usage=False):
            self.choices = [self.ChoiceWithTools(content, tool_calls, include_usage)]
            self.id = "dummy-chunk-id"
            self.created = 1234567890
            self.model = "dummy-model"
            self.usage = self.choices[0].usage

    def __init__(self, **kwargs):
        # Don't call super().__init__ - we'll override everything
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])
        
        self.chunks = []
        self.include_usage = kwargs.get("stream_options", {}).get("include_usage")
        
        # Check if we should emit tool calls or regular content
        if tools and not any(msg.get("role") == "tool" for msg in messages):
            # Emit tool call chunks
            tool_call_delta_1 = self.ChunkWithTools.ChoiceWithTools.DeltaWithTools.ToolCallDelta(
                index=0, tool_id="call_1", name="test_tool"
            )
            tool_call_delta_2 = self.ChunkWithTools.ChoiceWithTools.DeltaWithTools.ToolCallDelta(
                index=0, arguments='{"arg1": "test_value"}'
            )
            self.chunks = [
                (None, [tool_call_delta_1]),
                (None, [tool_call_delta_2]),
                (None, None),  # Final chunk
            ]
        else:
            # Emit regular content after tool results
            self.chunks = [
                ("Response after tool call", None),
                ("", None),
            ]
        
        self.current_chunk = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk < len(self.chunks):
            content, tool_calls = self.chunks[self.current_chunk]
            chunk = self.ChunkWithTools(content, tool_calls, self.include_usage)
            self.current_chunk += 1
            return chunk
        else:
            raise StopIteration


class MockResponseWithTools(MockResponse):
    """Mock response with tool calls."""

    class OutputWithTools(MockResponse.Output):
        def __init__(self, content_text=None, tool_call=None):
            if tool_call:
                self.type = "function_call"
                self.call_id = tool_call["call_id"]
                self.name = tool_call["name"]
                self.arguments = tool_call["arguments"]
            else:
                super().__init__(content_text)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        tools = kwargs.get("tools", [])
        input_data = kwargs.get("input", "")
        
        # Check if input contains tool results
        has_tool_results = False
        if isinstance(input_data, list):
            has_tool_results = any(
                item.get("type") == "function_call_output"
                for item in input_data
            )
        
        if tools and not has_tool_results:
            # First call - trigger tool use
            self.output = [
                self.OutputWithTools(
                    tool_call={
                        "call_id": "call_1",
                        "name": "test_tool",
                        "arguments": '{"arg1": "test_value"}',
                    }
                )
            ]
        else:
            # After tool results - normal response
            self.output = [self.Output("Response after tool call")]


class MockResponseStreamWithTools(MockResponseStream):
    """Mock streaming response with tool calls."""

    def __init__(self, **kwargs):
        tools = kwargs.get("tools", [])
        input_data = kwargs.get("input", "")
        
        # Check if input contains tool results
        has_tool_results = False
        if isinstance(input_data, list):
            has_tool_results = any(
                item.get("type") == "function_call_output"
                for item in input_data
            )
        
        self.response_id = "dummy-response-id"
        self.created_at = 1234567890
        self.model = kwargs.get("model", "gpt-4")
        self.events = []
        
        if tools and not has_tool_results:
            # First call - emit tool call events
            self.response_text = ""
            
            # Event 1: response.created
            self.events.append(
                self.Event("response.created", self.response_id, created_at=self.created_at)
            )
            
            # Event 2: response.output_item.added (tool call)
            tool_item = {
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_1",
                "name": "test_tool",
                "arguments": "",
            }
            event = self.Event("response.output_item.added", self.response_id)
            event.item = MagicMock()
            event.item.to_dict = lambda: tool_item
            event.output_index = 0
            self.events.append(event)
            
            # Event 3: response.function_call_arguments.delta
            event = self.Event("response.function_call_arguments.delta", self.response_id)
            event.item_id = "item_1"
            event.delta = '{"arg1": "test_value"}'
            self.events.append(event)
            
            # Event 4: response.function_call_arguments.done
            event = self.Event("response.function_call_arguments.done", self.response_id)
            event.item_id = "item_1"
            event.arguments = '{"arg1": "test_value"}'
            self.events.append(event)
            
            # Event 5: response.output_item.done (tool call complete)
            event = self.Event("response.output_item.done", self.response_id)
            event.item = MagicMock()
            tool_item_done = {
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_1",
                "name": "test_tool",
                "arguments": '{"arg1": "test_value"}',
            }
            event.item.to_dict = lambda: tool_item_done
            self.events.append(event)
            
            # Event 6: response.completed
            usage = self.Event.Usage(input_tokens=10, output_tokens=5, total_tokens=15)
            output = []
            event = self.Event(
                "response.completed",
                self.response_id,
                created_at=self.created_at,
                output=output,
                usage=usage,
            )
            self.events.append(event)
        else:
            # After tool results - normal streaming
            super().__init__(**kwargs)
            # Override response_text
            self.response_text = "Response after tool call"
            # Recreate events with new text
            self._recreate_events()
        
        self.current_event = 0
        self.include_usage = kwargs.get("stream_options", {}).get("include_usage", True)
    
    def _recreate_events(self):
        """Recreate events with new response text."""
        self.events = []
        
        # Event 1: response.created
        self.events.append(
            self.Event("response.created", self.response_id, created_at=self.created_at)
        )
        
        # Event 2: response.content.started
        self.events.append(
            self.Event("response.content.started", self.response_id, content_index=0)
        )
        
        # Event 3: response.content.delta
        self.events.append(
            self.Event(
                "response.content.delta",
                self.response_id,
                content_index=0,
                text=self.response_text,
            )
        )
        
        # Event 4: response.content.completed
        self.events.append(
            self.Event(
                "response.content.completed",
                self.response_id,
                content_index=0,
                text=self.response_text,
            )
        )
        
        # Event 5: response.completed with usage
        usage = self.Event.Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": self.response_text}],
            }
        ]
        self.events.append(
            self.Event(
                "response.completed",
                self.response_id,
                created_at=self.created_at,
                output=output,
                usage=usage,
            )
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_event < len(self.events):
            event = self.events[self.current_event]
            self.current_event += 1
            return event
        else:
            raise StopIteration


class DummyAgenticModel(AgenticModelClass):
    """Dummy Agentic model implementation for testing."""

    client = MockOpenAIClientWithTools()
    model = "dummy-model"
    
    # Override pool for testing - set to None to allow each test to control pool behavior
    # This prevents test pollution where one test's pool state affects another test
    _pool = None
