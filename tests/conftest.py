"""Conftest for OpenAI tests."""

import asyncio
import sys
from unittest import mock

import pytest

# Create mock modules
mock_fastmcp = mock.MagicMock()
mock_fastmcp.Client = mock.MagicMock()
mock_fastmcp.FastMCP = mock.MagicMock()

mock_mcp = mock.MagicMock()
mock_mcp.types = mock.MagicMock()
mock_mcp.shared = mock.MagicMock()
mock_mcp.shared.exceptions = mock.MagicMock()
mock_mcp.shared.exceptions.McpError = Exception

# Mock the fastmcp and mcp modules
sys.modules['fastmcp'] = mock_fastmcp
sys.modules['mcp'] = mock_mcp
sys.modules['mcp.shared'] = mock_mcp.shared
sys.modules['mcp.shared.exceptions'] = mock_mcp.shared.exceptions


@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """
    Override pytest’s default loop to ensure asyncio.get_event_loop()
    always returns a running loop on the main thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
