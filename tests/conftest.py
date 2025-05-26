"""Conftest for OpenAI tests."""

import sys
from unittest import mock

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
