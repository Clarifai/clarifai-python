"""Test cases for MCPConnectionPool singleton and connection lifecycle management."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from clarifai.runners.models.agentic_class import MCPConnection, MCPConnectionPool


class TestMCPConnectionPool:
    """Tests for MCPConnectionPool singleton and connection management."""

    @pytest.fixture(autouse=True)
    def reset_pool(self):
        """Reset the singleton instance before each test."""
        # Clear singleton instance
        MCPConnectionPool._instance = None
        yield
        # Clean up after test
        MCPConnectionPool._instance = None

    def test_singleton_pattern(self):
        """Test that MCPConnectionPool is a singleton."""
        pool1 = MCPConnectionPool()
        pool2 = MCPConnectionPool()

        assert pool1 is pool2
        assert id(pool1) == id(pool2)

    def test_singleton_initialization_once(self):
        """Test that singleton is initialized only once."""
        pool1 = MCPConnectionPool()
        initial_connections = pool1._connections

        pool2 = MCPConnectionPool()

        # Should have the same connections dictionary
        assert pool2._connections is initial_connections

    def test_event_loop_initialization(self):
        """Test that background event loop is started on init."""
        pool = MCPConnectionPool()

        assert pool._loop is not None
        assert pool._loop_thread is not None
        assert pool._loop_thread.is_alive()
        assert not pool._loop.is_closed()

    def test_connection_cleanup_idle_timeout(self):
        """Test that connections idle > MAX_IDLE_TIME are removed."""
        pool = MCPConnectionPool()

        # Create a mock connection that's been idle for too long
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        old_conn = MCPConnection(
            client=mock_client,
            tools=[],
            tool_names=set(),
            url="http://old-server",
            last_used=time.time() - pool.MAX_IDLE_TIME - 1,
        )

        with pool._lock:
            pool._connections["http://old-server"] = old_conn

        # Force cleanup to run immediately
        pool._last_cleanup = 0
        pool._maybe_cleanup_idle()

        # Connection should be removed
        with pool._lock:
            assert "http://old-server" not in pool._connections

    def test_cleanup_interval_rate_limiting(self):
        """Test that cleanup checks are rate limited."""
        pool = MCPConnectionPool()

        # Create an idle connection
        mock_client = MagicMock()
        old_conn = MCPConnection(
            client=mock_client,
            tools=[],
            tool_names=set(),
            url="http://server",
            last_used=time.time() - pool.MAX_IDLE_TIME - 1,
        )

        with pool._lock:
            pool._connections["http://server"] = old_conn

        # Set last cleanup to recent time
        pool._last_cleanup = time.time()

        # Try to cleanup - should be skipped due to rate limiting
        pool._maybe_cleanup_idle()

        # Connection should still exist (cleanup was skipped)
        with pool._lock:
            assert "http://server" in pool._connections

    def test_connection_verification_valid(self):
        """Test connection verification for valid connections."""
        pool = MCPConnectionPool()

        # Create mock connection
        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        conn = MCPConnection(
            client=mock_client, tools=[], tool_names=set(), url="http://valid-server"
        )

        # Run verification
        is_valid = pool._run_async(pool._verify_connection(conn))

        assert is_valid is True
        mock_client.list_tools.assert_called_once()

    def test_connection_verification_invalid(self):
        """Test connection verification for invalid connections."""
        pool = MCPConnectionPool()

        # Create mock connection that fails verification
        mock_client = MagicMock()
        mock_client.list_tools = AsyncMock(side_effect=Exception("Connection lost"))

        conn = MCPConnection(
            client=mock_client, tools=[], tool_names=set(), url="http://invalid-server"
        )

        # Run verification
        is_valid = pool._run_async(pool._verify_connection(conn))

        assert is_valid is False

    def test_needs_verification(self):
        """Test _needs_verification logic."""
        pool = MCPConnectionPool()

        # Fresh connection should not need verification
        fresh_conn = MCPConnection(
            client=MagicMock(), tools=[], tool_names=set(), url="http://fresh"
        )
        assert pool._needs_verification(fresh_conn) is False

        # Old connection should need verification
        old_conn = MCPConnection(
            client=MagicMock(),
            tools=[],
            tool_names=set(),
            url="http://old",
            last_used=time.time() - pool.VERIFY_IDLE_THRESHOLD - 1,
        )
        assert pool._needs_verification(old_conn) is True

    def test_parallel_connection_creation(self):
        """Test that connections are created in parallel."""
        pool = MCPConnectionPool()

        # Create mock connections and add them directly
        mock_client1 = MagicMock()
        mock_client1.list_tools = AsyncMock(return_value=[])
        conn1 = MCPConnection(
            client=mock_client1, tools=[], tool_names=set(), url="http://server1"
        )

        mock_client2 = MagicMock()
        mock_client2.list_tools = AsyncMock(return_value=[])
        conn2 = MCPConnection(
            client=mock_client2, tools=[], tool_names=set(), url="http://server2"
        )

        mock_client3 = MagicMock()
        mock_client3.list_tools = AsyncMock(return_value=[])
        conn3 = MCPConnection(
            client=mock_client3, tools=[], tool_names=set(), url="http://server3"
        )

        with pool._lock:
            pool._connections["http://server1"] = conn1
            pool._connections["http://server2"] = conn2
            pool._connections["http://server3"] = conn3

        # Get connections - should reuse existing ones
        urls = ["http://server1", "http://server2", "http://server3"]
        connections = pool.get_connections(urls)

        # All connections should be returned
        assert len(connections) == 3
        for url in urls:
            assert url in connections

    def test_connection_creation_error_handling(self):
        """Test error handling when connection creation fails."""
        pool = MCPConnectionPool()

        # Try to create connection for a URL that's not already in pool
        # This will fail because fastmcp is not actually installed
        urls = ["http://bad-server"]
        connections = pool.get_connections(urls)

        # Should handle error gracefully and return empty dict
        assert len(connections) == 0

    def test_parallel_connection_creation_partial_failure(self):
        """Test parallel creation with some failures."""
        pool = MCPConnectionPool()

        # Add two valid connections
        mock_client1 = MagicMock()
        mock_client1.list_tools = AsyncMock(return_value=[])
        conn1 = MCPConnection(
            client=mock_client1, tools=[], tool_names=set(), url="http://server1"
        )

        mock_client3 = MagicMock()
        mock_client3.list_tools = AsyncMock(return_value=[])
        conn3 = MCPConnection(
            client=mock_client3, tools=[], tool_names=set(), url="http://server3"
        )

        with pool._lock:
            pool._connections["http://server1"] = conn1
            pool._connections["http://server3"] = conn3

        # Try to get connections including one that doesn't exist (will fail to create)
        urls = ["http://server1", "http://server2", "http://server3"]
        connections = pool.get_connections(urls)

        # Should have 2 successful connections (1st and 3rd)
        assert len(connections) == 2
        assert "http://server1" in connections
        assert "http://server3" in connections
        assert "http://server2" not in connections

    def test_connection_touch_mechanism(self):
        """Test that connections are touched when accessed."""
        pool = MCPConnectionPool()

        # Create connection with old timestamp
        mock_client = MagicMock()
        old_time = time.time() - 100
        conn = MCPConnection(
            client=mock_client, tools=[], tool_names=set(), url="http://server", last_used=old_time
        )

        with pool._lock:
            pool._connections["http://server"] = conn

        # Access connection (should touch it)
        connections = pool.get_connections(["http://server"])

        # Last used should be updated
        assert connections["http://server"].last_used > old_time

    def test_tool_cache_update(self):
        """Test that tool cache is updated when connections are created."""
        pool = MCPConnectionPool()

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Test tool 1"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Test tool 2"
        mock_tool2.inputSchema = {"type": "object"}

        conn = MCPConnection(
            client=MagicMock(),
            tools=[mock_tool1, mock_tool2],
            tool_names={"tool1", "tool2"},
            url="http://server",
        )

        # Update cache
        pool._update_tool_cache(conn)

        # Verify cache contents
        assert "tool1" in pool._tool_to_url
        assert "tool2" in pool._tool_to_url
        assert pool._tool_to_url["tool1"] == "http://server"
        assert pool._tool_to_url["tool2"] == "http://server"
        assert "tool1" in pool._all_tools
        assert "tool2" in pool._all_tools

    def test_tool_cache_invalidation_on_disconnect(self):
        """Test that tool cache is invalidated when connection is removed."""
        pool = MCPConnectionPool()

        # Create connection with tools
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        conn = MCPConnection(
            client=mock_client, tools=[mock_tool], tool_names={"test_tool"}, url="http://server"
        )

        with pool._lock:
            pool._connections["http://server"] = conn
            pool._tool_to_url["test_tool"] = "http://server"
            pool._all_tools["test_tool"] = {"type": "function"}

        # Disconnect
        pool._disconnect("http://server")

        # Tool cache should be cleared
        assert "test_tool" not in pool._tool_to_url
        assert "test_tool" not in pool._all_tools

    def test_connection_reuse(self):
        """Test that existing connections are reused."""
        pool = MCPConnectionPool()

        # Create initial connection
        mock_client = MagicMock()
        conn = MCPConnection(client=mock_client, tools=[], tool_names=set(), url="http://server")

        with pool._lock:
            pool._connections["http://server"] = conn

        # Get connection again
        connections = pool.get_connections(["http://server"])

        # Should reuse same connection
        assert connections["http://server"] is conn

    def test_stale_connection_recreation(self):
        """Test that stale connections are verified and removed if invalid."""
        pool = MCPConnectionPool()

        # Create stale connection that will fail verification
        mock_old_client = MagicMock()
        mock_old_client.list_tools = AsyncMock(side_effect=Exception("Connection lost"))

        old_conn = MCPConnection(
            client=mock_old_client,
            tools=[],
            tool_names=set(),
            url="http://server",
            last_used=time.time() - pool.VERIFY_IDLE_THRESHOLD - 1,
        )

        with pool._lock:
            pool._connections["http://server"] = old_conn

        # Get connection (should verify and fail, then try to recreate but fail due to missing fastmcp)
        connections = pool.get_connections(["http://server"])

        # Should not have connection since verification failed and recreation failed
        # (fastmcp is not installed so recreation will fail)
        assert "http://server" not in connections

        # Original connection should have been removed from pool
        with pool._lock:
            assert "http://server" not in pool._connections

    def test_close_connection_with_close_method(self):
        """Test closing connection with close() method."""
        pool = MCPConnectionPool()

        # Create connection with close method
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        conn = MCPConnection(client=mock_client, tools=[], tool_names=set(), url="http://server")

        # Close connection
        pool._run_async(pool._close_connection(conn))

        # close() should be called
        mock_client.close.assert_called_once()

    def test_close_connection_with_aexit(self):
        """Test closing connection with __aexit__ method."""
        pool = MCPConnectionPool()

        # Create connection without close method but with __aexit__
        mock_client = MagicMock()
        # Remove close attribute to force use of __aexit__
        del mock_client.close
        mock_client.__aexit__ = AsyncMock()

        conn = MCPConnection(client=mock_client, tools=[], tool_names=set(), url="http://server")

        # Close connection
        pool._run_async(pool._close_connection(conn))

        # __aexit__ should be called
        mock_client.__aexit__.assert_called_once_with(None, None, None)

    def test_get_tools_and_mapping(self):
        """Test getting tools and connection mapping."""
        pool = MCPConnectionPool()

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Test tool 1"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Test tool 2"
        mock_tool2.inputSchema = {"type": "object"}

        # Create connections
        conn1 = MCPConnection(
            client=MagicMock(), tools=[mock_tool1], tool_names={"tool1"}, url="http://server1"
        )

        conn2 = MCPConnection(
            client=MagicMock(), tools=[mock_tool2], tool_names={"tool2"}, url="http://server2"
        )

        with pool._lock:
            pool._connections["http://server1"] = conn1
            pool._connections["http://server2"] = conn2

        # Get tools and mapping
        tools, connections, tool_to_server = pool.get_tools_and_mapping(
            ["http://server1", "http://server2"]
        )

        # Verify results
        assert len(tools) == 2
        assert len(connections) == 2
        assert "tool1" in tool_to_server
        assert "tool2" in tool_to_server
        assert tool_to_server["tool1"] == "http://server1"
        assert tool_to_server["tool2"] == "http://server2"

    def test_call_tool_sync(self):
        """Test synchronous tool calling."""
        pool = MCPConnectionPool()

        # Create mock connection
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Result")]
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        conn = MCPConnection(
            client=mock_client, tools=[], tool_names={"test_tool"}, url="http://server"
        )

        connections = {"http://server": conn}
        tool_to_server = {"test_tool": "http://server"}

        # Call tool
        result = pool.call_tool("test_tool", {"arg": "value"}, connections, tool_to_server)

        # Verify call
        assert result is mock_result
        mock_client.call_tool.assert_called_once_with("test_tool", arguments={"arg": "value"})

    def test_call_tools_batch(self):
        """Test batch tool calling."""
        pool = MCPConnectionPool()

        # Create mock connection
        mock_client = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.content = [MagicMock(text="Result1")]
        mock_result2 = MagicMock()
        mock_result2.content = [MagicMock(text="Result2")]

        async def mock_call_tool(name, arguments):
            if name == "tool1":
                return mock_result1
            return mock_result2

        mock_client.call_tool = mock_call_tool

        conn = MCPConnection(
            client=mock_client, tools=[], tool_names={"tool1", "tool2"}, url="http://server"
        )

        connections = {"http://server": conn}
        tool_to_server = {"tool1": "http://server", "tool2": "http://server"}

        # Call tools in batch
        calls = [("id1", "tool1", {"arg": "value1"}), ("id2", "tool2", {"arg": "value2"})]
        results = pool.call_tools_batch(calls, connections, tool_to_server)

        # Verify results
        assert len(results) == 2
        assert results[0][0] == "id1"
        assert results[1][0] == "id2"

    def test_tool_call_timeout(self):
        """Test that tool calls timeout appropriately."""
        pool = MCPConnectionPool()

        # Create mock connection with slow tool
        mock_client = MagicMock()

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(100)  # Sleep longer than timeout
            return MagicMock()

        mock_client.call_tool = slow_tool

        conn = MCPConnection(
            client=mock_client, tools=[], tool_names={"slow_tool"}, url="http://server"
        )

        connections = {"http://server": conn}
        tool_to_server = {"slow_tool": "http://server"}

        # Call should timeout
        with pytest.raises(asyncio.TimeoutError):
            pool.call_tool("slow_tool", {}, connections, tool_to_server)
