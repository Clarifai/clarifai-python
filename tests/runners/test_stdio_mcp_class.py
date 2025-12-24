"""Comprehensive tests for stdio_mcp_class.py"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from clarifai.runners.models.stdio_mcp_class import StdioMCPClient, StdioMCPModelClass


@pytest.fixture
def temp_config_file():
    """Create a temporary config.yaml file for testing."""
    config_data = {
        "mcp_server": {
            "command": "uvx",
            "args": ["mcp-server-calculator"],
            "env": {},
        },
        "secrets": [],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_config_file_with_secrets():
    """Create a temporary config.yaml file with secrets."""
    config_data = {
        "mcp_server": {
            "command": "uvx",
            "args": ["mcp-server-calculator"],
            "env": {},
        },
        "secrets": [
            {"id": "secret1", "value": "secret_value", "env_var": "SECRET_ENV_VAR"},
            {"id": "secret2", "value": None, "env_var": "SECRET_ENV_VAR_2"},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_mcp_imports():
    """Mock MCP-related imports."""

    # Create a mock TextContent class that works with isinstance
    class MockTextContent:
        pass

    with (
        patch("clarifai.runners.models.stdio_mcp_class.FastMCP") as mock_fastmcp,
        patch("clarifai.runners.models.stdio_mcp_class.ClientSession") as mock_session,
        patch("clarifai.runners.models.stdio_mcp_class.StdioServerParameters") as mock_params,
        patch("clarifai.runners.models.stdio_mcp_class.stdio_client") as mock_stdio_client,
        patch("clarifai.runners.models.stdio_mcp_class.Tool") as mock_tool,
        patch(
            "clarifai.runners.models.stdio_mcp_class.TextContent", MockTextContent
        ) as mock_text_content,
    ):
        yield {
            "FastMCP": mock_fastmcp,
            "ClientSession": mock_session,
            "StdioServerParameters": mock_params,
            "stdio_client": mock_stdio_client,
            "Tool": mock_tool,
            "TextContent": MockTextContent,
        }


class TestStdioMCPClient:
    """Test cases for StdioMCPClient class."""

    def test_init_with_args(self):
        """Test StdioMCPClient initialization with args."""
        client = StdioMCPClient(command="uvx", args=["mcp-server-calculator"], env={})
        assert client.command == "uvx"
        assert client.args == ["mcp-server-calculator"]
        assert client.env == {}
        assert client._started is False
        assert client._session is None

    def test_init_without_args_raises_error(self):
        """Test that initialization without args raises ValueError."""
        with pytest.raises(ValueError, match="args must be provided"):
            StdioMCPClient(command="uvx", args=None)

    def test_init_with_empty_args_raises_error(self):
        """Test that initialization with empty args raises ValueError."""
        with pytest.raises(ValueError, match="args must be provided"):
            StdioMCPClient(command="uvx", args=[])

    @pytest.mark.asyncio
    async def test_ensure_started_creates_session(self, mock_mcp_imports):
        """Test that _ensure_started creates a session."""
        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        await client._ensure_started()

        assert client._started is True
        assert client._session is not None
        mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_started_idempotent(self, mock_mcp_imports):
        """Test that _ensure_started is idempotent."""
        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        await client._ensure_started()
        await client._ensure_started()  # Call again

        # Should only initialize once
        assert mock_session.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_close_cleans_up_resources(self, mock_mcp_imports):
        """Test that close properly cleans up resources."""
        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_stdio_ctx.__aexit__ = AsyncMock()

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock()

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        await client._ensure_started()
        await client.close()

        assert client._started is False
        assert client._session is None
        mock_session_ctx.__aexit__.assert_called_once()
        mock_stdio_ctx.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_when_not_started(self):
        """Test that close does nothing when not started."""
        client = StdioMCPClient(command="uvx", args=["test"], env={})
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_mcp_imports):
        """Test listing tools from the stdio MCP server."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"

        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_mcp_imports):
        """Test calling a tool successfully."""
        # Create a mock that is an instance of TextContent
        TextContent = mock_mcp_imports["TextContent"]
        mock_text_content = MagicMock(spec=TextContent)
        mock_text_content.text = "result text"

        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [mock_text_content]
        mock_result.isError = False
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        result = await client.call_tool("test_tool", {"arg1": "value1"})

        assert result == "result text"
        mock_session.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})

    @pytest.mark.asyncio
    async def test_call_tool_with_error(self, mock_mcp_imports):
        """Test calling a tool that returns an error."""
        # Create a mock that is an instance of TextContent
        TextContent = mock_mcp_imports["TextContent"]
        mock_text_content = MagicMock(spec=TextContent)
        mock_text_content.text = "error message"

        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [mock_text_content]
        mock_result.isError = True
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        with pytest.raises(RuntimeError, match="Tool error: error message"):
            await client.call_tool("test_tool", {"arg1": "value1"})

    @pytest.mark.asyncio
    async def test_call_tool_with_multiple_content_parts(self, mock_mcp_imports):
        """Test calling a tool with multiple content parts."""
        # Create mocks that are instances of TextContent
        TextContent = mock_mcp_imports["TextContent"]
        mock_text_content1 = MagicMock(spec=TextContent)
        mock_text_content1.text = "part1"
        mock_text_content2 = MagicMock(spec=TextContent)
        mock_text_content2.text = "part2"

        mock_stdio_ctx = AsyncMock()
        mock_transport = (AsyncMock(), AsyncMock())
        mock_stdio_ctx.__aenter__ = AsyncMock(return_value=mock_transport)

        mock_session_ctx = AsyncMock()
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [mock_text_content1, mock_text_content2]
        mock_result.isError = False
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        mock_mcp_imports["stdio_client"].return_value = mock_stdio_ctx
        mock_mcp_imports["ClientSession"].return_value = mock_session_ctx

        client = StdioMCPClient(command="uvx", args=["test"], env={})
        result = await client.call_tool("test_tool", {})

        assert result == "part1\npart2"


class TestStdioMCPModelClass:
    """Test cases for StdioMCPModelClass."""

    def test_init(self):
        """Test StdioMCPModelClass initialization."""
        model = StdioMCPModelClass()
        assert model._stdio_client is None
        assert model._server is None
        assert model._tools_registered is False

    def test_json_type_to_python(self):
        """Test _json_type_to_python method."""
        model = StdioMCPModelClass()
        assert model._json_type_to_python("string") is str
        assert model._json_type_to_python("integer") is int
        assert model._json_type_to_python("number") is float
        assert model._json_type_to_python("boolean") is bool
        assert model._json_type_to_python("array") is list
        assert model._json_type_to_python("object") is dict
        assert model._json_type_to_python("unknown") is str  # Default

    @pytest.mark.asyncio
    async def test_create_tool_function(self):
        """Test _create_tool_function creates a valid function."""
        mock_stdio_client = AsyncMock()
        mock_stdio_client.call_tool = AsyncMock(return_value="result")

        model = StdioMCPModelClass()
        properties = {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        }
        required = ["a"]

        func = model._create_tool_function("test-tool", properties, required, mock_stdio_client)

        assert callable(func)
        assert func.__name__ == "test_tool"
        assert "a" in func.__annotations__
        assert "b" in func.__annotations__

        # Test calling the function
        result = await func("value_a", 42)
        assert result == "result"
        mock_stdio_client.call_tool.assert_called_once_with("test-tool", {"a": "value_a", "b": 42})

    @pytest.mark.asyncio
    async def test_create_tool_function_with_optional_params(self):
        """Test _create_tool_function with optional parameters."""
        mock_stdio_client = AsyncMock()
        mock_stdio_client.call_tool = AsyncMock(return_value="result")

        model = StdioMCPModelClass()
        properties = {
            "a": {"type": "string"},
            "b": {"type": "integer"},
        }
        required = []  # All optional

        func = model._create_tool_function("test-tool", properties, required, mock_stdio_client)

        # Call with only one parameter
        result = await func("value_a")
        assert result == "result"
        mock_stdio_client.call_tool.assert_called_once_with("test-tool", {"a": "value_a"})

    @pytest.mark.asyncio
    async def test_create_tool_function_with_error(self):
        """Test _create_tool_function error handling."""
        mock_stdio_client = AsyncMock()
        mock_stdio_client.call_tool = AsyncMock(side_effect=RuntimeError("Test error"))

        model = StdioMCPModelClass()
        properties = {"a": {"type": "string"}}
        required = ["a"]

        func = model._create_tool_function("test-tool", properties, required, mock_stdio_client)

        result = await func("value_a")
        assert "Error executing test-tool" in result
        assert "RuntimeError" in result
        assert "Test error" in result

    def test_find_config_file(self, temp_config_file):
        """Test _find_config_file finds the config file."""

        # Create a model class in a temporary directory
        class TestModel(StdioMCPModelClass):
            pass

        # The code expects: if model is at /path/to/1/model.py, config is at /path/to/config.yaml
        # So we need to create a model file path in a subdirectory of the config's directory
        config_dir = os.path.dirname(temp_config_file)
        model_dir = os.path.join(config_dir, "1")
        model_file = os.path.join(model_dir, "model.py")

        # Mock inspect.getfile to return our model file path
        with (
            patch("clarifai.runners.models.stdio_mcp_class.inspect.getfile") as mock_getfile,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.join") as mock_join,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.exists") as mock_exists,
        ):
            mock_getfile.return_value = model_file

            # Make join return the temp_config_file when joining parent dir with "config.yaml"
            def join_side_effect(*args):
                if len(args) == 2 and args[1] == "config.yaml":
                    return temp_config_file
                return os.path.join(*args)

            mock_join.side_effect = join_side_effect

            # Make exists return True for the config file path
            def exists_side_effect(path):
                return path == temp_config_file

            mock_exists.side_effect = exists_side_effect

            model = TestModel()
            config_path = model._find_config_file()

            assert config_path is not None
            assert config_path == temp_config_file

    def test_find_config_file_not_found(self):
        """Test _find_config_file when config file doesn't exist."""

        class TestModel(StdioMCPModelClass):
            pass

        with (
            patch("clarifai.runners.models.stdio_mcp_class.inspect.getfile") as mock_getfile,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.exists") as mock_exists,
        ):
            mock_getfile.return_value = "/some/path/model.py"
            mock_exists.return_value = False

            model = TestModel()
            config_path = model._find_config_file()

            assert config_path is None

    def test_load_secrets(self, temp_config_file_with_secrets):
        """Test _load_secrets loads secrets from config."""

        class TestModel(StdioMCPModelClass):
            pass

        with (
            patch("clarifai.runners.models.stdio_mcp_class.inspect.getfile") as mock_getfile,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.dirname") as mock_dirname,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.exists") as mock_exists,
            patch("clarifai.runners.models.stdio_mcp_class.os.path.abspath") as mock_abspath,
        ):
            mock_getfile.return_value = temp_config_file_with_secrets
            mock_abspath.return_value = temp_config_file_with_secrets
            mock_dirname.side_effect = (
                lambda x: os.path.dirname(x)
                if x != temp_config_file_with_secrets
                else os.path.dirname(temp_config_file_with_secrets)
            )
            mock_exists.return_value = True

            model = TestModel()
            # Override _find_config_file to return our temp file
            model._find_config_file = lambda: temp_config_file_with_secrets

            secrets = model._load_secrets()

            assert len(secrets) == 2
            assert secrets[0]["id"] == "secret1"
            assert secrets[0]["value"] == "secret_value"
            assert secrets[0]["env_var"] == "SECRET_ENV_VAR"

    def test_load_secrets_file_not_found(self):
        """Test _load_secrets when config file doesn't exist."""
        model = StdioMCPModelClass()
        model._find_config_file = lambda: None

        with pytest.raises(FileNotFoundError, match="config.yaml not found"):
            model._load_secrets()

    def test_load_mcp_config(self, temp_config_file):
        """Test _load_mcp_config loads MCP config."""
        model = StdioMCPModelClass()
        model._find_config_file = lambda: temp_config_file

        config = model._load_mcp_config()

        assert config["command"] == "uvx"
        assert config["args"] == ["mcp-server-calculator"]
        assert config["env"] == {}

    def test_load_mcp_config_missing_section(self):
        """Test _load_mcp_config when mcp_server section is missing."""
        config_data = {"secrets": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            model = StdioMCPModelClass()
            model._find_config_file = lambda: temp_file

            with pytest.raises(ValueError, match="Missing 'mcp_server' section"):
                model._load_mcp_config()
        finally:
            os.unlink(temp_file)

    def test_load_mcp_config_missing_command(self):
        """Test _load_mcp_config when command is missing."""
        config_data = {"mcp_server": {"args": ["test"]}, "secrets": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            model = StdioMCPModelClass()
            model._find_config_file = lambda: temp_file

            with pytest.raises(ValueError, match="'command' missing"):
                model._load_mcp_config()
        finally:
            os.unlink(temp_file)

    def test_load_mcp_config_missing_args(self):
        """Test _load_mcp_config when args is missing."""
        config_data = {"mcp_server": {"command": "uvx"}, "secrets": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            model = StdioMCPModelClass()
            model._find_config_file = lambda: temp_file

            with pytest.raises(ValueError, match="'args' missing"):
                model._load_mcp_config()
        finally:
            os.unlink(temp_file)

    def test_load_mcp_config_with_string_args(self):
        """Test _load_mcp_config when args is a string (should be converted to list)."""
        config_data = {"mcp_server": {"command": "uvx", "args": "single-arg"}, "secrets": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            model = StdioMCPModelClass()
            model._find_config_file = lambda: temp_file

            config = model._load_mcp_config()
            assert config["args"] == ["single-arg"]
        finally:
            os.unlink(temp_file)

    def test_get_stdio_client(self, temp_config_file):
        """Test _get_stdio_client creates and returns a client."""
        model = StdioMCPModelClass()
        model._find_config_file = lambda: temp_config_file
        model._load_secrets = lambda: []

        client = model._get_stdio_client()

        assert client is not None
        assert isinstance(client, StdioMCPClient)
        assert client.command == "uvx"
        assert client.args == ["mcp-server-calculator"]

    def test_get_stdio_client_with_secrets(self, temp_config_file_with_secrets):
        """Test _get_stdio_client includes secrets in environment."""
        model = StdioMCPModelClass()
        model._find_config_file = lambda: temp_config_file_with_secrets

        with patch.dict(os.environ, {"SECRET_ENV_VAR_2": "env_value"}):
            client = model._get_stdio_client()

            assert client is not None
            assert client.env.get("SECRET_ENV_VAR") == "secret_value"
            assert client.env.get("SECRET_ENV_VAR_2") == "env_value"

    def test_get_stdio_client_cached(self, temp_config_file):
        """Test _get_stdio_client returns cached client."""
        model = StdioMCPModelClass()
        model._find_config_file = lambda: temp_config_file
        model._load_secrets = lambda: []

        client1 = model._get_stdio_client()
        client2 = model._get_stdio_client()

        assert client1 is client2  # Should be the same instance

    def test_get_server_creates_fastmcp_server(self, mock_mcp_imports, temp_config_file):
        """Test get_server creates a FastMCP server."""
        mock_fastmcp_instance = MagicMock()
        mock_mcp_imports["FastMCP"].return_value = mock_fastmcp_instance

        model = StdioMCPModelClass()
        model._find_config_file = lambda: temp_config_file

        server = model.get_server()

        assert server is not None
        assert server == mock_fastmcp_instance
        mock_mcp_imports["FastMCP"].assert_called_once()

    def test_get_server_import_error(self):
        """Test get_server raises ImportError when fastmcp is not available."""
        with patch("clarifai.runners.models.stdio_mcp_class.FastMCP", None):
            model = StdioMCPModelClass()
            with pytest.raises(ImportError, match="fastmcp package is required"):
                model.get_server()

    @pytest.mark.asyncio
    async def test_background_shutdown(self):
        """Test _background_shutdown closes stdio client."""
        mock_stdio_client = AsyncMock()
        mock_stdio_client.close = AsyncMock()

        model = StdioMCPModelClass()
        model._stdio_client = mock_stdio_client

        # Mock parent's _background_shutdown to avoid calling it
        with patch.object(
            StdioMCPModelClass.__bases__[0], "_background_shutdown", new_callable=AsyncMock
        ) as mock_parent_shutdown:
            await model._background_shutdown()

        mock_stdio_client.close.assert_called_once()
        assert model._stdio_client is None
        # Verify parent shutdown is also called
        mock_parent_shutdown.assert_called_once()

    def test_shutdown_resets_tools_registered(self):
        """Test shutdown resets tools_registered flag."""
        model = StdioMCPModelClass()
        model._tools_registered = True

        # Mock parent shutdown to avoid actual shutdown logic
        with patch.object(StdioMCPModelClass.__bases__[0], "shutdown") as mock_parent_shutdown:
            model.shutdown()

        assert model._tools_registered is False
        # Verify parent shutdown is also called
        mock_parent_shutdown.assert_called_once()
