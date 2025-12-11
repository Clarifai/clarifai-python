"""Test cases for MCPModelClass and OpenAIModelClass."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel
from clarifai.runners.models.mcp_class import MCPModelClass
from clarifai.runners.models.openai_class import OpenAIModelClass


class TestModelClasses:
    """Tests for model classes."""

    def test_mcp_model_initialization(self):
        """Test that MCPModelClass requires subclass implementation."""
        # Test that subclass must implement get_server()
        with pytest.raises(NotImplementedError):
            MCPModelClass().get_server()

    def test_openai_model_initialization(self):
        """Test that OpenAIModelClass can be initialized."""
        model = DummyOpenAIModel()
        assert isinstance(model, OpenAIModelClass)

        # Test that subclass must have `client` attribute
        with pytest.raises(NotImplementedError):
            OpenAIModelClass().client

        # Test that client has required attributes
        client = model.client
        assert hasattr(client, 'chat')
        assert hasattr(client, 'completions')

    def test_openai_transport_non_streaming(self):
        """Test OpenAI transport method with non-streaming request."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": False,
        }

        # Call the transport method
        response_str = model.openai_transport(json.dumps(request))
        response = json.loads(response_str)

        # Verify response structure
        assert "id" in response
        assert "created" in response
        assert "model" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        assert "Echo: Hello, world!" in response["choices"][0]["message"]["content"]
        assert "usage" in response

    def test_openai_transport_streaming(self):
        """Test OpenAI transport method with streaming request."""
        model = DummyOpenAIModel()
        model.load_model()

        # Create a simple chat request with streaming
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "stream": True,
        }

        # Call the transport method
        response = model.openai_stream_transport(json.dumps(request))
        response_chunks = [json.loads(resp) for resp in response]

        assert isinstance(response_chunks, list)
        assert len(response_chunks) > 0

        # Check first chunk for content
        first_chunk = response_chunks[0]
        assert "id" in first_chunk
        assert "created" in first_chunk
        assert "model" in first_chunk
        assert "choices" in first_chunk
        assert len(first_chunk["choices"]) > 0
        assert "delta" in first_chunk["choices"][0]
        assert "content" in first_chunk["choices"][0]["delta"]
        assert "Echo: Hello, world!" in first_chunk["choices"][0]["delta"]["content"]

    def test_custom_method(self):
        """Test custom method on the DummyOpenAIModel."""
        model = DummyOpenAIModel()
        result = model.test_method("test input")
        assert result == "Test: test input"


class TestMCPModelClass:
    """Tests for MCPModelClass."""

    def test_mcp_model_requires_get_server_implementation(self):
        """Test that MCPModelClass requires subclasses to implement get_server()."""
        model = MCPModelClass()
        with pytest.raises(NotImplementedError, match="Subclasses must implement get_server"):
            model.get_server()

    def test_mcp_model_initialization_attributes(self):
        """Test that MCPModelClass initializes with correct attributes."""
        model = MCPModelClass()
        
        # Check initial state
        assert model._fastmcp_server is None
        assert model._client is None
        assert model._client_session is None
        assert model._loop is None
        assert model._thread is None
        assert hasattr(model, '_initialized')
        assert model._init_error is None

    @pytest.mark.skipif(
        not hasattr(__import__('sys').modules.get('fastmcp'), '__version__'),
        reason="fastmcp not available"
    )
    def test_mcp_model_load_starts_background_loop(self):
        """Test that load_model starts the background loop."""
        # Create a simple test model with a mock server
        try:
            from fastmcp import FastMCP
            
            class TestMCPModel(MCPModelClass):
                def get_server(self):
                    return FastMCP("test-server", instructions="test")
            
            model = TestMCPModel()
            model.load_model()
            
            # Verify background thread was started
            assert model._thread is not None
            assert model._thread.is_alive()
            assert model._loop is not None
            
            # Clean up
            model.shutdown()
            
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_mcp_model_shutdown_stops_background_loop(self):
        """Test that shutdown() stops the background loop."""
        try:
            from fastmcp import FastMCP
            
            class TestMCPModel(MCPModelClass):
                def get_server(self):
                    return FastMCP("test-server", instructions="test")
            
            model = TestMCPModel()
            model.load_model()
            
            # Verify thread is running
            assert model._thread is not None
            assert model._thread.is_alive()
            
            # Shutdown and verify cleanup
            model.shutdown()
            
            # Give thread time to stop
            time.sleep(0.5)
            
            assert model._loop is None
            assert model._thread is None
            
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_mcp_model_load_timeout(self):
        """Test that load_model raises error on initialization timeout."""
        class SlowMCPModel(MCPModelClass):
            def get_server(self):
                # Simulate slow initialization
                import time
                time.sleep(100)
                from fastmcp import FastMCP
                return FastMCP("slow-server", instructions="test")
        
        model = SlowMCPModel()
        
        with patch('clarifai.runners.models.mcp_class.MCPModelClass._start_background_loop'):
            # Mock the initialization to never complete
            model._initialized = MagicMock()
            model._initialized.wait.return_value = False
            
            with pytest.raises(RuntimeError, match="Background MCP initialization timed out"):
                model.load_model()

    def test_mcp_model_initialization_error_propagation(self):
        """Test that initialization errors are propagated correctly."""
        class FailingMCPModel(MCPModelClass):
            def get_server(self):
                raise ValueError("Test error")
        
        model = FailingMCPModel()
        
        with pytest.raises(ValueError, match="Test error"):
            model.load_model()

    @pytest.mark.skipif(
        not hasattr(__import__('sys').modules.get('fastmcp'), '__version__'),
        reason="fastmcp not available"
    )
    def test_mcp_transport_method_exists(self):
        """Test that mcp_transport method exists and is decorated."""
        from clarifai.runners.models.model_class import ModelClass
        
        # Check that mcp_transport is a method
        assert hasattr(MCPModelClass, 'mcp_transport')
        
        # Verify it's decorated with @ModelClass.method
        model = MCPModelClass()
        assert callable(getattr(model, 'mcp_transport', None))

    def test_mcp_model_get_result_type_mapping(self):
        """Test that _get_result_type correctly maps request types to result types."""
        try:
            from mcp import types
            
            model = MCPModelClass()
            
            # Test various request types
            test_cases = [
                (types.PingRequest(), types.EmptyResult),
                (types.ListToolsRequest(), types.ListToolsResult),
                (types.ListResourcesRequest(), types.ListResourcesResult),
            ]
            
            for request, expected_result in test_cases:
                # Create a client message wrapper
                client_msg = Mock()
                client_msg.root = request
                result_type = model._get_result_type(client_msg)
                assert result_type == expected_result
                
        except ImportError:
            pytest.skip("mcp package not installed")


class TestStdioMCPClient:
    """Tests for StdioMCPClient."""

    def test_stdio_client_initialization(self):
        """Test StdioMCPClient initialization."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPClient
            
            client = StdioMCPClient(
                command="npx",
                args=["-y", "test-server"],
                env={"TEST_VAR": "test_value"}
            )
            
            assert client.command == "npx"
            assert client.args == ["-y", "test-server"]
            assert client.env == {"TEST_VAR": "test_value"}
            assert client._started is False
            
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_stdio_client_requires_args(self):
        """Test that StdioMCPClient requires args parameter."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPClient
            
            with pytest.raises(ValueError, match="args must be provided"):
                StdioMCPClient(command="npx", args=None)
                
            with pytest.raises(ValueError, match="args must be provided"):
                StdioMCPClient(command="npx", args=[])
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_stdio_client_env_defaults_to_empty_dict(self):
        """Test that env parameter defaults to empty dict."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPClient
            
            client = StdioMCPClient(command="npx", args=["test"])
            assert client.env == {}
            
        except ImportError:
            pytest.skip("Required MCP packages not installed")


class TestStdioMCPModelClass:
    """Tests for StdioMCPModelClass."""

    def test_stdio_model_initialization(self):
        """Test StdioMCPModelClass initialization."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            assert model._stdio_client is None
            assert model._server is None
            assert model._tools_registered is False
            
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_json_type_to_python_conversion(self):
        """Test JSON type to Python type conversion."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            assert model._json_type_to_python("string") == str
            assert model._json_type_to_python("integer") == int
            assert model._json_type_to_python("number") == float
            assert model._json_type_to_python("boolean") == bool
            assert model._json_type_to_python("array") == list
            assert model._json_type_to_python("object") == dict
            assert model._json_type_to_python("unknown") == str  # default
            
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_find_config_file_not_found(self):
        """Test _find_config_file when config doesn't exist."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            class TestModel(StdioMCPModelClass):
                pass
            
            # Create an instance where the config won't be found
            model = TestModel()
            
            # Mock the class file location to a temp directory
            with patch('inspect.getfile') as mock_getfile:
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_dir = Path(tmpdir) / "test_model"
                    model_dir.mkdir()
                    mock_getfile.return_value = str(model_dir / "model.py")
                    
                    result = model._find_config_file()
                    assert result is None
                    
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_find_config_file_found(self):
        """Test _find_config_file when config exists."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            class TestModel(StdioMCPModelClass):
                pass
            
            model = TestModel()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create config file
                config_path = Path(tmpdir) / "config.yaml"
                config_path.write_text("model:\n  id: test\n")
                
                # Mock the class file to be in a subdirectory
                model_dir = Path(tmpdir) / "1"
                model_dir.mkdir()
                
                with patch('inspect.getfile') as mock_getfile:
                    mock_getfile.return_value = str(model_dir / "model.py")
                    
                    result = model._find_config_file()
                    assert result == str(config_path)
                    
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_mcp_config_missing_file(self):
        """Test _load_mcp_config raises error when config file is missing."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with patch.object(model, '_find_config_file', return_value=None):
                with pytest.raises(FileNotFoundError, match="config.yaml not found"):
                    model._load_mcp_config()
                    
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_mcp_config_missing_mcp_section(self):
        """Test _load_mcp_config raises error when mcp_server section is missing."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("model:\n  id: test\n")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    with pytest.raises(ValueError, match="Missing 'mcp_server' section"):
                        model._load_mcp_config()
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_mcp_config_missing_command(self):
        """Test _load_mcp_config raises error when command is missing."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("mcp_server:\n  args: ['test']\n")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    with pytest.raises(ValueError, match="'command' missing in mcp_server"):
                        model._load_mcp_config()
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_mcp_config_missing_args(self):
        """Test _load_mcp_config raises error when args is missing."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("mcp_server:\n  command: npx\n")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    with pytest.raises(ValueError, match="'args' missing in mcp_server"):
                        model._load_mcp_config()
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_mcp_config_valid(self):
        """Test _load_mcp_config with valid configuration."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("""
mcp_server:
  command: npx
  args:
    - "-y"
    - "test-server"
  env:
    TEST_VAR: test_value
""")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    config = model._load_mcp_config()
                    
                    assert config["command"] == "npx"
                    assert config["args"] == ["-y", "test-server"]
                    assert config["env"] == {"TEST_VAR": "test_value"}
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_secrets_missing_file(self):
        """Test _load_secrets raises error when config file is missing."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with patch.object(model, '_find_config_file', return_value=None):
                with pytest.raises(FileNotFoundError, match="config.yaml not found"):
                    model._load_secrets()
                    
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_secrets_valid(self):
        """Test _load_secrets with valid configuration."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("""
secrets:
  - id: secret1
    value: secret_value_1
    env_var: SECRET_VAR_1
  - id: secret2
    env_var: SECRET_VAR_2
""")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    secrets = model._load_secrets()
                    
                    assert len(secrets) == 2
                    assert secrets[0]["id"] == "secret1"
                    assert secrets[0]["value"] == "secret_value_1"
                    assert secrets[0]["env_var"] == "SECRET_VAR_1"
                    assert secrets[1]["id"] == "secret2"
                    assert secrets[1]["env_var"] == "SECRET_VAR_2"
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_load_secrets_empty(self):
        """Test _load_secrets with no secrets section."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("model:\n  id: test\n")
                config_path = f.name
            
            try:
                with patch.object(model, '_find_config_file', return_value=config_path):
                    secrets = model._load_secrets()
                    assert secrets == []
            finally:
                os.unlink(config_path)
                
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_create_tool_function_basic(self):
        """Test _create_tool_function creates valid async functions."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass, StdioMCPClient
            
            model = StdioMCPModelClass()
            
            # Mock stdio client
            mock_client = Mock(spec=StdioMCPClient)
            
            properties = {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            }
            required = ["param1"]
            
            func = model._create_tool_function(
                "test-tool",
                properties,
                required,
                mock_client
            )
            
            # Verify function attributes
            assert callable(func)
            assert asyncio.iscoroutinefunction(func)
            assert "param1" in func.__annotations__
            assert "param2" in func.__annotations__
            assert func.__annotations__["return"] == str
            
        except ImportError:
            pytest.skip("Required MCP packages not installed")

    def test_get_server_raises_import_error_without_fastmcp(self):
        """Test get_server raises ImportError when fastmcp is not available."""
        try:
            from clarifai.runners.models.stdio_mcp_class import StdioMCPModelClass
            
            model = StdioMCPModelClass()
            
            with patch('clarifai.runners.models.stdio_mcp_class.FastMCP', None):
                with pytest.raises(ImportError, match="fastmcp package is required"):
                    model.get_server()
                    
        except ImportError:
            pytest.skip("Required MCP packages not installed")
