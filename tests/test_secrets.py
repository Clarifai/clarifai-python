import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
from clarifai_grpc.grpc.api import service_pb2

from clarifai.utils.secrets import (
    get_request_secrets,
    get_secret,
    inject_secrets,
    load_secrets_file,
    start_secrets_watcher,
)

NOW = uuid.uuid4().hex[:10]


@pytest.fixture(scope="function")
def secrets_file():
    """Create a temporary secrets file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        secrets_path = Path(f.name)
    yield secrets_path
    # Cleanup
    secrets_path.unlink(missing_ok=True)


@pytest.fixture(scope="function")
def populated_secrets_file():
    """Create a secrets file with test content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("TEST_API_KEY=test_value\\nTEST_SECRET=secret123\\n")
        secrets_path = Path(f.name)
    yield secrets_path
    # Cleanup
    secrets_path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test."""
    original_path = os.environ.get("CLARIFAI_SECRETS_PATH")
    test_keys = ["TEST_API_KEY", "TEST_SECRET", "NEW_SECRET"]

    yield

    # Restore environment
    if original_path:
        os.environ["CLARIFAI_SECRETS_PATH"] = original_path
    elif "CLARIFAI_SECRETS_PATH" in os.environ:
        del os.environ["CLARIFAI_SECRETS_PATH"]

    # Clean up test keys
    for key in test_keys:
        os.environ.pop(key, None)


class TestSecretsSystem:
    """Tests for secrets management system."""

    def test_load_secrets_file(self, populated_secrets_file):
        """Test loading secrets from file."""
        result = load_secrets_file(populated_secrets_file)

        assert result == {"TEST_API_KEY": "test_value", "TEST_SECRET": "secret123"}
        assert os.environ["TEST_API_KEY"] == "test_value"
        assert os.environ["TEST_SECRET"] == "secret123"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        result = load_secrets_file(Path("/nonexistent/file.env"))
        assert result is None

    def test_inject_secrets_into_request(self, populated_secrets_file):
        """Test injecting secrets into protobuf request."""
        os.environ["CLARIFAI_SECRETS_PATH"] = str(populated_secrets_file)
        load_secrets_file(populated_secrets_file)

        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        secrets_from_request = get_request_secrets(request)
        assert secrets_from_request is not None
        assert "TEST_API_KEY" in secrets_from_request
        assert secrets_from_request["TEST_API_KEY"] == "test_value"

        # Test with None request (edge case)
        inject_secrets(None)

        # Test with partial request (edge case)
        partial_request = service_pb2.PostModelOutputsRequest()
        inject_secrets(partial_request)

    def test_file_watcher_detects_changes(self, secrets_file):
        """Test that file watcher detects file modifications."""
        # Create initial file
        secrets_file.write_text("INITIAL_KEY=initial_value\\n")

        callback_called = threading.Event()

        def reload_callback():
            callback_called.set()

        # Start watcher
        watcher = start_secrets_watcher(secrets_file, reload_callback, interval=0.1)
        assert watcher.is_alive()

        # Modify file
        time.sleep(0.2)  # Ensure different timestamp
        secrets_file.write_text("UPDATED_KEY=updated_value\\n")

        # Wait for callback
        assert callback_called.wait(timeout=2.0), "Callback should have been called"

    def test_secrets_helper_function(self, populated_secrets_file):
        """Test the public secrets() helper function."""
        load_secrets_file(populated_secrets_file)

        assert get_secret("TEST_API_KEY") == "test_value"
        assert get_secret("TEST_SECRET") == "secret123"
        assert get_secret("NONEXISTENT") is None

        # Test case-insensitive lookup
        assert get_secret("test_api_key") == "test_value"

    def test_malformed_secrets_file(self, secrets_file):
        """Test handling of malformed .env files."""
        malformed_content = """
        # Comment
        VALID_KEY=valid_value
        INVALID_LINE_NO_EQUALS
        ANOTHER_VALID=value_with=equals
        """
        secrets_file.write_text(malformed_content)

        result = load_secrets_file(secrets_file)

        assert result is not None
        assert "VALID_KEY" in result
        assert "ANOTHER_VALID" in result
        assert result["ANOTHER_VALID"] == "value_with=equals"
        assert "INVALID_LINE_NO_EQUALS" not in str(result)

    def test_end_to_end_workflow(self, secrets_file):
        """Test complete workflow from file creation to request processing."""
        # 1. Create secrets file
        secrets_file.write_text("E2E_KEY=e2e_value\\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # 2. Load secrets
        load_secrets_file(secrets_file)
        assert os.environ["E2E_KEY"] == "e2e_value"

        # 3. Inject into request
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        # 4. Verify secrets in request
        extracted = get_request_secrets(request)
        assert extracted is not None
        assert extracted["E2E_KEY"] == "e2e_value"

        # 5. Update file and verify new secrets
        secrets_file.write_text("E2E_KEY=updated_e2e\\n")
        load_secrets_file(secrets_file)

        new_request = service_pb2.PostModelOutputsRequest()
        inject_secrets(new_request)
        new_extracted = get_request_secrets(new_request)
        assert new_extracted is not None
        assert new_extracted["E2E_KEY"] == "updated_e2e"

        # Test precedence (env vars vs file vs request)
        os.environ["E2E_KEY"] = "env_value"
        secrets_file.write_text("E2E_KEY=file_value\nFILE_ONLY=file_only\n")

        load_secrets_file(secrets_file)
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        extracted = get_request_secrets(request)
        # File should override env in this implementation
        assert extracted is not None
        assert extracted["E2E_KEY"] == "file_value"
        assert extracted["FILE_ONLY"] == "file_only"

    def test_model_server_integration(self, secrets_file):
        """Test end-to-end ModelServer integration with secrets."""
        secrets_file.write_text("SERVER_SECRET=server_value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # Test ModelServer initialization loads secrets
        from clarifai.runners.server import ModelServer

        server = ModelServer("dummy_model_path")  # Would need mocking in real test

        # Verify secrets are loaded
        assert os.environ.get("SERVER_SECRET") == "server_value"

        # Test reload callback
        secrets_file.write_text("SERVER_SECRET=updated_value\n")
        server.reload_model_on_secrets_change()
        assert os.environ.get("SERVER_SECRET") == "updated_value"

    def test_model_reload_on_secrets_change(self, secrets_file, mocker):
        """Test that model is actually reloaded when secrets file changes."""
        # Setup initial secrets
        secrets_file.write_text("INITIAL_SECRET=initial_value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # Mock the ModelBuilder and ModelClass to avoid actual model loading
        mock_builder = mocker.patch('clarifai.runners.server.ModelBuilder')
        mock_model_instance = mocker.MagicMock()
        mock_builder.return_value.create_model_instance.return_value = mock_model_instance

        # Mock servicer and runner
        mock_servicer = mocker.MagicMock()
        mock_runner = mocker.MagicMock()

        from clarifai.runners.server import ModelServer

        server = ModelServer("dummy_path")
        server._servicer = mock_servicer
        server._runner = mock_runner

        # Verify initial model was created
        assert mock_builder.return_value.create_model_instance.call_count == 1

        # Change secrets file
        secrets_file.write_text("UPDATED_SECRET=updated_value\n")

        # Trigger reload
        server.reload_model_on_secrets_change()

        # Verify model was recreated
        assert mock_builder.return_value.create_model_instance.call_count == 2

        # Verify new model was set on servicer and runner
        mock_servicer.set_model.assert_called_once_with(mock_model_instance)
        mock_runner.set_model.assert_called_once_with(mock_model_instance)

        # Verify secrets were reloaded
        assert os.environ.get("UPDATED_SECRET") == "updated_value"

    def test_model_reload_error_handling(self, secrets_file, mocker):
        """Test error handling during model reload."""
        secrets_file.write_text("SECRET=value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # Mock builder to raise exception on second reload
        mock_builder = mocker.patch('clarifai.runners.server.ModelBuilder')
        mock_builder.return_value.create_model_instance.side_effect = [
            mocker.MagicMock(),  # First call succeeds
            Exception("Model creation failed"),  # Second call fails
        ]

        from clarifai.runners.server import ModelServer

        server = ModelServer("dummy_path")

        # Change secrets and trigger reload
        secrets_file.write_text("NEW_SECRET=new_value\n")

        # Should not crash even if model reload fails
        server.reload_model_on_secrets_change()

        # Verify it attempted to reload
        assert mock_builder.return_value.create_model_instance.call_count == 2

    def test_end_to_end_secrets_reload_integration(self, secrets_file, mocker):
        """Test complete integration: file change -> watcher -> reload -> model update."""
        secrets_file.write_text("WATCH_SECRET=initial\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # Mock components
        mock_builder = mocker.patch('clarifai.runners.server.ModelBuilder')
        mock_model = mocker.MagicMock()
        mock_builder.return_value.create_model_instance.return_value = mock_model

        reload_called = threading.Event()
        original_reload = None

        def track_reload():
            reload_called.set()
            if original_reload:
                original_reload()

        from clarifai.runners.server import ModelServer

        server = ModelServer("dummy_path")

        # Patch the reload method to track when it's called
        original_reload = server.reload_model_on_secrets_change
        server.reload_model_on_secrets_change = track_reload

        # Start file watcher (already done in ModelServer.__init__)
        # Change the file
        time.sleep(0.2)  # Ensure different timestamp
        secrets_file.write_text("WATCH_SECRET=updated\n")

        # Wait for reload to be triggered by file watcher
        assert reload_called.wait(timeout=3.0), "Model reload should have been triggered"

        # Verify secrets were updated
        assert os.environ.get("WATCH_SECRET") == "updated"
