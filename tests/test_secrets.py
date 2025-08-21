import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from clarifai_grpc.grpc.api import service_pb2

from clarifai.utils.secrets import (
    get_request_secrets,
    get_secret,
    inject_secrets,
    load_secrets,
    start_secrets_watcher,
)

NOW = uuid.uuid4().hex[:10]


@pytest.fixture(autouse=True)
def clear_secrets_cache():
    """Clear secrets cache before each test."""
    import clarifai.utils.secrets as sm

    sm._secrets_cache.clear()
    sm._last_cache_time = 0
    yield
    sm._secrets_cache.clear()
    sm._last_cache_time = 0


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
        f.write("TEST_API_KEY=test_value\nTEST_SECRET=secret123\n")
        secrets_path = Path(f.name)
    yield secrets_path
    # Cleanup
    secrets_path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test."""
    original_path = os.environ.get("CLARIFAI_SECRETS_PATH")
    test_keys = [
        "TEST_API_KEY",
        "TEST_SECRET",
        "NEW_SECRET",
        "E2E_KEY",
        "SERVER_SECRET",
        "INITIAL_SECRET",
        "UPDATED_SECRET",
        "PRECEDENCE_KEY",
        "RELOAD_SECRET",
    ]
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
        result = load_secrets(populated_secrets_file)
        assert result == {"TEST_API_KEY": "test_value", "TEST_SECRET": "secret123"}
        assert os.environ["TEST_API_KEY"] == "test_value"
        assert os.environ["TEST_SECRET"] == "secret123"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        result = load_secrets(Path("/nonexistent/file.env"))
        assert result is None

    def test_inject_secrets_into_request(self, populated_secrets_file):
        """Test injecting secrets into protobuf request."""
        os.environ["CLARIFAI_SECRETS_PATH"] = str(populated_secrets_file)
        load_secrets(populated_secrets_file)

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
        secrets_file.write_text("INITIAL_KEY=initial_value\n")

        callback_called = threading.Event()
        callback_count = 0

        def test_callback():
            nonlocal callback_count
            callback_count += 1
            callback_called.set()

        # Start watcher with short interval
        watcher_thread = start_secrets_watcher(secrets_file, test_callback, interval=0.1)

        try:
            # Verify thread started
            assert watcher_thread.is_alive(), "Watcher thread should be running"

            # Give watcher time to establish baseline
            time.sleep(0.2)

            # Modify file
            secrets_file.write_text("UPDATED_KEY=updated_value\n")

            # Wait for callback
            assert callback_called.wait(timeout=2.0), "Callback should have been triggered"
            assert callback_count > 0, "Callback should have been called"
        finally:
            # Note: The watcher thread is daemon, so it will die with the test
            pass

    def test_secrets_helper_function(self, populated_secrets_file):
        """Test the public get_secret() helper function."""
        load_secrets(populated_secrets_file)

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

        result = load_secrets(secrets_file)

        assert result is not None
        assert "VALID_KEY" in result
        assert "ANOTHER_VALID" in result
        assert result["ANOTHER_VALID"] == "value_with=equals"
        assert "INVALID_LINE_NO_EQUALS" not in str(result)

    def test_end_to_end_workflow(self, secrets_file):
        """Test complete workflow from file creation to request processing."""
        # 1. Create secrets file
        secrets_file.write_text("E2E_KEY=e2e_value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # 2. Load secrets
        load_secrets(secrets_file)
        assert os.environ["E2E_KEY"] == "e2e_value"

        # 3. Inject into request
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        # 4. Verify secrets in request
        extracted = get_request_secrets(request)
        assert extracted is not None
        assert extracted["E2E_KEY"] == "e2e_value"

        # 5. Update file and verify new secrets
        time.sleep(0.01)  # Ensure different mtime for cache invalidation
        secrets_file.write_text("E2E_KEY=updated_e2e\n")
        load_secrets(secrets_file)

        new_request = service_pb2.PostModelOutputsRequest()
        inject_secrets(new_request)
        new_extracted = get_request_secrets(new_request)
        assert new_extracted is not None
        assert new_extracted["E2E_KEY"] == "updated_e2e"

    def test_secrets_precedence(self, secrets_file):
        """Test precedence: file secrets should override environment variables."""
        # Set up environment variable
        os.environ["PRECEDENCE_KEY"] = "env_value"
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        # Create file with different value
        secrets_file.write_text("PRECEDENCE_KEY=file_value\nFILE_ONLY=file_only\n")

        load_secrets(secrets_file)
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        extracted = get_request_secrets(request)
        # File should override env in this implementation
        assert extracted is not None
        assert extracted["PRECEDENCE_KEY"] == "file_value"
        assert extracted["FILE_ONLY"] == "file_only"

    def test_model_server_integration(self, secrets_file):
        """Test ModelServer secrets integration by calling reload directly."""
        secrets_file.write_text("SERVER_SECRET=initial\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            # Disable the watcher for this test to focus on reload functionality
            with patch('clarifai.runners.server.start_secrets_watcher') as mock_watcher:
                mock_model = MagicMock()
                mock_builder.return_value.create_model_instance.return_value = mock_model

                from clarifai.runners.server import ModelServer

                server = ModelServer("dummy_path")

                # Verify watcher was attempted to be started
                mock_watcher.assert_called_once()

                # Verify initial secrets loaded
                assert os.environ.get("SERVER_SECRET") == "initial"
                assert mock_builder.return_value.create_model_instance.call_count == 1

                # Update secrets file
                secrets_file.write_text("SERVER_SECRET=updated\n")

                # Manually trigger reload (simulating what watcher would do)
                server.reload_model_on_secrets_change()

                # Verify secrets were reloaded and model was recreated
                assert os.environ.get("SERVER_SECRET") == "updated"
                assert mock_builder.return_value.create_model_instance.call_count == 2

    def test_model_reload_on_secrets_change(self, secrets_file):
        """Test that model is reloaded and servicer/runner are updated correctly."""
        # Setup initial secrets
        secrets_file.write_text("RELOAD_SECRET=initial_value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            mock_model_instance = MagicMock()
            mock_builder.return_value.create_model_instance.return_value = mock_model_instance

            # Mock servicer and runner
            mock_servicer = MagicMock()
            mock_runner = MagicMock()

            from clarifai.runners.server import ModelServer

            server = ModelServer("dummy_path")
            server._servicer = mock_servicer
            server._runner = mock_runner

            # Verify initial model was created and initial secrets loaded
            assert mock_builder.return_value.create_model_instance.call_count == 1
            assert os.environ.get("RELOAD_SECRET") == "initial_value"

            # Change secrets file - update the same key to test replacement
            time.sleep(0.01)  # Ensure different mtime for cache invalidation
            secrets_file.write_text("RELOAD_SECRET=updated_value\n")

            # Trigger reload
            server.reload_model_on_secrets_change()

            # Verify model was recreated
            assert mock_builder.return_value.create_model_instance.call_count == 2

            # Verify new model was set on servicer and runner
            mock_servicer.set_model.assert_called_once_with(mock_model_instance)
            mock_runner.set_model.assert_called_once_with(mock_model_instance)

            # Verify secrets were reloaded with new value
            assert os.environ.get("RELOAD_SECRET") == "updated_value"

    def test_model_reload_error_handling(self, secrets_file):
        """Test error handling during model reload."""
        secrets_file.write_text("SECRET=value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_file)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            mock_builder.return_value.create_model_instance.side_effect = [
                MagicMock(),  # First call succeeds
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
