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


@pytest.fixture(scope="function")
def secrets_directory():
    """Create a temporary secrets directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        secrets_path = Path(temp_dir)
        yield secrets_path


@pytest.fixture(scope="function")
def populated_secrets_directory():
    """Create a secrets directory with test content in the expected K8s structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        secrets_path = Path(temp_dir)

        # Create TEST_API_KEY secret in K8s structure
        api_key_dir = secrets_path / "TEST_API_KEY"
        api_key_dir.mkdir()
        (api_key_dir / "TEST_API_KEY").write_text("TEST_API_KEY=test_value\n")

        # Create TEST_SECRET secret in K8s structure
        test_secret_dir = secrets_path / "TEST_SECRET"
        test_secret_dir.mkdir()
        (test_secret_dir / "TEST_SECRET").write_text("TEST_SECRET=secret123\n")

        yield secrets_path


@pytest.fixture(scope="function")
def legacy_secrets_file():
    """Create a temporary legacy .env file for backward compatibility tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("LEGACY_KEY=legacy_value\nANOTHER_KEY=another_value\n")
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
        "LEGACY_KEY",
        "ANOTHER_KEY",
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


def create_secret_in_directory(base_path: Path, secret_name: str, secret_content: str):
    """Helper to create a secret in the K8s directory structure."""
    secret_dir = base_path / secret_name
    secret_dir.mkdir(exist_ok=True)
    (secret_dir / secret_name).write_text(secret_content)


class TestSecretsSystem:
    """Tests for secrets management system with K8s directory structure."""

    def test_load_secrets_directory(self, populated_secrets_directory):
        """Test loading secrets from K8s directory structure."""
        result = load_secrets(populated_secrets_directory)
        assert result == {"TEST_API_KEY": "test_value", "TEST_SECRET": "secret123"}
        assert os.environ["TEST_API_KEY"] == "test_value"
        assert os.environ["TEST_SECRET"] == "secret123"

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        result = load_secrets(Path("/nonexistent/directory"))
        assert result is None

    def test_load_empty_directory(self, secrets_directory):
        """Test loading from empty directory."""
        result = load_secrets(secrets_directory)
        assert result == {}

    def test_complex_secret_structures(self, secrets_directory):
        """Test secrets with complex .env content."""
        # Create a secret with multiple key-value pairs
        multi_content = "DB_HOST=localhost\nDB_PORT=5432\nDB_NAME=myapp\n"
        create_secret_in_directory(secrets_directory, "database-config", multi_content)

        # Create a secret with special characters
        special_content = "API_KEY=sk-1234567890abcdef\nSECRET_TOKEN=Bearer xyz123==\n"
        create_secret_in_directory(secrets_directory, "api-credentials", special_content)

        result = load_secrets(secrets_directory)
        assert result is not None
        assert result["DB_HOST"] == "localhost"
        assert result["DB_PORT"] == "5432"
        assert result["DB_NAME"] == "myapp"
        assert result["API_KEY"] == "sk-1234567890abcdef"
        assert result["SECRET_TOKEN"] == "Bearer xyz123=="

    def test_malformed_secret_files(self, secrets_directory):
        """Test handling of malformed secret files."""
        # Create a valid secret
        create_secret_in_directory(secrets_directory, "valid-secret", "VALID_KEY=valid_value\n")

        # Create a malformed secret (no equals signs)
        create_secret_in_directory(
            secrets_directory, "malformed-secret", "INVALID_LINE_NO_EQUALS\n"
        )

        # Create a secret with mixed valid/invalid content
        mixed_content = """
        # Comment
        VALID_KEY=valid_value
        INVALID_LINE_NO_EQUALS
        ANOTHER_VALID=value_with=equals
        """
        create_secret_in_directory(secrets_directory, "mixed-secret", mixed_content)

        result = load_secrets(secrets_directory)
        assert result is not None
        assert "VALID_KEY" in result
        assert "ANOTHER_VALID" in result
        assert result["ANOTHER_VALID"] == "value_with=equals"

    def test_inject_secrets_into_request(self, populated_secrets_directory):
        """Test injecting secrets into protobuf request with proper environment setup."""
        # Critical: Set environment BEFORE loading secrets
        os.environ["CLARIFAI_SECRETS_PATH"] = str(populated_secrets_directory)

        # Load secrets into environment
        result = load_secrets(populated_secrets_directory)
        assert result is not None, "Secrets should load successfully"
        assert len(result) > 0, "Should have loaded at least one secret"

        # Create and inject into request
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        # Verify injection worked
        secrets_from_request = get_request_secrets(request)
        assert secrets_from_request is not None, "Request should contain injected secrets"
        assert "TEST_API_KEY" in secrets_from_request, "TEST_API_KEY should be in request"
        assert secrets_from_request["TEST_API_KEY"] == "test_value", "Secret value should match"

        # Test edge cases
        inject_secrets(None)  # Should not crash

        empty_request = service_pb2.PostModelOutputsRequest()
        inject_secrets(empty_request)  # Should create proper structure

    def test_directory_watcher_detects_changes(self, secrets_directory):
        """Test directory watcher with corrected timing and change detection."""
        # Create initial secret
        create_secret_in_directory(secrets_directory, "INITIAL_KEY", "INITIAL_KEY=initial_value\n")

        callback_events = []
        callback_lock = threading.Lock()

        def test_callback():
            with callback_lock:
                callback_events.append(time.time())

        # Start watcher with aggressive interval for testing
        watcher_thread = start_secrets_watcher(secrets_directory, test_callback, interval=0.05)

        try:
            assert watcher_thread.is_alive(), "Watcher thread should be running"

            # Give watcher time to establish baseline and detect initial file
            time.sleep(0.2)

            # Verify initial state
            initial_count = len(callback_events)

            # Modify existing secret - this should trigger callback
            time.sleep(0.1)  # Ensure different mtime
            create_secret_in_directory(
                secrets_directory, "INITIAL_KEY", "INITIAL_KEY=updated_value\n"
            )

            # Wait for detection with timeout
            timeout = time.time() + 3.0
            while len(callback_events) <= initial_count and time.time() < timeout:
                time.sleep(0.05)

            assert len(callback_events) > initial_count, (
                f"Should detect file modification. Events: {len(callback_events)}"
            )

            # Test new file detection
            previous_count = len(callback_events)
            time.sleep(0.1)  # Ensure different mtime
            create_secret_in_directory(secrets_directory, "NEW_SECRET", "NEW_SECRET=new_value\n")

            # Wait for new file detection
            timeout = time.time() + 3.0
            while len(callback_events) <= previous_count and time.time() < timeout:
                time.sleep(0.05)

            assert len(callback_events) > previous_count, (
                f"Should detect new file. Events: {len(callback_events)}"
            )

        finally:
            # Cleanup - thread will die when test ends due to daemon=True
            pass

    def test_secrets_helper_function(self, populated_secrets_directory):
        """Test the public get_secret() helper function."""
        load_secrets(populated_secrets_directory)

        assert get_secret("TEST_API_KEY") == "test_value"
        assert get_secret("TEST_SECRET") == "secret123"
        assert get_secret("NONEXISTENT") is None

        # Test case-insensitive lookup for environment variables
        assert get_secret("test_api_key") == "test_value"

    def test_end_to_end_workflow(self, secrets_directory):
        """Test complete workflow from directory creation to request processing."""
        # 1. Create secrets directory structure
        create_secret_in_directory(secrets_directory, "E2E_KEY", "E2E_KEY=e2e_value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_directory)

        # 2. Load secrets
        load_secrets(secrets_directory)
        assert os.environ["E2E_KEY"] == "e2e_value"

        # 3. Inject into request
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        # 4. Verify secrets in request
        extracted = get_request_secrets(request)
        assert extracted is not None
        assert extracted["E2E_KEY"] == "e2e_value"

        # 5. Update secret and verify new value
        time.sleep(0.01)  # Ensure different mtime for change detection
        create_secret_in_directory(secrets_directory, "E2E_KEY", "E2E_KEY=updated_e2e\n")
        load_secrets(secrets_directory)

        new_request = service_pb2.PostModelOutputsRequest()
        inject_secrets(new_request)
        new_extracted = get_request_secrets(new_request)
        assert new_extracted is not None
        assert new_extracted["E2E_KEY"] == "updated_e2e"

    def test_secrets_precedence(self, secrets_directory):
        """Test precedence: loaded secrets should set environment variables."""
        # Set up environment variable
        os.environ["PRECEDENCE_KEY"] = "env_value"
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_directory)

        # Create secrets with different values
        create_secret_in_directory(
            secrets_directory,
            "precedence-secret",
            "PRECEDENCE_KEY=file_value\nFILE_ONLY=file_only\n",
        )

        load_secrets(secrets_directory)
        request = service_pb2.PostModelOutputsRequest()
        inject_secrets(request)

        extracted = get_request_secrets(request)
        assert extracted is not None
        assert extracted["PRECEDENCE_KEY"] == "file_value"  # File should override env
        assert extracted["FILE_ONLY"] == "file_only"

        # Verify environment was updated
        assert os.environ["PRECEDENCE_KEY"] == "file_value"
        assert os.environ["FILE_ONLY"] == "file_only"

    def test_model_server_integration(self, secrets_directory):
        """Test ModelServer integration with proper initialization sequence."""
        # Setup secrets BEFORE creating server
        create_secret_in_directory(secrets_directory, "server-secret", "SERVER_SECRET=initial\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_directory)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            # Don't mock the watcher - let it run normally for this test
            mock_model = MagicMock()
            mock_builder.return_value.create_model_instance.return_value = mock_model

            from clarifai.runners.server import ModelServer

            # Create server - this should load initial secrets and start watcher
            server = ModelServer("dummy_path")

            # Verify initial state
            assert mock_builder.return_value.create_model_instance.call_count == 1
            assert os.environ.get("SERVER_SECRET") == "initial", "Initial secret should be loaded"

            # Update secrets and trigger reload manually
            # (In production, the watcher would trigger this automatically)
            create_secret_in_directory(
                secrets_directory, "server-secret", "SERVER_SECRET=updated\n"
            )

            # Manually trigger reload to simulate watcher behavior
            server.reload_model_on_secrets_change()

            # Verify reload occurred
            assert os.environ.get("SERVER_SECRET") == "updated", "Secret should be updated"
            assert mock_builder.return_value.create_model_instance.call_count == 2, (
                "Model should be rebuilt"
            )

    def test_model_reload_on_secrets_change(self, secrets_directory):
        """Test comprehensive model reload with component updates."""
        # Setup initial state
        create_secret_in_directory(
            secrets_directory, "reload-secret", "RELOAD_SECRET=initial_value\n"
        )
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_directory)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            mock_model_instance = MagicMock()
            mock_builder.return_value.create_model_instance.return_value = mock_model_instance

            # Mock servicer and runner with proper set_model methods
            mock_servicer = MagicMock()
            mock_runner = MagicMock()

            from clarifai.runners.server import ModelServer

            server = ModelServer("dummy_path")
            server._servicer = mock_servicer
            server._runner = mock_runner

            # Verify initial state
            assert mock_builder.return_value.create_model_instance.call_count == 1
            assert os.environ.get("RELOAD_SECRET") == "initial_value"

            # Update secrets and trigger reload
            time.sleep(0.01)  # Ensure different mtime
            create_secret_in_directory(
                secrets_directory, "reload-secret", "RELOAD_SECRET=updated_value\n"
            )

            # Trigger reload sequence
            server.reload_model_on_secrets_change()

            # Verify complete reload sequence
            assert mock_builder.return_value.create_model_instance.call_count == 2, (
                "Model should be rebuilt"
            )
            mock_servicer.set_model.assert_called_once_with(mock_model_instance)
            mock_runner.set_model.assert_called_once_with(mock_model_instance)
            assert os.environ.get("RELOAD_SECRET") == "updated_value", (
                "Environment should be updated"
            )

    def test_model_reload_error_handling(self, secrets_directory):
        """Test error handling during model reload."""
        create_secret_in_directory(secrets_directory, "error-secret", "SECRET=value\n")
        os.environ["CLARIFAI_SECRETS_PATH"] = str(secrets_directory)

        with patch('clarifai.runners.server.ModelBuilder') as mock_builder:
            mock_builder.return_value.create_model_instance.side_effect = [
                MagicMock(),  # First call succeeds
                Exception("Model creation failed"),  # Second call fails
            ]

            from clarifai.runners.server import ModelServer

            server = ModelServer("dummy_path")

            # Change secrets and trigger reload
            create_secret_in_directory(secrets_directory, "new-secret", "NEW_SECRET=new_value\n")

            # Should not crash even if model reload fails
            server.reload_model_on_secrets_change()

            # Verify it attempted to reload
            assert mock_builder.return_value.create_model_instance.call_count == 2

    def test_backward_compatibility_with_legacy_files(self, legacy_secrets_file):
        """Test that the system gracefully handles legacy .env files (if supported)."""
        # This test would only pass if you add backward compatibility logic
        # For now, we expect it to fail gracefully
        result = load_secrets(legacy_secrets_file)
        # Since your current implementation expects directories, this should return None
        assert result is None

    def test_missing_secret_files_in_directories(self, secrets_directory):
        """Test handling of directories without corresponding secret files."""
        # Create directory structure but don't create the expected files
        orphan_dir = secrets_directory / "orphan-secret"
        orphan_dir.mkdir()
        # Don't create the secret file inside

        # Create a valid secret for comparison
        create_secret_in_directory(secrets_directory, "valid-secret", "VALID_KEY=valid_value\n")

        result = load_secrets(secrets_directory)
        assert result is not None
        assert "VALID_KEY" in result
        # Should not include anything from the orphan directory
        assert len([k for k in result.keys() if "orphan" in k.lower()]) == 0

    def test_empty_secret_files(self, secrets_directory):
        """Test handling of empty secret files."""
        # Create an empty secret file
        empty_dir = secrets_directory / "empty-secret"
        empty_dir.mkdir()
        (empty_dir / "empty-secret").write_text("")

        # Create a valid secret for comparison
        create_secret_in_directory(secrets_directory, "valid-secret", "VALID_KEY=valid_value\n")

        result = load_secrets(secrets_directory)
        assert result is not None
        assert "VALID_KEY" in result
        # Empty files should not contribute any keys
        assert len([k for k in result.keys() if "empty" in k.lower()]) == 0

    def test_extract_req_secrets_from_request(self):
        """Test extracting request-type secrets from PostModelOutputsRequest."""
        from google.protobuf.struct_pb2 import Struct

        from clarifai.utils.secrets import extract_req_secrets

        # Create a request with request-type secrets in model parameters
        request = service_pb2.PostModelOutputsRequest()
        request.model.model_version.output_info.params.CopyFrom(Struct())

        # Simulate the structure that would contain req-type secrets
        secrets_data = [
            {
                "id": "secret1",
                "secret_type": "req",
                "env_var": "REQ_API_KEY",
                "value": "req_secret_value_1",
            },
            {
                "id": "secret2",
                "secret_type": "system",  # This should be ignored
                "env_var": "SYSTEM_KEY",
                "value": "system_value",
            },
            {
                "id": "secret3",
                "secret_type": "req",
                "env_var": "REQ_TOKEN",
                "value": "req_secret_value_2",
            },
        ]

        # Set the secrets in the params
        request.model.model_version.output_info.params["secrets"] = secrets_data

        # Extract req-type secrets
        req_secrets = extract_req_secrets(request)

        # Should only extract the req-type secrets
        assert len(req_secrets) == 2
        assert {"env_var": "REQ_API_KEY", "value": "req_secret_value_1"} in req_secrets
        assert {"env_var": "REQ_TOKEN", "value": "req_secret_value_2"} in req_secrets

        # Should not include system-type secret
        system_secrets = [s for s in req_secrets if s["env_var"] == "SYSTEM_KEY"]
        assert len(system_secrets) == 0

    def test_extract_req_secrets_empty_request(self):
        """Test extracting secrets from empty or invalid requests."""
        from clarifai.utils.secrets import extract_req_secrets

        # Test with None request
        assert extract_req_secrets(None) == []

        # Test with empty request
        empty_request = service_pb2.PostModelOutputsRequest()
        assert extract_req_secrets(empty_request) == []

        # Test with request missing model field
        request_no_model = service_pb2.PostModelOutputsRequest()
        assert extract_req_secrets(request_no_model) == []

    def test_extract_req_secrets_malformed_data(self):
        """Test extracting secrets with malformed or incomplete data."""
        from google.protobuf.struct_pb2 import Struct

        from clarifai.utils.secrets import extract_req_secrets

        request = service_pb2.PostModelOutputsRequest()
        request.model.model_version.output_info.params.CopyFrom(Struct())

        # Test with malformed secrets (missing required fields)
        malformed_secrets = [
            {"id": "secret1", "secret_type": "req"},  # Missing env_var and value
            {"id": "secret2", "secret_type": "req", "env_var": "MISSING_VALUE"},  # Missing value
            {"id": "secret3", "secret_type": "req", "value": "missing_env_var"},  # Missing env_var
            {
                "id": "secret4",
                "secret_type": "req",
                "env_var": "VALID_KEY",
                "value": "valid_value",
            },  # Valid
        ]

        request.model.model_version.output_info.params["secrets"] = malformed_secrets
        req_secrets = extract_req_secrets(request)

        # Should only extract the valid secret
        assert len(req_secrets) == 1
        assert req_secrets[0] == {"env_var": "VALID_KEY", "value": "valid_value"}

    def test_temporary_env_vars_context_manager(self):
        """Test the temporary_env_vars context manager."""
        from clarifai.utils.secrets import temporary_env_vars

        # Set up initial environment state
        original_value = "original"
        os.environ["TEMP_TEST_VAR"] = original_value

        # Remove any existing TEST_NEW_VAR
        if "TEST_NEW_VAR" in os.environ:
            del os.environ["TEST_NEW_VAR"]

        test_env_vars = {"TEMP_TEST_VAR": "temporary_value", "TEST_NEW_VAR": "new_value"}

        # Test that variables are set within context
        with temporary_env_vars(test_env_vars):
            assert os.environ["TEMP_TEST_VAR"] == "temporary_value"
            assert os.environ["TEST_NEW_VAR"] == "new_value"

        # Test that original state is restored
        assert os.environ["TEMP_TEST_VAR"] == original_value
        assert "TEST_NEW_VAR" not in os.environ

    def test_temporary_env_vars_exception_handling(self):
        """Test that environment is restored even when exception occurs."""
        from clarifai.utils.secrets import temporary_env_vars

        # Set initial state
        os.environ["EXCEPTION_TEST_VAR"] = "original"

        test_env_vars = {"EXCEPTION_TEST_VAR": "temporary"}

        try:
            with temporary_env_vars(test_env_vars):
                assert os.environ["EXCEPTION_TEST_VAR"] == "temporary"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore original value even after exception
        assert os.environ["EXCEPTION_TEST_VAR"] == "original"

    def test_req_secrets_context_full_workflow(self):
        """Test the complete req_secrets_context workflow with multiple secrets."""
        from google.protobuf.struct_pb2 import Struct

        from clarifai.utils.secrets import req_secrets_context

        # Create request with multiple req-type secrets
        request = service_pb2.PostModelOutputsRequest()
        request.model.model_version.output_info.params.CopyFrom(Struct())

        secrets_data = [
            {
                "id": "secret1",
                "secret_type": "req",
                "env_var": "WORKFLOW_API_KEY",
                "value": "api_value",
            },
            {
                "id": "secret2",
                "secret_type": "req",
                "env_var": "WORKFLOW_TOKEN",
                "value": "token_value",
            },
        ]

        request.model.model_version.output_info.params["secrets"] = secrets_data

        # Clean initial state
        for var in ["WORKFLOW_API_KEY", "WORKFLOW_TOKEN"]:
            if var in os.environ:
                del os.environ[var]

        # Test that secrets are available within context
        with req_secrets_context(request):
            assert os.environ.get("WORKFLOW_API_KEY") == "api_value"
            assert os.environ.get("WORKFLOW_TOKEN") == "token_value"

        # Test that secrets are removed after context
        assert "WORKFLOW_API_KEY" not in os.environ
        assert "WORKFLOW_TOKEN" not in os.environ

    def test_req_secrets_context_with_existing_env_vars(self):
        """Test req_secrets_context preserves original environment variables."""
        from google.protobuf.struct_pb2 import Struct

        from clarifai.utils.secrets import req_secrets_context

        # Set up existing environment variable
        original_value = "original_api_key"
        os.environ["CONTEXT_API_KEY"] = original_value

        # Create request that will override the existing variable
        request = service_pb2.PostModelOutputsRequest()
        request.model.model_version.output_info.params.CopyFrom(Struct())

        secrets_data = [
            {
                "id": "override_secret",
                "secret_type": "req",
                "env_var": "CONTEXT_API_KEY",
                "value": "request_override_value",
            }
        ]

        request.model.model_version.output_info.params["secrets"] = secrets_data

        # Test override within context
        with req_secrets_context(request):
            assert os.environ["CONTEXT_API_KEY"] == "request_override_value"

        # Test restoration after context
        assert os.environ["CONTEXT_API_KEY"] == original_value
