"""
Shared test utilities for artifact-related tests.

This module provides common test fixtures and helper functions to reduce
code duplication across test files.
"""

from unittest.mock import Mock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2


def create_mock_artifact_version():
    """Helper function to create an ArtifactVersion with properly mocked dependencies.

    Returns:
        ArtifactVersion: A mocked ArtifactVersion instance with proper auth_helper setup.
    """
    from clarifai.client.artifact_version import ArtifactVersion

    with patch('clarifai.client.base.BaseClient.__init__', return_value=None):
        version = ArtifactVersion()
        # Mock the auth_helper attribute that would normally be set by BaseClient.__init__
        mock_auth_helper = Mock()
        mock_auth_helper.get_user_app_id_proto.return_value = resources_pb2.UserAppIDSet(
            user_id="test_user", app_id="test_app"
        )
        mock_auth_helper.user_id = "mock_user"
        mock_auth_helper.get_stub.return_value = Mock()
        mock_auth_helper.metadata = {}
        version.auth_helper = mock_auth_helper
        return version


@pytest.fixture
def mock_artifact_version():
    """Pytest fixture for creating a mock ArtifactVersion instance.

    Returns:
        ArtifactVersion: A mocked ArtifactVersion instance with proper auth_helper setup.
    """
    return create_mock_artifact_version()


def create_mock_context():
    """Create a mock context object for CLI tests.

    Returns:
        Mock: A mock context object with pat, api_base, and to_grpc attributes.
    """
    mock_current = Mock()
    mock_current.pat = "test_pat"
    mock_current.api_base = "api.clarifai.com"
    mock_current.to_grpc.return_value = {}
    mock_obj = Mock()
    mock_obj.current = mock_current
    return mock_obj


def setup_context_mock(mock_validate):
    """Setup context mock to properly set ctx.obj for CLI tests.

    Args:
        mock_validate: The mock validation function to configure.

    Returns:
        Mock: The configured mock context object.
    """
    mock_obj = create_mock_context()

    def setup_context(ctx):
        ctx.obj = mock_obj

    mock_validate.side_effect = setup_context
    return mock_obj


@pytest.fixture
def mock_cli_context(monkeypatch):
    """Pytest fixture for CLI context mocking.

    Args:
        monkeypatch: Pytest's monkeypatch fixture for patching.

    Returns:
        Mock: A mock context object configured for CLI testing.
    """
    mock_obj = create_mock_context()

    # Patch the validation function to set up the context
    def mock_validate_config(ctx):
        ctx.obj = mock_obj

    monkeypatch.setattr('clarifai.cli.base.validate_config_and_set_context', mock_validate_config)

    return mock_obj
