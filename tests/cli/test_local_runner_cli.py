"""Tests for the serve CLI command.

These tests verify the basic functionality of the `clarifai model serve` command
by mocking external dependencies and testing key behaviors.
"""

import os
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.utils.config import Context


def _make_mock_context(extra_env=None):
    """Create a mock click context with valid credentials.

    Uses a real Context object so dict key access (ctx.obj.current['env'])
    and attribute access (ctx.obj.current.pat) both work correctly.
    """
    env = {
        'CLARIFAI_USER_ID': 'test-user',
        'CLARIFAI_PAT': 'test-pat',
        'CLARIFAI_API_BASE': 'https://api.clarifai.com',
    }
    if extra_env:
        env.update(extra_env)

    context = Context('default', **env)
    mock_ctx = Mock()
    mock_ctx.obj = Mock()
    mock_ctx.obj.current = context
    mock_ctx.obj.contexts = OrderedDict({'default': context})
    mock_ctx.obj.to_yaml = Mock()
    return mock_ctx


def _make_mock_user_with_existing_resources():
    """Create a fully mocked User with existing compute cluster, nodepool, app, model, etc."""
    mock_user = MagicMock()

    mock_compute_cluster = MagicMock()
    mock_compute_cluster.id = "local-dev-cluster"
    mock_compute_cluster.cluster_type = "local-dev"
    mock_user.compute_cluster.return_value = mock_compute_cluster

    mock_nodepool = MagicMock()
    mock_nodepool.id = "local-dev-nodepool"
    mock_compute_cluster.nodepool.return_value = mock_nodepool

    mock_app = MagicMock()
    mock_app.id = "local-dev-app"
    mock_user.app.return_value = mock_app

    mock_model = MagicMock()
    mock_model.id = "dummy-runner-model"
    mock_model.model_type_id = "multimodal-to-text"
    mock_app.model.return_value = mock_model

    # Model version (created fresh every serve)
    mock_version_response = MagicMock()
    mock_version_response.model_version.id = "version-123"
    mock_version_response.load_info = MagicMock()
    mock_model.create_version.return_value = mock_version_response

    # Runner
    mock_runner = MagicMock()
    mock_runner.id = "runner-123"
    mock_nodepool.create_runner.return_value = mock_runner

    # Deployment
    mock_deployment = MagicMock()
    mock_deployment.id = "deployment-123"
    mock_nodepool.create_deployment.return_value = mock_deployment

    # Stale deployment cleanup — no existing deployment to clean up
    mock_nodepool.deployment.side_effect = Exception("Not found")

    return mock_user


class TestLocalRunnerCLI:
    """Test cases for the serve CLI command."""

    @pytest.fixture
    def dummy_model_dir(self):
        """Use the existing dummy_runner_models directory for testing."""
        # Get the path to the dummy_runner_models directory
        tests_dir = Path(__file__).parent.parent
        dummy_model_path = tests_dir / "runners" / "dummy_runner_models"

        if not dummy_model_path.exists():
            pytest.skip(f"Could not find dummy_runner_models at {dummy_model_path}")

        return str(dummy_model_path)

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_requires_installed_dependencies(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that serve checks for installed requirements (mode=none)."""
        mock_check_requirements.return_value = False
        mock_parse_requirements.return_value = []

        mock_builder = MagicMock()
        mock_builder.config = {
            "model": {"id": "dummy-runner-model", "model_type_id": "multimodal-to-text"},
            "toolkit": {},
        }
        mock_builder_class.return_value = mock_builder

        runner = CliRunner()
        runner.invoke(cli, ["login", "--user_id", "test-user", "--pat", "test-pat"])

        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir)],
        )

        assert result.exit_code == 1
        mock_check_requirements.assert_called()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_auto_creates_resources(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that serve auto-creates missing resources without prompting."""
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = []

        mock_builder = MagicMock()
        mock_builder.config = {
            "model": {"id": "dummy-runner-model", "model_type_id": "multimodal-to-text"},
            "toolkit": {},
        }
        mock_method_sig = MagicMock()
        mock_method_sig.name = "predict"
        mock_builder.get_method_signatures.return_value = [mock_method_sig]
        mock_builder_class.return_value = mock_builder

        mock_user = MagicMock()

        # Compute cluster doesn't exist first time, then succeeds after creation
        mock_compute_cluster = MagicMock()
        mock_compute_cluster.id = "local-dev-cluster"
        call_count = {"compute_cluster": 0}

        def compute_cluster_side_effect(cc_id):
            call_count["compute_cluster"] += 1
            if call_count["compute_cluster"] == 1:
                raise Exception("Cluster not found")
            return mock_compute_cluster

        mock_user.compute_cluster.side_effect = compute_cluster_side_effect
        mock_user.create_compute_cluster.return_value = mock_compute_cluster

        # Nodepool doesn't exist first time, then succeeds after creation
        mock_nodepool = MagicMock()
        mock_nodepool.id = "local-dev-nodepool"
        mock_compute_cluster.nodepool.side_effect = Exception("Nodepool not found")
        mock_compute_cluster.create_nodepool.return_value = mock_nodepool

        # App doesn't exist
        mock_app = MagicMock()
        mock_app.id = "local-dev-app"
        mock_user.app.side_effect = Exception("App not found")
        mock_user.create_app.return_value = mock_app

        # Model doesn't exist
        mock_model = MagicMock()
        mock_model.id = "dummy-runner-model"
        mock_model.model_type_id = "multimodal-to-text"
        mock_app.model.side_effect = Exception("Model not found")
        mock_app.create_model.return_value = mock_model

        # Version (always created)
        mock_version_response = MagicMock()
        mock_version_response.model_version.id = "version-123"
        mock_version_response.load_info = MagicMock()
        mock_model.create_version.return_value = mock_version_response

        # Runner
        mock_runner = MagicMock()
        mock_runner.id = "runner-123"
        mock_nodepool.create_runner.return_value = mock_runner

        # Stale deployment cleanup + new deployment creation
        mock_nodepool.deployment.side_effect = Exception("Not found")
        mock_deployment = MagicMock()
        mock_deployment.id = "deployment-123"
        mock_nodepool.create_deployment.return_value = mock_deployment

        mock_user_class.return_value = mock_user

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_ctx = _make_mock_context()

        def validate_ctx_mock(ctx):
            ctx.obj = mock_ctx.obj

        mock_validate_context.side_effect = validate_ctx_mock

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        mock_user.create_compute_cluster.assert_called_once()
        mock_user.create_app.assert_called_once()
        mock_app.create_model.assert_called_once()
        mock_model.create_version.assert_called_once()
        mock_nodepool.create_deployment.assert_called_once()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_uses_existing_resources(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that serve reuses existing resources but always creates a fresh version."""
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = []

        mock_builder = MagicMock()
        mock_builder.config = {
            "model": {"id": "dummy-runner-model", "model_type_id": "multimodal-to-text"},
            "toolkit": {},
        }
        mock_method_sig = MagicMock()
        mock_method_sig.name = "predict"
        mock_builder.get_method_signatures.return_value = [mock_method_sig]
        mock_builder_class.return_value = mock_builder

        mock_user = _make_mock_user_with_existing_resources()
        mock_user_class.return_value = mock_user

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_ctx = _make_mock_context()

        def validate_ctx_mock(ctx):
            ctx.obj = mock_ctx.obj

        mock_validate_context.side_effect = validate_ctx_mock

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Existing resources not re-created
        mock_user.create_compute_cluster.assert_not_called()
        mock_user.create_app.assert_not_called()
        # But version IS always created fresh
        mock_user.app().model().create_version.assert_called_once()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_with_concurrency_parameter(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that serve accepts and uses the --concurrency parameter."""
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = []

        mock_builder = MagicMock()
        mock_builder.config = {
            "model": {"id": "dummy-runner-model", "model_type_id": "multimodal-to-text"},
            "toolkit": {},
        }
        mock_method_sig = MagicMock()
        mock_method_sig.name = "predict"
        mock_builder.get_method_signatures.return_value = [mock_method_sig]
        mock_builder_class.return_value = mock_builder

        mock_user = _make_mock_user_with_existing_resources()
        mock_user_class.return_value = mock_user

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_ctx = _make_mock_context()

        def validate_ctx_mock(ctx):
            ctx.obj = mock_ctx.obj

        mock_validate_context.side_effect = validate_ctx_mock

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--concurrency", "24"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        mock_server.serve.assert_called_once()
        serve_kwargs = mock_server.serve.call_args[1]
        assert serve_kwargs["pool_size"] == 24
        assert serve_kwargs["num_threads"] == 24

    def test_local_runner_has_config_yaml_in_model_dir(self, dummy_model_dir):
        """Test that the dummy model directory contains a config.yaml file."""
        # This is a basic sanity test for the test fixture
        config_path = os.path.join(dummy_model_dir, "config.yaml")
        assert os.path.exists(config_path), "config.yaml should exist in the model directory"

        # Verify it can be loaded
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'model' in config
        assert config['model']['model_type_id'] == "multimodal-to-text"
        assert config['model']['id'] == "dummy-runner-model"

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_local_runner_model_serving(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Test that serve properly initializes and serves the model."""
        mock_check_requirements.return_value = True
        mock_parse_requirements.return_value = []

        mock_builder = MagicMock()
        mock_builder.config = {
            "model": {"id": "dummy-runner-model", "model_type_id": "multimodal-to-text"},
            "toolkit": {},
        }
        mock_method_sig = MagicMock()
        mock_method_sig.name = "predict"
        mock_builder.get_method_signatures.return_value = [mock_method_sig]
        mock_builder_class.return_value = mock_builder

        mock_user = _make_mock_user_with_existing_resources()
        mock_user_class.return_value = mock_user

        mock_server = MagicMock()
        mock_server_class.return_value = mock_server

        mock_ctx = _make_mock_context()

        def validate_ctx_mock(ctx):
            ctx.obj = mock_ctx.obj

        mock_validate_context.side_effect = validate_ctx_mock

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Verify ModelServer was instantiated (without model_builder arg)
        mock_server_class.assert_called_once_with(
            model_path=str(dummy_model_dir), model_runner_local=None
        )

        # Verify serve method was called with correct parameters
        mock_server.serve.assert_called_once()
        serve_kwargs = mock_server.serve.call_args[1]
        assert serve_kwargs["user_id"] == "test-user"
        assert serve_kwargs["base_url"] == "https://api.clarifai.com"
        assert serve_kwargs["pat"] == "test-pat"
        assert "pool_size" in serve_kwargs
        assert "num_threads" in serve_kwargs


STANDARD_PATCHES = [
    "clarifai.cli.model.validate_context",
    "clarifai.cli.model.parse_requirements",
    "clarifai.cli.model.check_requirements_installed",
    "clarifai.runners.models.model_builder.ModelBuilder",
    "clarifai.client.user.User",
    "clarifai.runners.server.ModelServer",
]


def _apply_standard_patches(
    mock_validate_context,
    mock_parse_requirements,
    mock_check_requirements,
    mock_builder_class,
    mock_user_class,
    mock_server_class,
    mock_ctx,
    model_id="dummy-runner-model",
    model_type_id="multimodal-to-text",
):
    """Wire up the standard mocks used by most serve tests."""
    mock_check_requirements.return_value = True
    mock_parse_requirements.return_value = []

    mock_builder = MagicMock()
    mock_builder.config = {
        "model": {"id": model_id, "model_type_id": model_type_id},
        "toolkit": {},
    }
    mock_method_sig = MagicMock()
    mock_method_sig.name = "predict"
    mock_builder.get_method_signatures.return_value = [mock_method_sig]
    mock_builder_class.return_value = mock_builder

    mock_user = _make_mock_user_with_existing_resources()
    mock_user_class.return_value = mock_user

    mock_server = MagicMock()
    mock_server_class.return_value = mock_server

    def validate_ctx_mock(ctx):
        ctx.obj = mock_ctx.obj

    mock_validate_context.side_effect = validate_ctx_mock

    return mock_user, mock_server


class TestKeepFlag:
    """Test cases for the --keep flag on serve command."""

    @pytest.fixture
    def dummy_model_dir(self):
        tests_dir = Path(__file__).parent.parent
        dummy_model_path = tests_dir / "runners" / "dummy_runner_models"
        if not dummy_model_path.exists():
            pytest.skip(f"Could not find dummy_runner_models at {dummy_model_path}")
        return str(dummy_model_path)

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_first_run_creates_resources_and_saves_context(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """First run with --keep creates all resources and saves IDs to context."""
        mock_ctx = _make_mock_context()
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Resources created
        mock_user.app().model().create_version.assert_called_once()
        # Context saved
        mock_ctx.obj.to_yaml.assert_called()
        # CLARIFAI_SERVE_STATE written
        env = mock_ctx.obj.current['env']
        assert 'CLARIFAI_SERVE_STATE' in env
        state = env['CLARIFAI_SERVE_STATE']
        assert 'dummy-runner-model' in state
        saved = state['dummy-runner-model']
        assert 'model_version_id' in saved
        assert 'runner_id' in saved
        assert 'deployment_id' in saved

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_reuses_saved_resources(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Second run with --keep reuses version/runner/deployment from context."""
        mock_ctx = _make_mock_context(
            extra_env={
                'CLARIFAI_SERVE_STATE': {
                    'dummy-runner-model': {
                        'compute_cluster_id': 'local-runner-compute-cluster',
                        'nodepool_id': 'local-runner-nodepool',
                        'model_version_id': 'saved-version-123',
                        'runner_id': 'saved-runner-456',
                        'deployment_id': 'local-dummy-runner-model',
                    },
                },
            }
        )
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        # The mock user from _make_mock_user_with_existing_resources sets
        # deployment.side_effect = Exception. Override it to simulate existing deployment.
        # We need to get the same nodepool mock that the serve code will use.
        mock_cc = mock_user.compute_cluster.return_value
        mock_np = mock_cc.nodepool.return_value

        # Version exists on platform
        mock_model = mock_user.app.return_value.model.return_value
        mock_model.model_info.model_version.id = ""
        mock_model.load_info.return_value = None

        # Runner exists in nodepool
        mock_saved_runner = MagicMock()
        mock_saved_runner.id = "saved-runner-456"
        mock_np.list_runners.return_value = [mock_saved_runner]

        # Deployment exists (override the side_effect from helper)
        mock_np.deployment.side_effect = None
        mock_np.deployment.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Output shows reuse (not "Creating ...")
        assert "Model version ready" in result.output
        assert "Runner ready" in result.output
        assert "Deployment ready" in result.output

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_skips_cleanup_on_exit(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """With --keep, cleanup does not delete deployment/runner/version."""
        mock_ctx = _make_mock_context()
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        assert "--keep mode" in result.output
        # No deletion calls
        mock_nodepool = mock_user.compute_cluster().nodepool()
        mock_nodepool.delete_deployments.assert_not_called()
        mock_nodepool.delete_runners.assert_not_called()

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_stale_version_falls_back_to_create(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """With --keep, if saved version is gone from platform, creates a new one."""
        mock_ctx = _make_mock_context(
            extra_env={
                'CLARIFAI_SERVE_STATE': {
                    'dummy-runner-model': {
                        'model_version_id': 'deleted-version',
                        'runner_id': 'deleted-runner',
                        'deployment_id': 'local-dummy-runner-model',
                    },
                },
            }
        )
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        # Version load fails (deleted on platform)
        mock_model = mock_user.app().model()
        mock_model.load_info.side_effect = Exception("Version not found")

        # Runner not found
        mock_nodepool = mock_user.compute_cluster().nodepool()
        mock_nodepool.list_runners.return_value = []
        mock_nodepool.deployment.side_effect = Exception("Not found")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Fell back to creating new version
        mock_model.create_version.assert_called_once()
        assert (
            "Saved version deleted-v" in result.output or "Creating model version" in result.output
        )

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_backward_compat_old_flat_format(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """With --keep, old flat CLARIFAI_* keys in context are read as fallback."""
        mock_ctx = _make_mock_context(
            extra_env={
                'CLARIFAI_MODEL_ID': 'dummy-runner-model',
                'CLARIFAI_MODEL_VERSION_ID': 'old-version-789',
                'CLARIFAI_RUNNER_ID': 'old-runner-abc',
                'CLARIFAI_DEPLOYMENT_ID': 'local-dummy-runner-model',
                'CLARIFAI_COMPUTE_CLUSTER_ID': 'local-runner-compute-cluster',
                'CLARIFAI_NODEPOOL_ID': 'local-runner-nodepool',
            }
        )
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        # Version exists
        mock_model = mock_user.app().model()
        mock_model.model_info.model_version.id = ""
        mock_model.load_info.return_value = None

        # Runner exists
        mock_nodepool = mock_user.compute_cluster().nodepool()
        mock_saved_runner = MagicMock()
        mock_saved_runner.id = "old-runner-abc"
        mock_nodepool.list_runners.return_value = [mock_saved_runner]
        mock_nodepool.deployment.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Reused from old flat format
        assert "Model version ready" in result.output
        assert "Runner ready" in result.output
        # Saved in new format
        env = mock_ctx.obj.current['env']
        assert 'CLARIFAI_SERVE_STATE' in env

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_keep_old_format_model_id_mismatch_errors(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """With --keep, old format with mismatched CLARIFAI_MODEL_ID errors out."""
        mock_ctx = _make_mock_context(
            extra_env={
                'CLARIFAI_MODEL_ID': 'some-other-model',
                'CLARIFAI_MODEL_VERSION_ID': 'old-version',
                'CLARIFAI_RUNNER_ID': 'old-runner',
            }
        )
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir), "--keep"],
        )

        assert result.exit_code == 1
        # Error message is in the exception, not always in output
        error_text = str(result.exception) if result.exception else result.output
        assert "some-other-model" in error_text
        assert "dummy-runner-model" in error_text

    @patch("clarifai.runners.server.ModelServer")
    @patch("clarifai.client.user.User")
    @patch("clarifai.runners.models.model_builder.ModelBuilder")
    @patch("clarifai.cli.model.check_requirements_installed")
    @patch("clarifai.cli.model.parse_requirements")
    @patch("clarifai.cli.model.validate_context")
    def test_without_keep_ephemeral_behavior_unchanged(
        self,
        mock_validate_context,
        mock_parse_requirements,
        mock_check_requirements,
        mock_builder_class,
        mock_user_class,
        mock_server_class,
        dummy_model_dir,
    ):
        """Without --keep, version is always created fresh (existing behavior)."""
        mock_ctx = _make_mock_context()
        mock_user, mock_server = _apply_standard_patches(
            mock_validate_context,
            mock_parse_requirements,
            mock_check_requirements,
            mock_builder_class,
            mock_user_class,
            mock_server_class,
            mock_ctx,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["model", "serve", str(dummy_model_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Command failed with: {result.output}"
        # Version always created fresh
        mock_user.app().model().create_version.assert_called_once()
        # No CLARIFAI_SERVE_STATE saved
        env = mock_ctx.obj.current['env']
        assert 'CLARIFAI_SERVE_STATE' not in env
