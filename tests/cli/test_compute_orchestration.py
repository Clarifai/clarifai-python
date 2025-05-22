import os
import uuid
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.client.compute_cluster import ComputeCluster
from clarifai.client.nodepool import Nodepool
from clarifai.client.user import User
from clarifai.utils.logging import logger

NOW = uuid.uuid4()

CREATE_COMPUTE_CLUSTER_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_COMPUTE_CLUSTER_ID = f"ci_test_cc_{NOW}"
CREATE_NODEPOOL_ID = f"ci_test_np_{NOW}"
CREATE_DEPLOYMENT_ID = f"ci_test_dep_{NOW}"

COMPUTE_CLUSTER_CONFIG_FILE = (
    "tests/compute_orchestration/configs/example_compute_cluster_config.yaml"
)
NODEPOOL_CONFIG_FILE = "tests/compute_orchestration/configs/example_nodepool_config.yaml"
DEPLOYMENT_CONFIG_FILE = "tests/compute_orchestration/configs/example_deployment_config.yaml"

CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]
CLARIFAI_ENV = os.environ.get("CLARIFAI_ENV", "prod")
CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


@pytest.fixture
def create_compute_cluster():
    return ComputeCluster(
        user_id=CREATE_COMPUTE_CLUSTER_USER_ID,
        compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID,
        pat=CLARIFAI_PAT,
        base_url=CLARIFAI_API_BASE,
    )


@pytest.fixture
def create_nodepool():
    return Nodepool(
        user_id=CREATE_COMPUTE_CLUSTER_USER_ID,
        nodepool_id=CREATE_NODEPOOL_ID,
        pat=CLARIFAI_PAT,
        base_url=CLARIFAI_API_BASE,
    )


@pytest.fixture
def cli_runner():
    return CliRunner(
        env={
            "CLARIFAI_USER_ID": CREATE_COMPUTE_CLUSTER_USER_ID,
            "CLARIFAI_PAT": CLARIFAI_PAT,
            "CLARIFAI_API_BASE": CLARIFAI_API_BASE,
        }
    )


@pytest.mark.requires_secrets
class TestComputeOrchestration:
    """Tests for the Compute Orchestration resources on CLI.
    CRUD operations are tested for each of the following resources:
    - compute cluster
    - nodepool
    - deployment
    """

    @classmethod
    def setup_class(cls):
        """Setup: Clean up any pre-existing resources before tests."""
        cls.client = User(
            user_id=CREATE_COMPUTE_CLUSTER_USER_ID, pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE
        )
        cls.compute_cluster = create_compute_cluster
        cls.nodepool = create_nodepool
        cls._cleanup_resources()

    @classmethod
    def teardown_class(cls):
        """Teardown: Clean up any resources created during the tests."""
        cls._cleanup_resources()

    @classmethod
    def _cleanup_resources(cls):
        """Helper function to delete any existing resources."""
        try:
            cls.nodepool.delete_deployments([CREATE_DEPLOYMENT_ID])
        except Exception:
            pass  # Ignore if not found

        try:
            cls.compute_cluster.delete_nodepools([CREATE_NODEPOOL_ID])
        except Exception:
            pass  # Ignore if not found

        try:
            cls.client.delete_compute_clusters([CREATE_COMPUTE_CLUSTER_ID])
        except Exception:
            pass  # Ignore if not found

    def test_create_compute_cluster(self, cli_runner):
        with open(COMPUTE_CLUSTER_CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        config["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
        with open(COMPUTE_CLUSTER_CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli,
            [
                "computecluster",
                "create",
                CREATE_COMPUTE_CLUSTER_ID,
                "--config",
                COMPUTE_CLUSTER_CONFIG_FILE,
            ],
        )
        assert result.exit_code == 0, logger.exception(result)

    def test_create_nodepool(self, cli_runner):
        with open(NODEPOOL_CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        config["nodepool"]["id"] = CREATE_NODEPOOL_ID
        with open(NODEPOOL_CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli,
            [
                "nodepool",
                "create",
                CREATE_COMPUTE_CLUSTER_ID,
                CREATE_NODEPOOL_ID,
                "--config",
                NODEPOOL_CONFIG_FILE,
            ],
        )
        assert result.exit_code == 0, logger.exception(result)

    @pytest.mark.coverage_only
    def test_create_deployment(self, cli_runner):
        with open(DEPLOYMENT_CONFIG_FILE) as f:
            config = yaml.safe_load(f)
        config["deployment"]["id"] = CREATE_DEPLOYMENT_ID
        config["deployment"]["nodepools"][0]["id"] = CREATE_NODEPOOL_ID
        config["deployment"]["nodepools"][0]["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
        with open(DEPLOYMENT_CONFIG_FILE, "w") as f:
            yaml.dump(config, f)

        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli,
            [
                "deployment",
                "create",
                CREATE_NODEPOOL_ID,
                CREATE_DEPLOYMENT_ID,
                "--config",
                DEPLOYMENT_CONFIG_FILE,
            ],
        )
        assert result.exit_code == 0, logger.exception(result)

    def test_list_compute_clusters(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(cli, ["computecluster", "list"])
        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    def test_list_nodepools(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(cli, ["nodepool", "list", CREATE_COMPUTE_CLUSTER_ID])
        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    def test_list_deployments(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(cli, ["deployment", "list", CREATE_NODEPOOL_ID])

        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    @pytest.mark.coverage_only
    def test_delete_deployment(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli, ["deployment", "delete", CREATE_NODEPOOL_ID, CREATE_DEPLOYMENT_ID]
        )
        assert result.exit_code == 0, logger.exception(result)

    def test_delete_nodepool(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli, ["nodepool", "delete", CREATE_COMPUTE_CLUSTER_ID, CREATE_NODEPOOL_ID]
        )
        assert result.exit_code == 0, logger.exception(result)

    def test_delete_compute_cluster(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(cli, ["computecluster", "delete", CREATE_COMPUTE_CLUSTER_ID])
        assert result.exit_code == 0, logger.exception(result)
        
        
@pytest.mark.requires_secrets
class TestLocalDevCLI:
    """Tests for the local_dev CLI functionality."""
    
    @pytest.fixture
    def mock_user(self):
        """Mock User class and its methods."""
        with mock.patch("clarifai.cli.model.User") as mock_user:
            # Create mock for the compute_cluster method
            mock_compute_cluster = mock.MagicMock()
            mock_compute_cluster.cluster_type = 'local-dev'
            mock_user.return_value.compute_cluster.return_value = mock_compute_cluster
            
            # Create mock for nodepool
            mock_nodepool = mock.MagicMock()
            mock_compute_cluster.nodepool.return_value = mock_nodepool
            
            # Create mock for runner
            mock_runner = mock.MagicMock()
            mock_nodepool.runner.return_value = mock_runner
            
            yield mock_user
    
    @pytest.fixture
    def mock_model_builder(self):
        """Mock ModelBuilder class."""
        with mock.patch("clarifai.cli.model.ModelBuilder") as mock_builder:
            mock_instance = mock_builder.return_value
            mock_instance.get_method_signatures.return_value = [
                {"method_name": "test_method", "parameters": []}
            ]
            yield mock_builder
    
    @pytest.fixture
    def mock_serve(self):
        """Mock serve function."""
        with mock.patch("clarifai.cli.model.serve") as mock_serve:
            yield mock_serve
    
    @pytest.fixture
    def mock_validate_context(self):
        """Mock validate_context function."""
        with mock.patch("clarifai.cli.model.validate_context") as mock_validate:
            yield mock_validate
    
    @pytest.fixture
    def mock_code_script(self):
        """Mock code_script module."""
        with mock.patch("clarifai.cli.model.code_script") as mock_code_script:
            mock_code_script.generate_client_script.return_value = "TEST_SCRIPT"
            yield mock_code_script
            
    @pytest.fixture
    def mock_input(self, monkeypatch):
        """Mock input function to always return 'y'."""
        monkeypatch.setattr('builtins.input', lambda _: 'y')
    
    @pytest.fixture
    def model_path_fixture(self, tmpdir):
        """Create a temporary model path with config.yaml."""
        model_dir = tmpdir.mkdir("model")
        
        # Create a basic config.yaml file
        config_content = {
            "model": {
                "user_id": "test-user",
                "app_id": "test-app",
                "model_id": "test-model",
                "version_id": "1"
            }
        }
        
        with open(f"{model_dir}/config.yaml", "w") as f:
            yaml.dump(config_content, f)
            
        return str(model_dir)
    
    def test_local_dev_all_resources_exist(
        self, cli_runner, mock_user, mock_model_builder, mock_serve, 
        mock_validate_context, mock_code_script, model_path_fixture
    ):
        """Test local_dev function when all resources exist."""
        # Set up the context
        ctx_mock = mock.MagicMock()
        ctx_mock.obj.current.name = "test-context"
        ctx_mock.obj.current.user_id = "test-user"
        ctx_mock.obj.current.pat = "test-pat"
        ctx_mock.obj.current.api_base = "https://api.test.com"
        ctx_mock.obj.current.compute_cluster_id = "test-cluster"
        ctx_mock.obj.current.nodepool_id = "test-nodepool"
        ctx_mock.obj.current.runner_id = "test-runner"
        ctx_mock.obj.current.app_id = "test-app"
        ctx_mock.obj.current.model_id = "test-model"
        
        with mock.patch("click.pass_context", return_value=ctx_mock):
            from clarifai.cli.model import local_dev
            
            # Call the function
            result = cli_runner.invoke(cli, ["model", "local-dev", model_path_fixture])
            
            # Verify interactions
            mock_validate_context.assert_called_once()
            mock_user.assert_called_once()
            mock_user.return_value.compute_cluster.assert_called_once_with("test-cluster")
            mock_code_script.generate_client_script.assert_called_once()
            mock_serve.assert_called_once()
    
    def test_local_dev_no_runner(
        self, cli_runner, mock_user, mock_model_builder, mock_serve, 
        mock_validate_context, mock_code_script, model_path_fixture
    ):
        """Test local_dev function when compute cluster and nodepool exist but runner doesn't."""
        # Set up the context
        ctx_mock = mock.MagicMock()
        ctx_mock.obj.current.name = "test-context"
        ctx_mock.obj.current.user_id = "test-user"
        ctx_mock.obj.current.pat = "test-pat"
        ctx_mock.obj.current.api_base = "https://api.test.com"
        ctx_mock.obj.current.compute_cluster_id = "test-cluster"
        ctx_mock.obj.current.nodepool_id = "test-nodepool"
        ctx_mock.obj.current.app_id = "test-app"
        ctx_mock.obj.current.model_id = "test-model"
        
        # Set up runner not found exception
        mock_nodepool = mock_user.return_value.compute_cluster.return_value.nodepool.return_value
        mock_nodepool.runner.side_effect = AttributeError("Runner not found in nodepool.")
        
        with mock.patch("click.pass_context", return_value=ctx_mock):
            # Call the function
            result = cli_runner.invoke(cli, ["model", "local-dev", model_path_fixture])
            
            # Verify interactions
            mock_validate_context.assert_called_once()
            mock_user.assert_called_once()
            mock_user.return_value.compute_cluster.assert_called_once_with("test-cluster")
            mock_nodepool.create_runner.assert_called_once()
            mock_code_script.generate_client_script.assert_called_once()
            mock_serve.assert_called_once()
    
    def test_local_dev_no_nodepool(
        self, cli_runner, mock_user, mock_model_builder, mock_serve, 
        mock_validate_context, mock_code_script, model_path_fixture, mock_input
    ):
        """Test local_dev function when compute cluster exists but nodepool doesn't."""
        # Set up the context
        ctx_mock = mock.MagicMock()
        ctx_mock.obj.current.name = "test-context"
        ctx_mock.obj.current.user_id = "test-user"
        ctx_mock.obj.current.pat = "test-pat"
        ctx_mock.obj.current.api_base = "https://api.test.com"
        ctx_mock.obj.current.compute_cluster_id = "test-cluster"
        ctx_mock.obj.current.app_id = "test-app"
        ctx_mock.obj.current.model_id = "test-model"
        
        # Set up nodepool not found exception
        mock_compute_cluster = mock_user.return_value.compute_cluster.return_value
        mock_compute_cluster.nodepool.side_effect = Exception("Nodepool not found.")
        
        with mock.patch("click.pass_context", return_value=ctx_mock):
            # Call the function
            result = cli_runner.invoke(cli, ["model", "local-dev", model_path_fixture])
            
            # Verify interactions
            mock_validate_context.assert_called_once()
            mock_user.assert_called_once()
            mock_user.return_value.compute_cluster.assert_called_once_with("test-cluster")
            mock_compute_cluster.create_nodepool.assert_called_once()
            mock_code_script.generate_client_script.assert_called_once()
            mock_serve.assert_called_once()
    
    def test_local_dev_no_compute_cluster(
        self, cli_runner, mock_user, mock_model_builder, mock_serve, 
        mock_validate_context, mock_code_script, model_path_fixture, mock_input
    ):
        """Test local_dev function when compute cluster doesn't exist."""
        # Set up the context
        ctx_mock = mock.MagicMock()
        ctx_mock.obj.current.name = "test-context"
        ctx_mock.obj.current.user_id = "test-user"
        ctx_mock.obj.current.pat = "test-pat"
        ctx_mock.obj.current.api_base = "https://api.test.com"
        ctx_mock.obj.current.app_id = "test-app"
        ctx_mock.obj.current.model_id = "test-model"
        
        # Set up compute cluster not found exception
        mock_user.return_value.compute_cluster.side_effect = Exception("Compute cluster not found.")
        
        with mock.patch("click.pass_context", return_value=ctx_mock):
            # Call the function
            result = cli_runner.invoke(cli, ["model", "local-dev", model_path_fixture])
            
            # Verify interactions
            mock_validate_context.assert_called_once()
            mock_user.assert_called_once()
            mock_user.return_value.compute_cluster.assert_called_once()
            mock_user.return_value.create_compute_cluster.assert_called_once()
            mock_code_script.generate_client_script.assert_called_once()
            mock_serve.assert_called_once()
