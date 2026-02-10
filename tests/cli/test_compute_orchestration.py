import os
import uuid

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
        result = cli_runner.invoke(
            cli, ["nodepool", "list", "--compute_cluster_id", CREATE_COMPUTE_CLUSTER_ID]
        )
        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    def test_list_deployments(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli, ["deployment", "list", "--nodepool_id", CREATE_NODEPOOL_ID]
        )

        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    def test_list_deployments_with_cluster_id(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli, ["deployment", "list", "--compute_cluster_id", CREATE_COMPUTE_CLUSTER_ID]
        )

        assert result.exit_code == 0, logger.exception(result)
        assert "USER_ID" in result.output

    def test_list_deployments_with_nodepool_and_cluster_id(self, cli_runner):
        cli_runner.invoke(cli, ["login", "--env", CLARIFAI_ENV])
        result = cli_runner.invoke(
            cli,
            [
                "deployment",
                "list",
                "--nodepool_id",
                CREATE_NODEPOOL_ID,
                "--compute_cluster_id",
                CREATE_COMPUTE_CLUSTER_ID,
            ],
        )

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
