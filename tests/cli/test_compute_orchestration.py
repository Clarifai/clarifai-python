import os
import uuid

import pytest
import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.client.compute_cluster import ComputeCluster
from clarifai.client.nodepool import Nodepool
from clarifai.client.user import User

COMPUTE_CLUSTER_CONFIG_FILE = "tests/compute_orchestration/configs/example_compute_cluster_config.yaml"
NODEPOOL_CONFIG_FILE = "tests/compute_orchestration/configs/example_nodepool_config.yaml"
DEPLOYMENT_CONFIG_FILE = "tests/compute_orchestration/configs/example_deployment_config.yaml"

CLARIFAI_USER_ID = os.environ["CLARIFAI_USER_ID"]
CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]
CLARIFAI_ENV = os.environ.get("CLARIFAI_ENV", "prod")
CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


@pytest.fixture
def test_cli():
  return CliRunner(env={
      "CLARIFAI_USER_ID": CLARIFAI_USER_ID,
      "CLARIFAI_PAT": CLARIFAI_PAT,
      "CLARIFAI_API_BASE": CLARIFAI_API_BASE
  })


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

    NOW = uuid.uuid4()
    cls.CREATE_COMPUTE_CLUSTER_ID = f"ci_test_cc_{NOW}"
    cls.CREATE_NODEPOOL_ID = f"ci_test_np_{NOW}"
    cls.CREATE_DEPLOYMENT_ID = f"ci_test_dep_{NOW}"

    cls.client = User(user_id=CLARIFAI_USER_ID, pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE)

    cls.compute_cluster = ComputeCluster(
        user_id=CLARIFAI_USER_ID,
        compute_cluster_id=cls.CREATE_COMPUTE_CLUSTER_ID,
        pat=CLARIFAI_PAT,
        base_url=CLARIFAI_API_BASE)

    cls.nodepool = Nodepool(
        user_id=CLARIFAI_USER_ID,
        nodepool_id=cls.CREATE_NODEPOOL_ID,
        pat=CLARIFAI_PAT,
        base_url=CLARIFAI_API_BASE)

    cls._cleanup_resources()

  @classmethod
  def teardown_class(cls):
    """Teardown: Clean up any resources created during the tests."""
    cls._cleanup_resources()

  @classmethod
  def _cleanup_resources(cls):
    """Helper function to delete any existing resources."""
    try:
      cls.nodepool.delete_deployments([cls.CREATE_DEPLOYMENT_ID])
    except Exception:
      pass  # Ignore if not found

    try:
      cls.compute_cluster.delete_nodepools([cls.CREATE_NODEPOOL_ID])
    except Exception:
      pass  # Ignore if not found

    try:
      cls.client.delete_compute_clusters([cls.CREATE_COMPUTE_CLUSTER_ID])
    except Exception:
      pass  # Ignore if not found

  def test_create_compute_cluster(self, test_cli):
    with open(COMPUTE_CLUSTER_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["compute_cluster"]["id"] = self.CREATE_COMPUTE_CLUSTER_ID
    with open(COMPUTE_CLUSTER_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)

    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, [
        "computecluster", "create", "--config", COMPUTE_CLUSTER_CONFIG_FILE,
        "--compute_cluster_id", self.CREATE_COMPUTE_CLUSTER_ID
    ])
    assert result.exit_code == 0

  def test_create_nodepool(self, test_cli):
    with open(NODEPOOL_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["nodepool"]["id"] = self.CREATE_NODEPOOL_ID
    with open(NODEPOOL_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)

    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, [
        "nodepool", "create", "--compute_cluster_id", self.CREATE_COMPUTE_CLUSTER_ID, "--config",
        NODEPOOL_CONFIG_FILE, "--nodepool_id", self.CREATE_NODEPOOL_ID
    ])
    assert result.exit_code == 0

  @pytest.mark.coverage_only
  def test_create_deployment(self, test_cli):
    with open(DEPLOYMENT_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["deployment"]["id"] = self.CREATE_DEPLOYMENT_ID
    config["deployment"]["nodepools"][0]["id"] = self.CREATE_NODEPOOL_ID
    config["deployment"]["nodepools"][0]["compute_cluster"]["id"] = self.CREATE_COMPUTE_CLUSTER_ID
    with open(DEPLOYMENT_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)

    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, [
        "deployment", "create", "--nodepool_id", self.CREATE_NODEPOOL_ID, "--config",
        DEPLOYMENT_CONFIG_FILE, "--deployment_id", self.CREATE_DEPLOYMENT_ID
    ])
    assert result.exit_code == 0

  def test_list_compute_clusters(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, ["computecluster", "list"])
    assert result.exit_code == 0
    assert "List of Compute Clusters" in result.output

  def test_list_nodepools(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(
        cli, ["nodepool", "list", "--compute_cluster_id", self.CREATE_COMPUTE_CLUSTER_ID])
    assert result.exit_code == 0
    assert "List of Nodepools" in result.output

  def test_list_deployments(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, ["deployment", "list", "--nodepool_id", self.CREATE_NODEPOOL_ID])
    assert result.exit_code == 0
    assert "List of Deployments" in result.output

  @pytest.mark.coverage_only
  def test_delete_deployment(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, [
        "deployment", "delete", "--nodepool_id", self.CREATE_NODEPOOL_ID, "--deployment_id",
        self.CREATE_DEPLOYMENT_ID
    ])
    assert result.exit_code == 0

  def test_delete_nodepool(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(cli, [
        "nodepool", "delete", "--compute_cluster_id", self.CREATE_COMPUTE_CLUSTER_ID,
        "--nodepool_id", self.CREATE_NODEPOOL_ID
    ])
    assert result.exit_code == 0

  def test_delete_compute_cluster(self, test_cli):
    test_cli.invoke(cli, ["login", "--env", CLARIFAI_ENV])
    result = test_cli.invoke(
        cli, ["computecluster", "delete", "--compute_cluster_id", self.CREATE_COMPUTE_CLUSTER_ID])
    assert result.exit_code == 0
