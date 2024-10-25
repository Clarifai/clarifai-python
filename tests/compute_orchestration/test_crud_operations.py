import logging
import os
import uuid
import pytest
import yaml

from clarifai.client.compute_cluster import ComputeCluster
from clarifai.client.nodepool import Nodepool
from clarifai.client.user import User

NOW = uuid.uuid4().hex[:10]

CREATE_COMPUTE_CLUSTER_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_COMPUTE_CLUSTER_ID = f"ci_test_compute_cluster_{NOW}"
CREATE_NODEPOOL_ID = f"ci_test_nodepool_{NOW}"
CREATE_DEPLOYMENT_ID = f"ci_test_deployment_{NOW}"

COMPUTE_CLUSTER_CONFIG_FILE = "tests/compute_orchestration/configs/example_compute_cluster_config.yaml"
NODEPOOL_CONFIG_FILE = "tests/compute_orchestration/configs/example_nodepool_config.yaml"
DEPLOYMENT_CONFIG_FILE = "tests/compute_orchestration/configs/example_deployment_config.yaml"

CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def client():
  return User(user_id=CREATE_COMPUTE_CLUSTER_USER_ID, pat=CLARIFAI_PAT)


@pytest.fixture
def create_compute_cluster():
  return ComputeCluster(
      user_id=CREATE_COMPUTE_CLUSTER_USER_ID,
      compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID,
      pat=CLARIFAI_PAT)


@pytest.fixture
def create_nodepool():
  return Nodepool(
      user_id=CREATE_COMPUTE_CLUSTER_USER_ID, nodepool_id=CREATE_NODEPOOL_ID, pat=CLARIFAI_PAT)


@pytest.mark.requires_secrets
class TestComputeOrchestration:
  """Tests for the Compute Orchestration resources.
    CRUD operations are tested for each of the following resources:
    - compute cluster
    - nodepool
    - deployment
    """

  @classmethod
  def setup_class(cls):
    """Setup: Clean up any pre-existing resources before tests."""
    cls.client = User(user_id=CREATE_COMPUTE_CLUSTER_USER_ID, pat=CLARIFAI_PAT)
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

  def test_create_compute_cluster(self, client, caplog):
    with open(COMPUTE_CLUSTER_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
    with open(COMPUTE_CLUSTER_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)
    with caplog.at_level(logging.INFO):
      client.create_compute_cluster(
          compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID,
          config_filepath=COMPUTE_CLUSTER_CONFIG_FILE)
      assert "Compute Cluster created" in caplog.text

  def test_create_nodepool(self, create_compute_cluster, caplog):
    with open(NODEPOOL_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["nodepool"]["id"] = CREATE_NODEPOOL_ID
    with open(NODEPOOL_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)
    with caplog.at_level(logging.INFO):
      create_compute_cluster.create_nodepool(
          nodepool_id=CREATE_NODEPOOL_ID, config_filepath=NODEPOOL_CONFIG_FILE)
      assert "Nodepool created" in caplog.text

  def test_create_deployment(self, create_nodepool, caplog):
    with open(DEPLOYMENT_CONFIG_FILE) as f:
      config = yaml.safe_load(f)
    config["deployment"]["id"] = CREATE_DEPLOYMENT_ID
    config["deployment"]["nodepools"][0]["id"] = CREATE_NODEPOOL_ID
    config["deployment"]["nodepools"][0]["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
    with open(DEPLOYMENT_CONFIG_FILE, "w") as f:
      yaml.dump(config, f)
    with caplog.at_level(logging.INFO):
      create_nodepool.create_deployment(
          deployment_id=CREATE_DEPLOYMENT_ID, config_filepath=DEPLOYMENT_CONFIG_FILE)
      assert "Deployment created" in caplog.text

  def test_get_compute_cluster(self, client):
    compute_cluster = client.compute_cluster(compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID)
    assert compute_cluster.id == CREATE_COMPUTE_CLUSTER_ID and compute_cluster.user_id == CREATE_COMPUTE_CLUSTER_USER_ID

  def test_get_nodepool(self, create_compute_cluster):
    nodepool = create_compute_cluster.nodepool(nodepool_id=CREATE_NODEPOOL_ID)
    assert nodepool.id == CREATE_NODEPOOL_ID and nodepool.compute_cluster.id == CREATE_COMPUTE_CLUSTER_ID and nodepool.compute_cluster.user_id == CREATE_COMPUTE_CLUSTER_USER_ID

  def test_get_deployment(self, create_nodepool):
    deployment = create_nodepool.deployment(deployment_id=CREATE_DEPLOYMENT_ID)
    assert deployment.id == CREATE_DEPLOYMENT_ID and deployment.nodepools[0].id == CREATE_NODEPOOL_ID and deployment.nodepools[0].compute_cluster.id == CREATE_COMPUTE_CLUSTER_ID and deployment.user_id == CREATE_COMPUTE_CLUSTER_USER_ID

  def test_list_compute_clusters(self, client):
    all_compute_clusters = list(client.list_compute_clusters())
    assert len(all_compute_clusters) >= 1

  def test_list_nodepools(self, create_compute_cluster):
    all_nodepools = list(create_compute_cluster.list_nodepools())
    assert len(all_nodepools) >= 1

  def test_list_deployments(self, create_nodepool):
    all_deployments = list(create_nodepool.list_deployments())
    assert len(all_deployments) >= 1

  def test_delete_deployment(self, create_nodepool, caplog):
    with caplog.at_level(logging.INFO):
      create_nodepool.delete_deployments(deployment_ids=[CREATE_DEPLOYMENT_ID])
      assert "SUCCESS" in caplog.text

  def test_delete_nodepool(self, create_compute_cluster, caplog):
    with caplog.at_level(logging.INFO):
      create_compute_cluster.delete_nodepools(nodepool_ids=[CREATE_NODEPOOL_ID])
      assert "SUCCESS" in caplog.text

  def test_delete_compute_cluster(self, client, caplog):
    with caplog.at_level(logging.INFO):
      client.delete_compute_clusters(compute_cluster_ids=[CREATE_COMPUTE_CLUSTER_ID])
      assert "SUCCESS" in caplog.text
