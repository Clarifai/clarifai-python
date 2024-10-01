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


@pytest.fixture(scope="class")
def client():
  """Fixture to create a Clarifai user client."""
  return User(user_id=CREATE_COMPUTE_CLUSTER_USER_ID, pat=CLARIFAI_PAT)


@pytest.fixture(scope="class")
def create_compute_cluster(client):
  """Fixture to create and clean up a compute cluster."""
  cluster = ComputeCluster(
      user_id=CREATE_COMPUTE_CLUSTER_USER_ID,
      compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID,
      pat=CLARIFAI_PAT)

  # Set up compute cluster
  with open(COMPUTE_CLUSTER_CONFIG_FILE) as f:
    config = yaml.safe_load(f)
  config["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
  with open(COMPUTE_CLUSTER_CONFIG_FILE, "w") as f:
    yaml.dump(config, f)

  client.create_compute_cluster(
      compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID, config_filepath=COMPUTE_CLUSTER_CONFIG_FILE)

  yield cluster

  # Teardown: Clean up compute cluster after the test
  client.delete_compute_clusters(compute_cluster_ids=[CREATE_COMPUTE_CLUSTER_ID])


@pytest.fixture(scope="class")
def create_nodepool(client):
  """Fixture to create and clean up a nodepool."""
  nodepool = Nodepool(
      user_id=CREATE_COMPUTE_CLUSTER_USER_ID, nodepool_id=CREATE_NODEPOOL_ID, pat=CLARIFAI_PAT)

  # Set up nodepool
  with open(NODEPOOL_CONFIG_FILE) as f:
    config = yaml.safe_load(f)
  config["nodepool"]["id"] = CREATE_NODEPOOL_ID
  with open(NODEPOOL_CONFIG_FILE, "w") as f:
    yaml.dump(config, f)

  client.create_nodepool(nodepool_id=CREATE_NODEPOOL_ID, config_filepath=NODEPOOL_CONFIG_FILE)

  yield nodepool

  # Teardown: Clean up nodepool after the test
  client.delete_nodepools(nodepool_ids=[CREATE_NODEPOOL_ID])


@pytest.fixture(scope="class")
def create_deployment(client, create_compute_cluster, create_nodepool):
  """Fixture to create and clean up a deployment."""
  with open(DEPLOYMENT_CONFIG_FILE) as f:
    config = yaml.safe_load(f)
  config["deployment"]["id"] = CREATE_DEPLOYMENT_ID
  config["deployment"]["nodepools"][0]["id"] = CREATE_NODEPOOL_ID
  config["deployment"]["nodepools"][0]["compute_cluster"]["id"] = CREATE_COMPUTE_CLUSTER_ID
  with open(DEPLOYMENT_CONFIG_FILE, "w") as f:
    yaml.dump(config, f)

  client.create_deployment(
      deployment_id=CREATE_DEPLOYMENT_ID, config_filepath=DEPLOYMENT_CONFIG_FILE)

  yield

  # Teardown: Clean up deployment after the test
  client.delete_deployments(deployment_ids=[CREATE_DEPLOYMENT_ID])


@pytest.mark.requires_secrets
class TestComputeOrchestration:
  """Tests for Compute Orchestration resources."""

  def test_create_compute_cluster(self, caplog, create_compute_cluster):
    """Test to create a compute cluster."""
    with caplog.at_level(logging.INFO):
      assert "Compute Cluster created" in caplog.text

  def test_create_nodepool(self, caplog, create_nodepool):
    """Test to create a nodepool."""
    with caplog.at_level(logging.INFO):
      assert "Nodepool created" in caplog.text

  def test_create_deployment(self, caplog, create_deployment):
    """Test to create a deployment."""
    with caplog.at_level(logging.INFO):
      assert "Deployment created" in caplog.text

  def test_get_compute_cluster(self, client):
    """Test to retrieve a compute cluster."""
    compute_cluster = client.compute_cluster(compute_cluster_id=CREATE_COMPUTE_CLUSTER_ID)
    assert compute_cluster.id == CREATE_COMPUTE_CLUSTER_ID

  def test_get_nodepool(self, client):
    """Test to retrieve a nodepool."""
    nodepool = client.nodepool(nodepool_id=CREATE_NODEPOOL_ID)
    assert nodepool.id == CREATE_NODEPOOL_ID

  def test_get_deployment(self, client):
    """Test to retrieve a deployment."""
    deployment = client.deployment(deployment_id=CREATE_DEPLOYMENT_ID)
    assert deployment.id == CREATE_DEPLOYMENT_ID

  def test_list_compute_clusters(self, client):
    """Test to list all compute clusters."""
    all_compute_clusters = list(client.list_compute_clusters())
    assert len(all_compute_clusters) >= 1

  def test_list_nodepools(self, client):
    """Test to list all nodepools."""
    all_nodepools = list(client.list_nodepools())
    assert len(all_nodepools) >= 1

  def test_list_deployments(self, client):
    """Test to list all deployments."""
    all_deployments = list(client.list_deployments())
    assert len(all_deployments) >= 1
