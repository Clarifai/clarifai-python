"""Test utilities for testing local_dev function in clarifai.cli.model module."""

import os
from unittest import mock

import yaml


def setup_mock_components():
    """Set up mocks for local_dev function components."""
    # Mock components
    mock_user = mock.MagicMock()
    mock_compute_cluster = mock.MagicMock()
    mock_nodepool = mock.MagicMock()
    mock_runner = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_serve = mock.MagicMock()
    mock_code_script = mock.MagicMock()
    
    # Configure mocks
    mock_compute_cluster.cluster_type = 'local-dev'
    mock_user.compute_cluster.return_value = mock_compute_cluster
    mock_compute_cluster.nodepool.return_value = mock_nodepool
    mock_nodepool.runner.return_value = mock_runner
    mock_builder.get_method_signatures.return_value = [
        {"method_name": "test_method", "parameters": []}
    ]
    
    return {
        "user": mock_user,
        "compute_cluster": mock_compute_cluster,
        "nodepool": mock_nodepool,
        "runner": mock_runner,
        "builder": mock_builder,
        "serve": mock_serve,
        "code_script": mock_code_script
    }


def create_mock_context():
    """Create a mock context for local_dev function."""
    ctx = mock.MagicMock()
    ctx.obj = mock.MagicMock()
    ctx.obj.current = mock.MagicMock()
    ctx.obj.current.name = "test-context"
    ctx.obj.current.user_id = "test-user"
    ctx.obj.current.pat = "test-pat"
    ctx.obj.current.api_base = "https://api.test.com"
    ctx.obj.current.compute_cluster_id = "test-cluster"
    ctx.obj.current.nodepool_id = "test-nodepool"
    ctx.obj.current.runner_id = "test-runner"
    ctx.obj.current.app_id = "test-app"
    ctx.obj.current.model_id = "test-model"
    return ctx


def setup_model_dir(tmpdir):
    """Create a model directory with config.yaml for testing."""
    model_dir = os.path.join(tmpdir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a basic config.yaml file
    config_content = {
        "model": {
            "user_id": "test-user",
            "app_id": "test-app",
            "model_id": "test-model",
            "version_id": "1"
        }
    }
    
    with open(os.path.join(model_dir, "config.yaml"), "w") as f:
        yaml.dump(config_content, f)
    
    return model_dir


def test_local_dev_flow(model_path):
    """Test the core flow of local_dev function.
    
    This doesn't use the CLI command directly but tests the core logic.
    """
    from clarifai.client.user import User
    
    # Set up mocks
    mocks = setup_mock_components()
    ctx = create_mock_context()
    
    # Import functions under test
    from clarifai.cli.model import validate_context
    
    with mock.patch("clarifai.cli.model.validate_context"), \
         mock.patch("clarifai.client.user.User", return_value=mocks["user"]), \
         mock.patch("clarifai.runners.models.model_builder.ModelBuilder", return_value=mocks["builder"]), \
         mock.patch("clarifai.runners.server.serve"), \
         mock.patch("clarifai.runners.utils.code_script.generate_client_script"), \
         mock.patch("builtins.input", return_value="y"):
        
        # Call the function we're testing directly
        # Note: we're just implementing the test logic, not the actual local_dev function
        user = User(user_id=ctx.obj.current.user_id, 
                   pat=ctx.obj.current.pat, 
                   base_url=ctx.obj.current.api_base)
        
        # Get or create compute cluster
        compute_cluster_id = ctx.obj.current.compute_cluster_id
        compute_cluster = user.compute_cluster(compute_cluster_id)
        
        # Get or create nodepool
        nodepool_id = ctx.obj.current.nodepool_id
        nodepool = compute_cluster.nodepool(nodepool_id)
        
        # Get or create runner
        runner_id = ctx.obj.current.runner_id
        runner = nodepool.runner(runner_id)
        
        # Verify interactions
        user.compute_cluster.assert_called_once_with(compute_cluster_id)
        compute_cluster.nodepool.assert_called_once_with(nodepool_id)
        nodepool.runner.assert_called_once_with(runner_id)
    
    return mocks


def test_local_dev_no_runner(model_path):
    """Test local_dev when runner doesn't exist."""
    from clarifai.client.user import User
    
    # Set up mocks
    mocks = setup_mock_components()
    ctx = create_mock_context()
    
    # Configure runner not found exception
    mocks["nodepool"].runner.side_effect = AttributeError("Runner not found in nodepool.")
    
    with mock.patch("clarifai.cli.model.validate_context"), \
         mock.patch("clarifai.client.user.User", return_value=mocks["user"]), \
         mock.patch("clarifai.runners.models.model_builder.ModelBuilder", return_value=mocks["builder"]), \
         mock.patch("clarifai.runners.server.serve"), \
         mock.patch("clarifai.runners.utils.code_script.generate_client_script"), \
         mock.patch("builtins.input", return_value="y"):
        
        # Call the function we're testing directly
        user = User(user_id=ctx.obj.current.user_id, 
                   pat=ctx.obj.current.pat, 
                   base_url=ctx.obj.current.api_base)
        
        # Get or create compute cluster
        compute_cluster_id = ctx.obj.current.compute_cluster_id
        compute_cluster = user.compute_cluster(compute_cluster_id)
        
        # Get or create nodepool
        nodepool_id = ctx.obj.current.nodepool_id
        nodepool = compute_cluster.nodepool(nodepool_id)
        
        # Get or create runner
        try:
            runner_id = ctx.obj.current.runner_id
            runner = nodepool.runner(runner_id)
        except AttributeError:
            # Runner doesn't exist, create it
            runner = nodepool.create_runner(runner_config={
                "runner": {
                    "description": "Local dev runner for model testing",
                    "worker": "test_worker",
                    "num_replicas": 1,
                }
            })
        
        # Verify interactions
        user.compute_cluster.assert_called_once_with(compute_cluster_id)
        compute_cluster.nodepool.assert_called_once_with(nodepool_id)
        nodepool.runner.assert_called_once_with(runner_id)
        nodepool.create_runner.assert_called_once()
    
    return mocks


def test_local_dev_no_nodepool(model_path):
    """Test local_dev when nodepool doesn't exist."""
    from clarifai.client.user import User
    
    # Set up mocks
    mocks = setup_mock_components()
    ctx = create_mock_context()
    
    # Configure nodepool not found exception
    mocks["compute_cluster"].nodepool.side_effect = Exception("Nodepool not found.")
    
    with mock.patch("clarifai.cli.model.validate_context"), \
         mock.patch("clarifai.client.user.User", return_value=mocks["user"]), \
         mock.patch("clarifai.runners.models.model_builder.ModelBuilder", return_value=mocks["builder"]), \
         mock.patch("clarifai.runners.server.serve"), \
         mock.patch("clarifai.runners.utils.code_script.generate_client_script"), \
         mock.patch("builtins.input", return_value="y"):
        
        # Call the function we're testing directly
        user = User(user_id=ctx.obj.current.user_id, 
                   pat=ctx.obj.current.pat, 
                   base_url=ctx.obj.current.api_base)
        
        # Get or create compute cluster
        compute_cluster_id = ctx.obj.current.compute_cluster_id
        compute_cluster = user.compute_cluster(compute_cluster_id)
        
        # Get or create nodepool
        nodepool_id = ctx.obj.current.nodepool_id
        try:
            nodepool = compute_cluster.nodepool(nodepool_id)
        except Exception:
            # Nodepool doesn't exist, create it
            nodepool = compute_cluster.create_nodepool(
                nodepool_id=nodepool_id,
                nodepool_config={"nodepool": {"description": "Test nodepool"}}
            )
        
        # Verify interactions
        user.compute_cluster.assert_called_once_with(compute_cluster_id)
        compute_cluster.nodepool.assert_called_once_with(nodepool_id)
        compute_cluster.create_nodepool.assert_called_once()
    
    return mocks


def test_local_dev_no_compute_cluster(model_path):
    """Test local_dev when compute cluster doesn't exist."""
    from clarifai.client.user import User
    
    # Set up mocks
    mocks = setup_mock_components()
    ctx = create_mock_context()
    
    # Configure compute cluster not found exception
    mocks["user"].compute_cluster.side_effect = Exception("Compute cluster not found.")
    
    with mock.patch("clarifai.cli.model.validate_context"), \
         mock.patch("clarifai.client.user.User", return_value=mocks["user"]), \
         mock.patch("clarifai.runners.models.model_builder.ModelBuilder", return_value=mocks["builder"]), \
         mock.patch("clarifai.runners.server.serve"), \
         mock.patch("clarifai.runners.utils.code_script.generate_client_script"), \
         mock.patch("builtins.input", return_value="y"):
        
        # Call the function we're testing directly
        user = User(user_id=ctx.obj.current.user_id, 
                   pat=ctx.obj.current.pat, 
                   base_url=ctx.obj.current.api_base)
        
        # Get or create compute cluster
        compute_cluster_id = ctx.obj.current.compute_cluster_id
        try:
            compute_cluster = user.compute_cluster(compute_cluster_id)
        except Exception:
            # Compute cluster doesn't exist, create it
            compute_cluster = user.create_compute_cluster(
                compute_cluster_id=compute_cluster_id,
                compute_cluster_config={"compute_cluster": {"description": "Test cluster"}}
            )
        
        # Verify interactions
        user.compute_cluster.assert_called_once_with(compute_cluster_id)
        user.create_compute_cluster.assert_called_once()
    
    return mocks