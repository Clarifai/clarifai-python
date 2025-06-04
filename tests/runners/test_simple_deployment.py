import unittest
from unittest.mock import patch, MagicMock, call


class TestModelDeploymentSetup(unittest.TestCase):
    """Test case for the model deployment setup functionality."""

    @patch('builtins.input')
    def test_deployment_setup_existing_resources(self, mock_input):
        """Test the deployment setup with existing compute cluster and nodepool."""
        # Create all our mocks first
        mock_user = MagicMock()
        mock_compute_cluster = MagicMock()
        mock_nodepool = MagicMock()
        mock_logger = MagicMock()
        
        # Setup the User mock
        mock_compute_cluster_obj = MagicMock()
        mock_compute_cluster_obj.id = "test-cc"
        mock_compute_cluster_obj.description = "Test compute cluster"
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster_obj]
        
        # Setup the ComputeCluster mock
        mock_nodepool_obj = MagicMock()
        mock_nodepool_obj.id = "test-np"
        mock_nodepool_obj.description = "Test nodepool"
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool_obj]
        
        # Setup builder mock
        builder = MagicMock()
        builder.user_id = "test-user"
        builder.app_id = "test-app"
        builder.model_id = "test-model"
        builder._client._session.pat = "test-pat"
        builder._client._session.base = "https://api.clarifai.com"
        
        # Mock input responses
        mock_input.side_effect = ["1", "1", "test-deployment", "n"]
        
        # Create module-level patches
        with patch('clarifai.runners.models.model_builder.User', return_value=mock_user), \
             patch('clarifai.runners.models.model_builder.ComputeCluster', return_value=mock_compute_cluster), \
             patch('clarifai.runners.models.model_builder.Nodepool', return_value=mock_nodepool), \
             patch('clarifai.runners.models.model_builder.logger', mock_logger):
             
            # Import the function here to avoid circular imports in tests
            from clarifai.runners.models.model_builder import setup_deployment_for_model
            
            # Call the function under test
            setup_deployment_for_model(builder)
        
        # Verify the User initialization
        from clarifai.runners.models.model_builder import User
        User.assert_called_once_with(
            user_id="test-user",
            pat="test-pat",
            base_url="https://api.clarifai.com"
        )
        
        # Verify compute cluster selection
        mock_user.list_compute_clusters.assert_called_once()
        
        # Verify nodepool selection
        mock_compute_cluster.list_nodepools.assert_called_once()
        
        # Verify deployment creation
        from clarifai.runners.models.model_builder import Nodepool
        Nodepool.assert_called_once_with(
            nodepool_id=mock_nodepool_obj.id,
            user_id="test-user",
            pat="test-pat",
            base_url="https://api.clarifai.com"
        )
        
        # Verify deployment config
        expected_deployment_config = {
            "deployment": {
                "id": "test-deployment",
                "description": "Deployment for test-model",
                "worker": {
                    "model": {
                        "id": "test-model",
                        "user_id": "test-user",
                        "app_id": "test-app",
                    }
                },
                "nodepools": [
                    {
                        "id": "test-np",
                        "compute_cluster": {
                            "id": "test-cc",
                            "user_id": "test-user"
                        }
                    }
                ]
            }
        }
        mock_nodepool.create_deployment.assert_called_once()
        
        # Verify logging messages
        mock_logger.info.assert_any_call("Checking for available compute clusters...")
        mock_logger.info.assert_any_call("Available compute clusters:")
        mock_logger.info.assert_any_call("Checking for available nodepools in compute cluster 'test-cc'...")
        mock_logger.info.assert_any_call("Available nodepools:")
        mock_logger.info.assert_any_call("Creating deployment 'test-deployment'...")
        mock_logger.info.assert_any_call("Deployment 'test-deployment' created successfully.")