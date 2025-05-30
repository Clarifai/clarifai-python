import unittest
from unittest.mock import patch, MagicMock


class TestModelDeploymentSetup(unittest.TestCase):
    """Test case for the model deployment setup functionality."""

    @patch('clarifai.client.user.User')
    @patch('clarifai.client.compute_cluster.ComputeCluster')
    @patch('clarifai.client.nodepool.Nodepool')
    @patch('builtins.input')
    @patch('clarifai.utils.logging.logger')
    def test_deployment_setup_existing_resources(self, mock_logger, mock_input, 
                                               mock_nodepool, mock_compute_cluster, 
                                               mock_user):
        """Test the deployment setup with existing compute cluster and nodepool."""
        # Import the function here to avoid circular imports in tests
        from clarifai.runners.models.model_builder import setup_deployment_for_model
        
        # Setup mocks
        builder = MagicMock()
        builder.user_id = "test-user"
        builder.app_id = "test-app"
        builder.model_id = "test-model"
        builder._client._session.pat = "test-pat"
        builder._client._session.base = "https://api.clarifai.com"
        
        # Mock User
        mock_user_instance = mock_user.return_value
        mock_compute_cluster_obj = MagicMock()
        mock_compute_cluster_obj.id = "test-cc"
        mock_compute_cluster_obj.description = "Test compute cluster"
        mock_user_instance.list_compute_clusters.return_value = [mock_compute_cluster_obj]
        
        # Mock ComputeCluster
        mock_compute_cluster_instance = mock_compute_cluster.return_value
        mock_nodepool_obj = MagicMock()
        mock_nodepool_obj.id = "test-np"
        mock_nodepool_obj.description = "Test nodepool"
        mock_compute_cluster_instance.list_nodepools.return_value = [mock_nodepool_obj]
        
        # Mock Nodepool
        mock_nodepool_instance = mock_nodepool.return_value
        
        # Mock input responses
        mock_input.side_effect = ["1", "1", "test-deployment", "n"]
        
        # Call the function
        setup_deployment_for_model(builder)
        
        # Verify user was initialized correctly
        mock_user.assert_called_once_with(
            user_id="test-user",
            pat="test-pat",
            base_url="https://api.clarifai.com"
        )
        
        # Verify compute cluster selection
        mock_user_instance.list_compute_clusters.assert_called_once()
        mock_compute_cluster.assert_called_once_with(
            compute_cluster_id=mock_compute_cluster_obj.id,
            user_id="test-user",
            pat="test-pat",
            base_url="https://api.clarifai.com"
        )
        
        # Verify nodepool selection
        mock_compute_cluster_instance.list_nodepools.assert_called_once()
        
        # Verify deployment creation
        mock_nodepool.assert_called_once_with(
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
        mock_nodepool_instance.create_deployment.assert_called_once_with(expected_deployment_config)
        
        # Verify logging messages
        mock_logger.info.assert_any_call("Checking for available compute clusters...")
        mock_logger.info.assert_any_call("Available compute clusters:")
        mock_logger.info.assert_any_call("Checking for available nodepools in compute cluster 'test-cc'...")
        mock_logger.info.assert_any_call("Available nodepools:")
        mock_logger.info.assert_any_call("Creating deployment 'test-deployment'...")
        mock_logger.info.assert_any_call("Deployment 'test-deployment' created successfully.")

    @patch('clarifai.client.user.User')
    @patch('builtins.input')
    @patch('clarifai.utils.logging.logger')
    def test_deployment_setup_create_new_compute_cluster(self, mock_logger, mock_input, mock_user):
        """Test the deployment setup with creating a new compute cluster."""
        # Import the function here to avoid circular imports in tests
        from clarifai.runners.models.model_builder import setup_deployment_for_model
        
        # Setup mocks
        builder = MagicMock()
        builder.user_id = "test-user"
        builder.app_id = "test-app"
        builder.model_id = "test-model"
        builder._client._session.pat = "test-pat"
        builder._client._session.base = "https://api.clarifai.com"
        
        # Mock User
        mock_user_instance = mock_user.return_value
        # No existing compute clusters
        mock_user_instance.list_compute_clusters.return_value = []
        
        # Mock creating compute cluster
        mock_cc = MagicMock()
        mock_cc.id = "new-cc"
        mock_user_instance.create_compute_cluster.return_value = mock_cc
        
        # Abort after creating compute cluster to simplify test
        mock_input.side_effect = ["new-cc"]
        
        # Abort after creating the compute cluster
        mock_cc.list_nodepools = MagicMock(side_effect=Exception("Test exception"))
        
        # This should abort after creating the compute cluster due to the exception
        with self.assertRaises(Exception):
            setup_deployment_for_model(builder)
        
        # Verify user was initialized correctly
        mock_user.assert_called_once_with(
            user_id="test-user",
            pat="test-pat",
            base_url="https://api.clarifai.com"
        )
        
        # Verify compute cluster creation
        mock_user_instance.list_compute_clusters.assert_called_once()
        mock_user_instance.create_compute_cluster.assert_called_once()
        
        # Verify expected config for compute cluster creation
        expected_cc_config = {
            "compute_cluster": {
                "id": "new-cc",
                "cluster_type": "users",
                "description": "Compute cluster for test-model",
            }
        }
        # Check that the config matches (ignoring the specific object instance)
        actual_config = mock_user_instance.create_compute_cluster.call_args[0][0]
        self.assertEqual(actual_config["compute_cluster"]["id"], expected_cc_config["compute_cluster"]["id"])
        self.assertEqual(actual_config["compute_cluster"]["cluster_type"], expected_cc_config["compute_cluster"]["cluster_type"])
        self.assertEqual(actual_config["compute_cluster"]["description"], expected_cc_config["compute_cluster"]["description"])
        
        # Verify logging messages
        mock_logger.info.assert_any_call("Checking for available compute clusters...")
        mock_logger.info.assert_any_call("No compute clusters found.")
        mock_logger.info.assert_any_call("Creating new compute cluster 'new-cc'...")
        mock_logger.info.assert_any_call("Compute cluster 'new-cc' created successfully.")