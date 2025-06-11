"""Tests for model deployment setup functionality."""

import os
from unittest.mock import MagicMock, patch, Mock
import pytest

from clarifai.runners.models.model_builder import setup_deployment_for_model, upload_model, ModelBuilder


class TestModelDeploymentSetup:
    """Test cases for model deployment setup functionality."""

    @pytest.fixture
    def mock_builder(self):
        """Create a mock ModelBuilder instance."""
        builder = Mock(spec=ModelBuilder)
        builder.config = {
            'model': {
                'user_id': 'test_user',
                'app_id': 'test_app',
                'id': 'test_model'
            }
        }
        builder.client = Mock()
        builder.client.pat = 'test_pat'
        builder.client.base = 'https://api.clarifai.com'
        return builder

    @pytest.fixture
    def mock_compute_cluster(self):
        """Create a mock compute cluster."""
        cluster = Mock()
        cluster.id = 'test-cluster'
        cluster.description = 'Test compute cluster'
        return cluster

    @pytest.fixture
    def mock_nodepool(self):
        """Create a mock nodepool."""
        nodepool = Mock()
        nodepool.id = 'test-nodepool'
        nodepool.description = 'Test nodepool'
        return nodepool

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_with_existing_resources(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup when compute clusters and nodepools already exist."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: choose existing cluster (1), choose existing nodepool (1), don't open browser
        mock_input.side_effect = ['1', '1', 'n']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify User was created correctly
        mock_user_class.assert_called_once_with(
            user_id='test_user', 
            pat='test_pat', 
            base_url='https://api.clarifai.com'
        )
        
        # Verify compute clusters were listed
        mock_user.list_compute_clusters.assert_called_once()
        
        # Verify nodepools were listed for the selected cluster
        mock_compute_cluster.list_nodepools.assert_called_once()
        
        # Verify browser was not opened (user chose 'n')
        mock_webbrowser.open.assert_not_called()

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_create_new_compute_cluster_choice(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup when user chooses to create a new compute cluster."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        # First call returns existing cluster, but user chooses to create new one
        mock_user.list_compute_clusters.side_effect = [[mock_compute_cluster], [mock_compute_cluster]]
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: create new cluster ('n'), open browser ('y'), press enter, choose cluster (1), choose nodepool (1), don't open deployment browser ('n')
        mock_input.side_effect = ['n', 'y', '', '1', '1', 'n']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify compute cluster creation URL was opened
        expected_cluster_url = 'https://clarifai.com/settings/compute/new'
        mock_webbrowser.open.assert_called_with(expected_cluster_url)
        
        # Verify compute clusters were listed twice (before and after creation)
        assert mock_user.list_compute_clusters.call_count == 2

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_no_compute_clusters_initially(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup when no compute clusters exist initially."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        # First call returns empty list, second call (after user creates) returns cluster
        mock_user.list_compute_clusters.side_effect = [[], [mock_compute_cluster]]
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: open browser ('y'), press enter, choose cluster (1), choose nodepool (1), don't open deployment browser ('n')
        mock_input.side_effect = ['y', '', '1', '1', 'n']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify compute cluster creation URL was opened
        expected_cluster_url = 'https://clarifai.com/settings/compute/new'
        mock_webbrowser.open.assert_called_with(expected_cluster_url)
        
        # Verify compute clusters were listed twice (before and after creation)
        assert mock_user.list_compute_clusters.call_count == 2

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_create_new_nodepool_choice(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup when user chooses to create a new nodepool."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        # First call returns existing nodepool, but user chooses to create new one
        mock_compute_cluster.list_nodepools.side_effect = [[mock_nodepool], [mock_nodepool]]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: choose cluster (1), create new nodepool ('n'), open browser ('y'), press enter, choose nodepool (1), don't open deployment browser ('n')
        mock_input.side_effect = ['1', 'n', 'y', '', '1', 'n']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify nodepool creation URL was opened
        expected_nodepool_url = 'https://clarifai.com/settings/compute/test-cluster/nodepools/new'
        mock_webbrowser.open.assert_called_with(expected_nodepool_url)
        
        # Verify nodepools were listed twice (before and after creation)
        assert mock_compute_cluster.list_nodepools.call_count == 2

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_no_nodepools_initially(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup when no nodepools exist initially."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        # First call returns empty list, second call (after user creates) returns nodepool
        mock_compute_cluster.list_nodepools.side_effect = [[], [mock_nodepool]]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: choose cluster (1), open browser ('y'), press enter, choose nodepool (1), don't open deployment browser ('n')
        mock_input.side_effect = ['1', 'y', '', '1', 'n']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify nodepool creation URL was opened
        expected_nodepool_url = 'https://clarifai.com/settings/compute/test-cluster/nodepools/new'
        mock_webbrowser.open.assert_called_with(expected_nodepool_url)
        
        # Verify nodepools were listed twice (before and after creation)
        assert mock_compute_cluster.list_nodepools.call_count == 2

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_open_deployment_url(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup opens deployment creation URL in browser."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: choose cluster (1), choose nodepool (1), open deployment browser ('y')
        mock_input.side_effect = ['1', '1', 'y']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify deployment creation URL was opened
        expected_deployment_url = 'https://clarifai.com/settings/compute/deployments/new?computeClusterId=test-cluster&nodePoolId=test-nodepool'
        mock_webbrowser.open.assert_called_with(expected_deployment_url)

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_no_compute_clusters_after_creation(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, mock_builder
    ):
        """Test deployment setup handles case where no compute clusters exist after creation attempt."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        # Both calls return empty list (user didn't create cluster)
        mock_user.list_compute_clusters.return_value = []
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: don't open browser ('n'), press enter
        mock_input.side_effect = ['n', '']
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify function returns early and doesn't proceed to nodepool selection
        # We know this because webbrowser.open is never called for deployment URL
        calls = mock_webbrowser.open.call_args_list
        deployment_url_calls = [call for call in calls if 'deployments/new' in str(call)]
        assert len(deployment_url_calls) == 0

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    @patch('builtins.input')
    def test_deployment_setup_invalid_cluster_choice(
        self, mock_input, mock_webbrowser, mock_url_helper, mock_user_class, 
        mock_builder, mock_compute_cluster
    ):
        """Test deployment setup handles invalid cluster choice."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        
        mock_url_helper_instance = Mock()
        mock_url_helper.return_value = mock_url_helper_instance
        mock_url_helper_instance.ui = 'https://clarifai.com'
        
        # Mock user inputs: invalid choice ('invalid'), then don't open browser ('n'), press enter, then choose cluster (1), choose nodepool (1), don't open deployment browser ('n')
        mock_input.side_effect = ['invalid', 'n', '', '1', '1', 'n']
        
        # Mock compute cluster to have nodepools
        mock_nodepool = Mock()
        mock_nodepool.id = 'test-nodepool'
        mock_nodepool.description = 'Test nodepool'
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        # Execute
        setup_deployment_for_model(mock_builder)
        
        # Verify function handles invalid input gracefully but still continues with deployment setup
        # Since user chose 'n' for opening deployment browser, verify that the deployment URL was not opened in browser
        calls = mock_webbrowser.open.call_args_list
        deployment_url_calls = [call for call in calls if 'deployments/new' in str(call)]
        # No deployment URL should be opened since user chose 'n'
        assert len(deployment_url_calls) == 0
        
        # But verify that compute cluster URL was still opened when creating new cluster
        cluster_url_calls = [call for call in calls if 'settings/compute/new' in str(call)]
        assert len(cluster_url_calls) == 0  # User chose 'n' for opening browser for cluster creation too

    @patch('clarifai.runners.models.model_builder.User')
    @patch('clarifai.runners.models.model_builder.webbrowser')
    def test_deployment_setup_webbrowser_exception(
        self, mock_webbrowser, mock_user_class, mock_builder, mock_compute_cluster, mock_nodepool
    ):
        """Test deployment setup handles webbrowser exceptions gracefully."""
        # Setup mocks
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_user.list_compute_clusters.return_value = [mock_compute_cluster]
        mock_compute_cluster.list_nodepools.return_value = [mock_nodepool]
        
        # Make webbrowser.open raise an exception
        mock_webbrowser.open.side_effect = Exception("Browser not available")
        
        with patch('builtins.input') as mock_input:
            with patch('clarifai.runners.models.model_builder.ClarifaiUrlHelper') as mock_url_helper:
                mock_url_helper_instance = Mock()
                mock_url_helper.return_value = mock_url_helper_instance
                mock_url_helper_instance.ui = 'https://clarifai.com'
                
                # Mock user inputs: choose cluster (1), choose nodepool (1), open deployment browser ('y')
                mock_input.side_effect = ['1', '1', 'y']
                
                # Execute - should not raise exception
                setup_deployment_for_model(mock_builder)
                
                # Verify webbrowser.open was called despite the exception
                mock_webbrowser.open.assert_called()


class TestUploadModelWithDeployment:
    """Test cases for model upload with deployment setup integration."""

    @patch('clarifai.runners.models.model_builder.setup_deployment_for_model')
    @patch('clarifai.runners.models.model_builder.ModelBuilder')
    @patch('builtins.input')
    def test_upload_model_with_deployment_yes(self, mock_input, mock_builder_class, mock_setup_deployment):
        """Test upload_model calls deployment setup when user chooses yes."""
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.create_dockerfile.return_value = None
        mock_builder.check_model_exists.return_value = False
        mock_builder.model_ui_url = 'https://clarifai.com/test/model'
        mock_builder.upload_model_version.return_value = Mock()
        
        # Mock user inputs: press enter to continue, yes to deploy
        mock_input.side_effect = ['', 'y']
        
        # Execute
        upload_model('test_folder', 'upload', False)
        
        # Verify deployment setup was called
        mock_setup_deployment.assert_called_once_with(mock_builder)
        
        # Verify builder methods were called
        mock_builder.create_dockerfile.assert_called_once()
        mock_builder.check_model_exists.assert_called_once()
        mock_builder.upload_model_version.assert_called_once()

    @patch('clarifai.runners.models.model_builder.setup_deployment_for_model')
    @patch('clarifai.runners.models.model_builder.ModelBuilder')
    @patch('builtins.input')
    def test_upload_model_with_deployment_no(self, mock_input, mock_builder_class, mock_setup_deployment):
        """Test upload_model skips deployment setup when user chooses no."""
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.create_dockerfile.return_value = None
        mock_builder.check_model_exists.return_value = False
        mock_builder.model_ui_url = 'https://clarifai.com/test/model'
        mock_builder.upload_model_version.return_value = Mock()
        
        # Mock user inputs: press enter to continue, no to deploy
        mock_input.side_effect = ['', 'n']
        
        # Execute
        upload_model('test_folder', 'upload', False)
        
        # Verify deployment setup was NOT called
        mock_setup_deployment.assert_not_called()
        
        # Verify builder methods were still called
        mock_builder.create_dockerfile.assert_called_once()
        mock_builder.check_model_exists.assert_called_once()
        mock_builder.upload_model_version.assert_called_once()

    @patch('clarifai.runners.models.model_builder.setup_deployment_for_model')
    @patch('clarifai.runners.models.model_builder.ModelBuilder')
    @patch('builtins.input')
    def test_upload_model_skip_dockerfile(self, mock_input, mock_builder_class, mock_setup_deployment):
        """Test upload_model with skip_dockerfile=True."""
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.create_dockerfile.return_value = None
        mock_builder.check_model_exists.return_value = False
        mock_builder.model_ui_url = 'https://clarifai.com/test/model'
        mock_builder.upload_model_version.return_value = Mock()
        
        # Mock user inputs: press enter to continue, no to deploy
        mock_input.side_effect = ['', 'n']
        
        # Execute with skip_dockerfile=True
        upload_model('test_folder', 'upload', True)
        
        # Verify dockerfile creation was skipped
        mock_builder.create_dockerfile.assert_not_called()
        
        # Verify other methods were still called
        mock_builder.check_model_exists.assert_called_once()
        mock_builder.upload_model_version.assert_called_once()
        
        # Verify deployment setup was NOT called
        mock_setup_deployment.assert_not_called()