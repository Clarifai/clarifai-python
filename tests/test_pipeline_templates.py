"""Tests for pipeline template CLI commands."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from clarifai.cli.pipeline import init
from clarifai.cli.pipeline_template import info, list_templates


class TestPipelineTemplateCLI:
    """Test cases for pipeline template CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_success(self, mock_template_manager_class):
        """Test successful template listing."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock templates data
        mock_templates = [
            {
                'name': 'image-classification',
                'type': 'train',
                'description': 'Image training pipeline',
            },
            {'name': 'text-prep', 'type': 'data', 'description': 'Text preprocessing pipeline'},
        ]
        mock_manager.list_templates.return_value = mock_templates

        # Run command
        result = self.runner.invoke(list_templates)

        # Verify
        assert result.exit_code == 0
        assert 'image-classification' in result.output
        assert 'text-prep' in result.output
        assert 'Found 2 template(s) total' in result.output
        assert 'Available types: data, train' in result.output
        mock_manager.list_templates.assert_called_once_with(None)

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_with_type_filter(self, mock_template_manager_class):
        """Test template listing with type filter."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock filtered templates
        mock_templates = [
            {
                'name': 'image-classification',
                'type': 'train',
                'description': 'Image training pipeline',
            },
        ]
        mock_manager.list_templates.return_value = mock_templates

        # Run command with type filter
        result = self.runner.invoke(list_templates, ['--type', 'train'])

        # Verify
        assert result.exit_code == 0
        assert 'image-classification' in result.output
        assert "Found 1 template(s) of type 'train'" in result.output
        mock_manager.list_templates.assert_called_once_with('train')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_empty_result(self, mock_template_manager_class):
        """Test template listing when no templates found."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.list_templates.return_value = []

        # Run command
        result = self.runner.invoke(list_templates)

        # Verify
        assert result.exit_code == 0
        assert 'No templates found' in result.output
        mock_manager.list_templates.assert_called_once_with(None)

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_with_error(self, mock_template_manager_class):
        """Test template listing error handling."""
        # Mock template manager to raise error
        mock_template_manager_class.side_effect = Exception("Template root not found")

        # Run command
        result = self.runner.invoke(list_templates)

        # Verify error handling
        assert result.exit_code == 0  # CLI handles errors gracefully
        assert 'Error: Could not list templates' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_success(self, mock_template_manager_class):
        """Test successful template info display."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info
        mock_info = {
            'name': 'image-classification',
            'type': 'train',
            'path': '/path/to/template',
            'step_directories': ['LoadDatasetStep', 'TrainModelStep'],
            'parameters': [
                {'description': 'Data Path', 'name': 'DATA_PATH'},
                {'description': 'Batch Size', 'name': 'BATCH_SIZE'},
            ],
            'config': {'pipeline': {'id': 'image-classification'}},
        }
        mock_manager.get_template_info.return_value = mock_info

        # Run command
        result = self.runner.invoke(info, ['image-classification'])

        # Verify
        assert result.exit_code == 0
        assert 'Template: image-classification' in result.output
        assert 'Type: train' in result.output
        assert 'LoadDatasetStep' in result.output
        assert 'TrainModelStep' in result.output
        assert 'DATA_PATH (Data Path)' in result.output
        assert 'BATCH_SIZE (Batch Size)' in result.output
        assert 'Default Pipeline ID: image-classification' in result.output
        mock_manager.get_template_info.assert_called_once_with('image-classification')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_not_found(self, mock_template_manager_class):
        """Test template info when template not found."""
        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        # Run command
        result = self.runner.invoke(info, ['nonexistent-template'])

        # Verify
        assert result.exit_code == 0
        assert "Template 'nonexistent-template' not found" in result.output
        mock_manager.get_template_info.assert_called_once_with('nonexistent-template')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_with_error(self, mock_template_manager_class):
        """Test template info error handling."""
        # Mock template manager to raise error
        mock_template_manager_class.side_effect = Exception("File not found")

        # Run command
        result = self.runner.invoke(info, ['test-template'])

        # Verify error handling
        assert result.exit_code == 0  # CLI handles errors gracefully
        assert 'Error: Could not get template information' in result.output


class TestPipelineInitWithTemplate:
    """Test cases for pipeline init with template option."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('clarifai.cli.pipeline._init_from_template')
    @patch('clarifai.cli.pipeline._init_interactive')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    @patch('clarifai.cli.pipeline._show_completion_message')
    def test_init_with_template_option(
        self, mock_completion, mock_prepare_path, mock_interactive, mock_template
    ):
        """Test that --template option calls template initialization."""
        # Mock the helper functions
        mock_prepare_path.return_value = '/test/path'
        mock_template.return_value = True

        result = self.runner.invoke(init, ['--template', 'image-classification', '.'])

        # Should call template init, not interactive
        mock_prepare_path.assert_called_once_with('.', 'image-classification')
        mock_template.assert_called_once_with('/test/path', 'image-classification')
        mock_interactive.assert_not_called()
        mock_completion.assert_called_once_with('/test/path')

    @patch('clarifai.cli.pipeline._init_from_template')
    @patch('clarifai.cli.pipeline._init_interactive')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    @patch('clarifai.cli.pipeline._show_completion_message')
    def test_init_without_template_option(
        self, mock_completion, mock_prepare_path, mock_interactive, mock_template
    ):
        """Test that without --template option calls interactive initialization."""
        # Mock the helper functions
        mock_prepare_path.return_value = '/test/path'
        mock_interactive.return_value = True

        result = self.runner.invoke(init, ['.'])

        # Should call interactive init, not template
        mock_prepare_path.assert_called_once_with('.', None)
        mock_interactive.assert_called_once_with('/test/path')
        mock_template.assert_not_called()
        mock_completion.assert_called_once_with('/test/path')

    @patch('clarifai.utils.template_manager.TemplateManager')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    def test_init_from_template_success(self, mock_prepare_path, mock_template_manager_class):
        """Test successful template-based initialization."""
        # Mock path preparation
        mock_prepare_path.return_value = '/test/path'

        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        # Mock template info
        mock_info = {
            'name': 'test-template',
            'type': 'train',
            'step_directories': ['StepA', 'StepB'],
            'parameters': [
                {
                    'description': 'Test Param',
                    'name': 'TEST_PARAM',
                    'placeholder': '<TEST_PARAM_VALUE>',
                }
            ],
            'config': {'pipeline': {'id': 'test-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = True

        # Use isolated filesystem for testing
        with self.runner.isolated_filesystem():
            # Import the actual function to test
            from clarifai.cli.pipeline import _init_from_template

            # Mock click.prompt to avoid stdin issues during testing
            with patch('click.prompt') as mock_prompt:
                mock_prompt.side_effect = [
                    'test-user',  # User ID
                    'test-app',  # App ID
                    'my-pipeline',  # Pipeline ID
                    'test-value',  # Test Param value
                ]

                # Test the function directly
                result = _init_from_template('/test/path', 'test-template')

                # Should return True for success
                assert result is True

    @patch('clarifai.utils.template_manager.TemplateManager')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    def test_init_from_template_not_found(self, mock_prepare_path, mock_template_manager_class):
        """Test template initialization when template not found."""
        # Mock path preparation
        mock_prepare_path.return_value = '/test/path'

        # Mock template manager
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        # Use isolated filesystem for testing
        with self.runner.isolated_filesystem():
            # Import the actual function to test
            from clarifai.cli.pipeline import _init_from_template

            # Test the function directly
            result = _init_from_template('/test/path', 'nonexistent')

            # Should return False for failure
            assert result is False


class TestPipelineTemplateIntegration:
    """Integration tests for pipeline template functionality."""

    def test_template_manager_integration(self):
        """Test that TemplateManager integrates correctly with real template repository."""
        from clarifai.utils.template_manager import TemplateManager

        # Create template manager (will use real template path)
        manager = TemplateManager()

        # Should be able to list templates
        templates = manager.list_templates()

        # Should find some templates (based on our test repository)
        assert len(templates) > 0

        # Each template should have required fields
        for template in templates:
            assert 'name' in template
            assert 'type' in template
            assert 'description' in template

    def test_parameter_extraction_integration(self):
        """Test parameter extraction with real template."""
        from clarifai.utils.template_manager import TemplateManager

        manager = TemplateManager()

        # Try to get info for image-classification template
        info = manager.get_template_info('image-classification')

        if info:  # Only test if template exists
            assert 'parameters' in info
            assert len(info['parameters']) > 0

            # Check parameter structure
            for param in info['parameters']:
                assert 'name' in param
                assert 'description' in param
                assert 'placeholder' in param
                assert param['placeholder'].startswith('<')
                assert param['placeholder'].endswith('_VALUE>')

    def test_cli_help_messages(self):
        """Test that CLI help messages are accessible."""
        runner = CliRunner()

        # Test list command help
        result = runner.invoke(list_templates, ['--help'])
        assert result.exit_code == 0
        assert 'List available pipeline templates' in result.output

        # Test info command help
        result = runner.invoke(info, ['--help'])
        assert result.exit_code == 0
        assert 'Show detailed information about a specific template' in result.output
