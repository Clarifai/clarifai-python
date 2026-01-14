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
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        mock_templates = [
            {
                'name': 'image-classification',
                'type': 'train',
                'description': 'Image training pipeline',
            },
            {'name': 'text-prep', 'type': 'data', 'description': 'Text preprocessing pipeline'},
        ]
        mock_manager.list_templates.return_value = mock_templates

        result = self.runner.invoke(list_templates)

        assert result.exit_code == 0
        assert 'image-classification' in result.output
        assert 'text-prep' in result.output
        assert 'Found 2 template(s) total' in result.output
        assert 'Available types: data, train' in result.output
        mock_manager.list_templates.assert_called_once_with(None)

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_with_type_filter(self, mock_template_manager_class):
        """Test template listing with type filter."""
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        mock_templates = [
            {
                'name': 'image-classification',
                'type': 'train',
                'description': 'Image training pipeline',
            },
        ]
        mock_manager.list_templates.return_value = mock_templates

        result = self.runner.invoke(list_templates, ['--type', 'train'])

        assert result.exit_code == 0
        assert 'image-classification' in result.output
        assert "Found 1 template(s) of type 'train'" in result.output
        mock_manager.list_templates.assert_called_once_with('train')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_empty_result(self, mock_template_manager_class):
        """Test template listing when no templates found."""
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.list_templates.return_value = []

        result = self.runner.invoke(list_templates)

        assert result.exit_code == 0
        assert 'No templates found' in result.output
        mock_manager.list_templates.assert_called_once_with(None)

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_list_templates_with_error(self, mock_template_manager_class):
        """Test template listing error handling."""
        mock_template_manager_class.side_effect = Exception("Template root not found")

        result = self.runner.invoke(list_templates)

        assert result.exit_code == 0
        assert 'Error: Could not list templates' in result.output

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_success(self, mock_template_manager_class):
        """Test successful template info display."""
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        mock_info = {
            'name': 'image-classification',
            'type': 'train',
            'path': '/path/to/template',
            'step_directories': ['LoadDatasetStep', 'TrainModelStep'],
            'parameters': [
                {'name': 'DATA_PATH', 'default_value': '/default/path', 'type': 'str'},
                {'name': 'BATCH_SIZE', 'default_value': 32, 'type': 'int'},
            ],
            'config': {'pipeline': {'id': 'image-classification'}},
        }
        mock_manager.get_template_info.return_value = mock_info

        result = self.runner.invoke(info, ['image-classification'])

        assert result.exit_code == 0
        assert 'Template: image-classification' in result.output
        assert 'Type: train' in result.output
        assert 'LoadDatasetStep' in result.output
        assert 'TrainModelStep' in result.output
        assert 'DATA_PATH (default: /default/path)' in result.output
        assert 'BATCH_SIZE (default: 32)' in result.output
        mock_manager.get_template_info.assert_called_once_with('image-classification')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_not_found(self, mock_template_manager_class):
        """Test template info when template not found."""
        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        result = self.runner.invoke(info, ['nonexistent-template'])

        assert result.exit_code == 0
        assert "Template 'nonexistent-template' not found" in result.output
        mock_manager.get_template_info.assert_called_once_with('nonexistent-template')

    @patch('clarifai.cli.pipeline_template.TemplateManager')
    def test_template_info_with_error(self, mock_template_manager_class):
        """Test template info error handling."""
        mock_template_manager_class.side_effect = Exception("File not found")

        result = self.runner.invoke(info, ['test-template'])

        assert result.exit_code == 0
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
        mock_prepare_path.return_value = '/test/path'
        mock_template.return_value = True

        result = self.runner.invoke(init, ['--template', 'image-classification', '.'])

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
        mock_prepare_path.return_value = '/test/path'
        mock_interactive.return_value = True

        result = self.runner.invoke(init, ['.'])

        mock_prepare_path.assert_called_once_with('.', None)
        mock_interactive.assert_called_once_with('/test/path')
        mock_template.assert_not_called()
        mock_completion.assert_called_once_with('/test/path')

    @patch('clarifai.utils.template_manager.TemplateManager')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    def test_init_from_template_success(self, mock_prepare_path, mock_template_manager_class):
        """Test successful template-based initialization."""
        mock_prepare_path.return_value = '/test/path'

        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager

        mock_info = {
            'name': 'test-template',
            'type': 'train',
            'step_directories': ['StepA', 'StepB'],
            'parameters': [
                {
                    'name': 'TEST_PARAM',
                    'default_value': 'default-value',
                    'type': 'str',
                }
            ],
            'config': {'pipeline': {'id': 'test-template'}},
        }
        mock_manager.get_template_info.return_value = mock_info
        mock_manager.copy_template.return_value = True

        with self.runner.isolated_filesystem():
            from clarifai.cli.pipeline import _init_from_template

            with patch('click.prompt') as mock_prompt:
                mock_prompt.side_effect = [
                    'test-user',
                    'test-app',
                    'my-pipeline',
                    'test-value',
                ]

                result = _init_from_template('/test/path', 'test-template')

                assert result is True

    @patch('clarifai.utils.template_manager.TemplateManager')
    @patch('clarifai.cli.pipeline._prepare_pipeline_path')
    def test_init_from_template_not_found(self, mock_prepare_path, mock_template_manager_class):
        """Test template initialization when template not found."""
        mock_prepare_path.return_value = '/test/path'

        mock_manager = Mock()
        mock_template_manager_class.return_value = mock_manager
        mock_manager.get_template_info.return_value = None

        with self.runner.isolated_filesystem():
            from clarifai.cli.pipeline import _init_from_template

            result = _init_from_template('/test/path', 'nonexistent')

            assert result is False


class TestPipelineTemplateIntegration:
    """Integration tests for pipeline template functionality."""

    def test_template_manager_integration(self):
        """Test that TemplateManager integrates correctly with real template repository."""
        from clarifai.utils.template_manager import TemplateManager

        manager = TemplateManager()
        templates = manager.list_templates()

        assert len(templates) > 0

        for template in templates:
            assert 'name' in template
            assert 'type' in template

    def test_parameter_extraction_integration(self):
        """Test parameter extraction with real template."""
        from clarifai.utils.template_manager import TemplateManager

        manager = TemplateManager()
        info = manager.get_template_info('classifier-pipeline-resnet')

        if info:
            assert 'parameters' in info
            assert len(info['parameters']) > 0

            for param in info['parameters']:
                assert 'name' in param
                assert 'default_value' in param
                assert 'type' in param

    def test_cli_help_messages(self):
        """Test that CLI help messages are accessible."""
        runner = CliRunner()

        result = runner.invoke(list_templates, ['--help'])
        assert result.exit_code == 0
        assert 'List available pipeline templates' in result.output

        result = runner.invoke(info, ['--help'])
        assert result.exit_code == 0
        assert 'Show detailed information about a specific template' in result.output
