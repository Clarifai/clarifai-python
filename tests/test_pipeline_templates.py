"""Tests for pipeline template CLI commands."""

import os
import tempfile
from unittest.mock import Mock, patch

import yaml
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
                {'name': 'EXAMPLE_PATH', 'default_value': '/default/path'},
                {'name': 'EXAMPLE_SIZE', 'default_value': 32},
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
        assert 'EXAMPLE_PATH (default: /default/path)' in result.output
        assert 'EXAMPLE_SIZE (default: 32)' in result.output
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

        self.runner.invoke(init, ['--template', 'image-classification', '.'])

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

        self.runner.invoke(init, ['.'])

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

    def test_cli_help_messages(self):
        """Test that CLI help messages are accessible."""
        runner = CliRunner()

        result = runner.invoke(list_templates, ['--help'])
        assert result.exit_code == 0
        assert 'List available pipeline templates' in result.output

        result = runner.invoke(info, ['--help'])
        assert result.exit_code == 0
        assert 'Show detailed information about a specific template' in result.output


class TestTemplateManagerYAMLProcessing:
    """Test cases for YAML processing functionality with generic parameter handling."""

    def test_yaml_parameter_extraction_robustness(self):
        """Test that parameter extraction handles various YAML structures without parameter dependencies."""
        from clarifai.utils.template_manager import TemplateManager

        manager = TemplateManager()

        # Test with valid argo spec using generic example parameters
        config = {
            'pipeline': {
                'orchestration_spec': {
                    'argo_orchestration_spec': '''
                        spec:
                          arguments:
                            parameters:
                              - name: example_param1
                                value: default1
                              - name: example_param2
                                value: default2
                    '''
                }
            }
        }

        params = manager._extract_parameters_from_config(config)
        assert len(params) == 2
        assert params[0]['name'] == 'example_param1'
        assert params[0]['default_value'] == 'default1'
        assert params[1]['name'] == 'example_param2'
        assert params[1]['default_value'] == 'default2'

    def test_yaml_parameter_extraction_edge_cases(self):
        """Test parameter extraction edge cases."""
        from clarifai.utils.template_manager import TemplateManager

        manager = TemplateManager()

        # Test with empty config
        params = manager._extract_parameters_from_config({})
        assert params == []

        # Test with missing pipeline section
        config = {'other': 'data'}
        params = manager._extract_parameters_from_config(config)
        assert params == []

        # Test with malformed argo spec
        config = {
            'pipeline': {
                'orchestration_spec': {'argo_orchestration_spec': 'invalid: yaml: content'}
            }
        }
        params = manager._extract_parameters_from_config(config)
        assert params == []

    def test_yaml_substitution_preserves_structure(self):
        """Test that YAML substitution maintains proper structure and formatting."""
        from clarifai.utils.template_manager import TemplateManager

        # Create a test config file
        test_config = {
            'pipeline': {
                'id': '<YOUR_PIPELINE_ID>',
                'user_id': '<YOUR_USER_ID>',
                'app_id': '<YOUR_APP_ID>',
                'orchestration_spec': {
                    'argo_orchestration_spec': yaml.dump(
                        {
                            'metadata': {'name': 'test'},
                            'spec': {
                                'arguments': {
                                    'parameters': [
                                        {'name': 'user_id', 'value': '<YOUR_USER_ID>'},
                                        {'name': 'app_id', 'value': '<YOUR_APP_ID>'},
                                    ]
                                }
                            },
                        }
                    )
                },
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.safe_dump(test_config, f)

            manager = TemplateManager()
            substitutions = {
                'user_id': 'test-user-123',
                'app_id': 'test-app-456',
                'id': 'test-pipeline-789',
            }

            manager._apply_config_substitutions(config_path, substitutions)

            # Verify the file is still valid YAML and has correct substitutions
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)

            assert updated_config['pipeline']['id'] == 'test-pipeline-789'
            assert updated_config['pipeline']['user_id'] == 'test-user-123'
            assert updated_config['pipeline']['app_id'] == 'test-app-456'
