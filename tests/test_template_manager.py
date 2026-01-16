"""Tests for pipeline template functionality."""

import os
import tempfile

import yaml

from clarifai.utils.template_manager import TemplateManager


class TestTemplateManager:
    """Test cases for TemplateManager."""

    def test_template_discovery(self):
        """Test that templates can be discovered."""
        manager = TemplateManager()
        templates = manager.list_templates()

        # Should find at least the test templates
        assert len(templates) > 0

        # Check structure
        for template in templates:
            assert 'name' in template
            assert 'type' in template

    def test_template_filtering(self):
        """Test that templates can be filtered by type."""
        manager = TemplateManager()

        # Get all templates (call for completeness; result not directly used here)
        manager.list_templates()

        # Get classifier and detector templates
        classifier_templates = manager.list_templates('classifier')
        detector_templates = manager.list_templates('detector')

        # Should have some of each type
        assert len(classifier_templates) > 0
        assert len(detector_templates) > 0

        # Verify type filtering works correctly
        for template in classifier_templates:
            assert template['type'] == 'classifier'

        for template in detector_templates:
            assert template['type'] == 'detector'

    def test_template_info_extraction(self):
        """Test that template info can be extracted."""
        manager = TemplateManager()

        info = manager.get_template_info('classifier-pipeline-resnet')
        assert info is not None

        # Check required fields
        assert info['name'] == 'classifier-pipeline-resnet'
        assert info['type'] == 'classifier'
        assert 'parameters' in info
        assert 'step_directories' in info
        assert 'config' in info

        # Should have some parameters
        assert len(info['parameters']) > 0

        # Check parameter structure
        for param in info['parameters']:
            assert 'name' in param
            assert 'default_value' in param

    def test_parameter_extraction_from_config(self):
        """Test parameter extraction from config.yaml files."""
        manager = TemplateManager()

        info = manager.get_template_info('classifier-pipeline-resnet')
        assert info is not None

        parameters = info['parameters']

        # Should find parameters from the config
        assert len(parameters) > 0

        # Check parameter structure
        param_names = [p['name'] for p in parameters]

        # Should have some parameters - specific parameters depend on the template
        assert len(param_names) > 0
        # Core system parameters should always be present
        assert 'user_id' in param_names
        assert 'app_id' in param_names

        # Check parameter details
        for param in parameters:
            assert 'name' in param
            assert 'default_value' in param

    def test_template_copying(self):
        """Test template copying with substitutions."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test copying actual template from repository
            substitutions = {
                'user_id': 'test-user',
                'app_id': 'test-app',
                'id': 'test-pipeline',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            # Check that files were created
            config_path = os.path.join(temp_dir, 'config.yaml')
            assert os.path.exists(config_path)

            # Check substitutions were applied
            with open(config_path, 'r') as f:
                content = f.read()
                assert (
                    'test-user' in content or 'test-app' in content or 'test-pipeline' in content
                )

    def test_nonexistent_template(self):
        """Test handling of nonexistent template."""
        manager = TemplateManager()

        # Should return None for nonexistent template
        info = manager.get_template_info('nonexistent-template')
        assert info is None

        # Should return False for copying nonexistent template
        with tempfile.TemporaryDirectory() as temp_dir:
            success = manager.copy_template('nonexistent-template', temp_dir, {})
            assert not success


class TestYAMLSubstitution:
    """Test cases for YAML-based parameter substitution."""

    def test_yaml_structure_preservation(self):
        """Test that YAML structure is preserved during substitution."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            substitutions = {
                'user_id': 'yaml-test-user',
                'app_id': 'yaml-test-app',
                'id': 'yaml-test-pipeline',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            # Read the config and verify it's valid YAML
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Verify structure is intact
            assert 'pipeline' in config
            assert 'id' in config['pipeline']
            assert 'user_id' in config['pipeline']
            assert 'app_id' in config['pipeline']

            # Verify substitutions were applied
            assert config['pipeline']['id'] == 'yaml-test-pipeline'
            assert config['pipeline']['user_id'] == 'yaml-test-user'
            assert config['pipeline']['app_id'] == 'yaml-test-app'

    def test_argo_orchestration_spec_substitution(self):
        """Test that Argo orchestration spec gets proper substitutions for system parameters."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            substitutions = {
                'user_id': 'argo-user',
                'app_id': 'argo-app',
                'id': 'argo-pipeline',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Verify argo spec exists and was modified
            assert 'orchestration_spec' in config['pipeline']
            argo_spec_raw = config['pipeline']['orchestration_spec']['argo_orchestration_spec']

            # The argo spec might be stored as a dict (after our processing) or string
            if isinstance(argo_spec_raw, str):
                argo_spec = yaml.safe_load(argo_spec_raw)
            else:
                argo_spec = argo_spec_raw

            # Check generateName was set correctly
            assert argo_spec['metadata']['generateName'] == 'argo-pipeline-'

            # Check that basic user/app parameters were substituted
            parameters = argo_spec['spec']['arguments']['parameters']
            param_values = {p['name']: p['value'] for p in parameters}
            assert param_values['user_id'] == 'argo-user'
            assert param_values['app_id'] == 'argo-app'

    def test_pipeline_step_substitution(self):
        """Test that pipeline step configs get proper substitutions."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            substitutions = {
                'user_id': 'step-user',
                'app_id': 'step-app',
                'id': 'step-pipeline',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            # Check pipeline step config
            step_dir = os.path.join(temp_dir, 'model-version-train-ps')
            if os.path.exists(step_dir):
                step_config_path = os.path.join(step_dir, 'config.yaml')
                if os.path.exists(step_config_path):
                    with open(step_config_path, 'r') as f:
                        step_config = yaml.safe_load(f)

                    assert 'pipeline_step' in step_config
                    assert step_config['pipeline_step']['user_id'] == 'step-user'
                    assert step_config['pipeline_step']['app_id'] == 'step-app'

    def test_parameter_extraction_accuracy(self):
        """Test that parameter extraction finds all expected parameters."""
        manager = TemplateManager()
        info = manager.get_template_info('classifier-pipeline-resnet')

        if info:
            parameters = info['parameters']
            param_names = [p['name'] for p in parameters]

            # Should contain key system parameters that are always present
            required_params = ['user_id', 'app_id']
            for required in required_params:
                assert required in param_names, f"Missing required system parameter: {required}"

            # Each parameter should have required fields
            for param in parameters:
                assert 'name' in param
                assert 'default_value' in param
                # default_value should not be empty/None for required system parameters
                if param['name'] in required_params:
                    assert param['default_value'] is not None

    def test_public_repository_access(self):
        """Test that public repository can be accessed without authentication."""
        manager = TemplateManager()

        # Test with default public repository URL
        assert manager.git_repo_url.startswith('https://github.com/')
        assert 'pipeline-examples' in manager.git_repo_url

        # Test with custom public repository URL
        custom_manager = TemplateManager('https://github.com/example/public-templates.git')
        assert custom_manager.git_repo_url == 'https://github.com/example/public-templates.git'

    def test_dynamic_parameter_handling(self):
        """Test that substitution works with any template-specific parameters."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with arbitrary template parameters
            substitutions = {
                'user_id': 'test-user',
                'app_id': 'test-app',
                'id': 'test-pipeline',
                # These could be any template-specific parameters
                'arbitrary_param_1': 'value1',
                'arbitrary_param_2': 'value2',
                'custom_setting': 'custom_value',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            # The key point is that basic system parameters always work
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # System parameters should be applied regardless of template
            assert config['pipeline']['id'] == 'test-pipeline'
            assert config['pipeline']['user_id'] == 'test-user'
            assert config['pipeline']['app_id'] == 'test-app'

    def test_literal_block_scalar_preservation(self):
        """Test that argo_orchestration_spec maintains proper literal block scalar format (|) after substitutions."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            substitutions = {
                'user_id': 'literal-user',
                'app_id': 'literal-app',
                'id': 'literal-pipeline',
            }

            # Copy a real template (which will have proper argo_orchestration_spec format)
            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            config_path = os.path.join(temp_dir, 'config.yaml')

            # Read the generated config and check that literal block scalar format is preserved
            with open(config_path, 'r') as f:
                result_content = f.read()

            # Check that literal block scalar format is preserved with proper | format (not |-)
            assert 'argo_orchestration_spec: |' in result_content, (
                f"Literal block scalar format should use '|' (not '|-'), but got:\n{result_content[:1000]}..."
            )

            # Ensure we're not getting the stripped format
            assert 'argo_orchestration_spec: |-' not in result_content, (
                f"Should not use '|-' format, but found it in:\n{result_content[:1000]}..."
            )

            # Verify the config is valid and substitutions were applied
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert config['pipeline']['id'] == 'literal-pipeline'
            assert config['pipeline']['user_id'] == 'literal-user'
            assert config['pipeline']['app_id'] == 'literal-app'

            # Verify the argo config was properly updated
            argo_spec_raw = config['pipeline']['orchestration_spec']['argo_orchestration_spec']

            # The argo spec might be stored as a dict (after our processing) or string
            if isinstance(argo_spec_raw, str):
                argo_spec = yaml.safe_load(argo_spec_raw)
            else:
                argo_spec = argo_spec_raw

            assert argo_spec['metadata']['generateName'] == 'literal-pipeline-'

            # Verify parameter substitutions in argo spec
            params = argo_spec['spec']['arguments']['parameters']
            param_values = {p['name']: p['value'] for p in params}
            assert param_values['user_id'] == 'literal-user'
            assert param_values['app_id'] == 'literal-app'
