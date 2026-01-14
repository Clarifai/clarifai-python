"""Tests for pipeline template functionality."""

import os
import tempfile

from clarifai.utils.template_manager import TemplateManager


class TestTemplateManager:
    """Test cases for TemplateManager."""

    def test_template_discovery(self):
        """Test that templates can be discovered."""
        manager = TemplateManager()
        templates = manager.list_templates()

        # Should find at least the test templates
        assert len(templates) > 0

        # Check structure - current implementation returns name and type only
        for template in templates:
            assert 'name' in template
            assert 'type' in template
            # Note: description is no longer returned in list_templates

    def test_template_filtering(self):
        """Test that templates can be filtered by type."""
        manager = TemplateManager()

        # Get all templates
        all_templates = manager.list_templates()

        # Get classifier templates (available in the test repository)
        classifier_templates = manager.list_templates('classifier')

        # Get detector templates (available in the test repository)
        detector_templates = manager.list_templates('detector')

        # Should have some of each type
        assert len(classifier_templates) > 0
        assert len(detector_templates) > 0

        # All classifier templates should be type 'classifier'
        for template in classifier_templates:
            assert template['type'] == 'classifier'

        # All detector templates should be type 'detector'
        for template in detector_templates:
            assert template['type'] == 'detector'

    def test_template_info_extraction(self):
        """Test that template info can be extracted."""
        manager = TemplateManager()

        # Test with actual template from repository
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

        # Check parameter structure (new format from YAML config)
        for param in info['parameters']:
            assert 'name' in param
            assert 'default_value' in param
            assert 'type' in param

    def test_parameter_extraction_from_config(self):
        """Test parameter extraction from config.yaml files."""
        manager = TemplateManager()

        # Test with actual template
        info = manager.get_template_info('classifier-pipeline-resnet')
        assert info is not None

        parameters = info['parameters']

        # Should find parameters from the config
        assert len(parameters) > 0

        # Check parameter structure (extracted from YAML config)
        param_names = [p['name'] for p in parameters]

        # These are actual parameters from classifier-pipeline-resnet template
        assert 'model_id' in param_names
        assert 'dataset_id' in param_names

        # Check parameter details
        for param in parameters:
            assert 'name' in param
            assert 'default_value' in param
            assert 'type' in param

    def test_template_copying(self):
        """Test template copying with substitutions."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test copying actual template from repository
            substitutions = {
                'YOUR_USER_ID': 'test-user',
                'YOUR_APP_ID': 'test-app',
                'YOUR_DATASET_ID': 'test-dataset',
                'YOUR_PIPELINE_ID': 'test-pipeline',
            }

            success = manager.copy_template('classifier-pipeline-resnet', temp_dir, substitutions)
            assert success

            # Check that files were created
            config_path = os.path.join(temp_dir, 'config.yaml')
            assert os.path.exists(config_path)

            # Check substitutions were applied
            with open(config_path, 'r') as f:
                content = f.read()
                assert 'test-user' in content or 'test-app' in content

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
