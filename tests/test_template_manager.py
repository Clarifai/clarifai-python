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

        # Check structure
        for template in templates:
            assert 'name' in template
            assert 'type' in template
            assert 'description' in template

    def test_template_filtering(self):
        """Test that templates can be filtered by type."""
        manager = TemplateManager()

        # Get all templates
        all_templates = manager.list_templates()

        # Get train templates
        train_templates = manager.list_templates('train')

        # Get data templates
        data_templates = manager.list_templates('data')

        # Should have some of each type
        assert len(train_templates) > 0
        assert len(data_templates) > 0

        # All train templates should be type 'train'
        for template in train_templates:
            assert template['type'] == 'train'

        # All data templates should be type 'data'
        for template in data_templates:
            assert template['type'] == 'data'

    def test_template_info_extraction(self):
        """Test that template info can be extracted."""
        manager = TemplateManager()

        # Test with image-classification template
        info = manager.get_template_info('image-classification')
        assert info is not None

        # Check required fields
        assert info['name'] == 'image-classification'
        assert info['type'] == 'train'
        assert 'parameters' in info
        assert 'step_directories' in info
        assert 'config' in info

        # Should have some parameters
        assert len(info['parameters']) > 0

        # Check parameter structure
        for param in info['parameters']:
            assert 'name' in param
            assert 'description' in param
            assert 'placeholder' in param

    def test_parameter_extraction_from_readme(self):
        """Test parameter extraction from README files."""
        manager = TemplateManager()

        # Create a temporary directory with a test README
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            readme_path = os.path.join(temp_dir, 'README.md')

            test_readme_content = """# Test Pipeline Template

## Pipeline Steps

### LoadDatasetStep
Description: Load the dataset
- `DATA_PATH` (string): Path to the dataset directory

### TrainModelStep  
Description: Train the model
- `BATCH_SIZE` (int): Training batch size (default: 32)
- `LEARNING_RATE` (float): Model learning rate

## Use Case
This template is used for image classification training.
"""

            with open(readme_path, 'w') as f:
                f.write(test_readme_content)

            parameters, use_case, step_descriptions = manager.extract_info_from_readme(temp_dir)

            # Should find parameters in the readme
            assert len(parameters) >= 3  # At least the 3 parameters we defined

            # Check parameter structure
            param_names = [p['name'] for p in parameters]
            assert 'DATA_PATH' in param_names
            assert 'BATCH_SIZE' in param_names
            assert 'LEARNING_RATE' in param_names

            # Check parameter display names and structure
            for param in parameters:
                if param['name'] == 'DATA_PATH':
                    assert param['description'] == 'Path to the dataset directory'
                    assert param['placeholder'] == '<DATA_PATH_VALUE>'

            # Check use case extraction
            assert 'image classification training' in use_case

    def test_template_copying(self):
        """Test template copying with substitutions."""
        manager = TemplateManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test copying image-classification template
            substitutions = {
                '<USER_ID_VALUE>': 'test-user',
                '<APP_ID_VALUE>': 'test-app',
                '<DATA_PATH>': '/test/path',
                '<BATCH_SIZE>': '64',
            }

            success = manager.copy_template('image-classification', temp_dir, substitutions)
            assert success

            # Check that files were created
            config_path = os.path.join(temp_dir, 'config.yaml')
            assert os.path.exists(config_path)

            # Check substitutions
            with open(config_path, 'r') as f:
                content = f.read()
                assert 'test-user' in content
                assert 'test-app' in content
                # Note: substitutions depend on the actual template content

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
