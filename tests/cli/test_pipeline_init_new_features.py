"""Tests for the new features in pipeline init command."""

import os
import yaml
from click.testing import CliRunner

from clarifai.cli.pipeline import init


class TestPipelineInitNewFeatures:
    """Test cases for new features in pipeline init command."""

    def test_init_command_with_pipeline_name(self):
        """Test that init command uses provided pipeline name."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['blog-generator'])

            assert result.exit_code == 0

            # Load and validate the generated config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Check that pipeline ID matches the provided name
            assert config['pipeline']['id'] == 'blog-generator'

    def test_init_command_with_user_id_and_app_id(self):
        """Test that init command uses provided user_id and app_id."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['my-pipeline', '--user_id', 'test_user', '--app_id', 'test_app'])

            assert result.exit_code == 0

            # Load and validate the main config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Check that user_id and app_id are set correctly without TODO comments
            assert config['pipeline']['user_id'] == 'test_user'
            assert config['pipeline']['app_id'] == 'test_app'
            assert config['pipeline']['id'] == 'my-pipeline'

            # Check that template references use the actual IDs
            argo_spec = config['pipeline']['orchestration_spec']['argo_orchestration_spec']
            assert 'users/test_user/apps/test_app/pipeline_steps/stepA' in argo_spec
            assert 'users/test_user/apps/test_app/pipeline_steps/stepB' in argo_spec
            # TODO comments should not be present when values are provided
            assert 'TODO: please fill in' not in argo_spec

            # Check step configs
            with open('stepA/config.yaml', 'r') as f:
                step_config = yaml.safe_load(f)
            assert step_config['pipeline_step']['user_id'] == 'test_user'
            assert step_config['pipeline_step']['app_id'] == 'test_app'

    def test_init_command_with_partial_params(self):
        """Test that init command handles partial user_id/app_id parameters."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['test-pipeline', '--user_id', 'partial_user'])

            assert result.exit_code == 0

            # Load and validate the main config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            # Check that user_id is set, app_id still has default
            assert config['pipeline']['user_id'] == 'partial_user'
            assert config['pipeline']['app_id'] == 'your_app_id'

            # Check that TODO comments are present only where needed
            config_content = open('config.yaml', 'r').read()
            assert 'user_id: "partial_user"' in config_content
            assert 'TODO: please fill in' in config_content  # Should be present for app_id and template refs

    def test_init_command_argo_spec_changes(self):
        """Test that Argo spec has input arguments and no generateName."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            result = runner.invoke(init, ['test-pipeline'])

            assert result.exit_code == 0

            # Load and validate the config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            argo_spec_str = config['pipeline']['orchestration_spec']['argo_orchestration_spec']
            argo_spec = yaml.safe_load(argo_spec_str)

            # Check that generateName is not present in metadata
            assert 'metadata' not in argo_spec or 'generateName' not in argo_spec.get('metadata', {})

            # Check that input arguments are present for both steps
            templates = argo_spec['spec']['templates']
            sequence_template = next(t for t in templates if t['name'] == 'sequence')
            steps = sequence_template['steps']

            # Check step-a has arguments
            step_a = steps[0][0]  # First step in first group
            assert step_a['name'] == 'step-a'
            assert 'arguments' in step_a
            assert 'parameters' in step_a['arguments']
            
            step_a_params = step_a['arguments']['parameters']
            input_text_param = next(p for p in step_a_params if p['name'] == 'input_text')
            assert input_text_param['value'] == 'Input Text to Step A'

            # Check step-b has arguments  
            step_b = steps[1][0]  # First step in second group
            assert step_b['name'] == 'step-b'
            assert 'arguments' in step_b
            assert 'parameters' in step_b['arguments']
            
            step_b_params = step_b['arguments']['parameters']
            input_text_param = next(p for p in step_b_params if p['name'] == 'input_text')
            assert input_text_param['value'] == 'Input Text to Step B'

    def test_init_command_with_custom_path_and_name(self):
        """Test init command with both pipeline name and custom path."""
        runner = CliRunner(env={"PYTHONIOENCODING": "utf-8"})

        with runner.isolated_filesystem():
            # Create custom directory first
            os.makedirs('custom_dir', exist_ok=True)
            result = runner.invoke(init, ['amazing-pipeline', 'custom_dir'])

            assert result.exit_code == 0

            # Check that files were created in the custom directory
            assert os.path.exists('custom_dir/config.yaml')
            assert os.path.exists('custom_dir/stepA/1/pipeline_step.py')
            assert os.path.exists('custom_dir/stepB/1/pipeline_step.py')
            
            # Check that the pipeline name is correct
            with open('custom_dir/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            assert config['pipeline']['id'] == 'amazing-pipeline'