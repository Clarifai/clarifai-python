"""Template management utilities for pipeline templates."""

import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import yaml

from clarifai.runners.pipelines.pipeline_builder import LiteralBlockDumper
from clarifai.utils.logging import logger


class TemplateManager:
    """Manages pipeline templates from remote public Git repository."""

    def __init__(self, git_repo_url: str = "https://github.com/Clarifai/pipeline-examples.git"):
        """Initialize template manager with Git repository.

        Args:
            git_repo_url: URL of the public Git repository containing templates
        """
        self.git_repo_url = git_repo_url

    def _shallow_clone_repo(self, target_dir: str) -> bool:
        """Perform a shallow clone of the public repository.

        Args:
            target_dir: Directory to clone into

        Returns:
            True if successful, False otherwise
        """
        try:
            # Shallow clone with depth 1, targeting the main branch
            result = subprocess.run(
                [
                    'git',
                    'clone',
                    '--depth=1',
                    '--branch=main',
                    '--no-checkout',
                    self.git_repo_url,
                    target_dir,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.debug(f"Git clone successful: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Git clone error: {e}")
            return False

    def _checkout_specific_folder(self, repo_dir: str, folder_name: str) -> bool:
        """Use sparse checkout to get only a specific folder.

        Args:
            repo_dir: Directory containing the cloned repo
            folder_name: Name of the folder to checkout

        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize sparse checkout
            subprocess.run(
                ['git', '-C', repo_dir, 'sparse-checkout', 'init', '--cone'],
                capture_output=True,
                text=True,
                check=True,
            )

            # Set the folder to checkout
            subprocess.run(
                ['git', '-C', repo_dir, 'sparse-checkout', 'set', folder_name],
                capture_output=True,
                text=True,
                check=True,
            )

            # Checkout the files
            subprocess.run(
                ['git', '-C', repo_dir, 'checkout'], capture_output=True, text=True, check=True
            )

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Sparse checkout failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Sparse checkout error: {e}")
            return False

    def list_templates(self, template_type: Optional[str] = None) -> List[Dict[str, str]]:
        """List available pipeline templates from Git repository.

        Args:
            template_type: Optional filter by template type (classifier, detector, etc.)

        Returns:
            List of template dictionaries with name and type
        """
        templates = []

        # Create temporary directory for git operations
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, 'repo')

            # Shallow clone the repository
            if not self._shallow_clone_repo(repo_dir):
                logger.error("Failed to clone template repository")
                return templates

            try:
                # List all directories in the repository root
                result = subprocess.run(
                    [
                        'git',
                        '-C',
                        repo_dir,
                        'ls-tree',
                        '--name-only',
                        '-d',
                        'main',
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                folder_names = result.stdout.strip().split('\n')

                for folder_name in folder_names:
                    if not folder_name:  # Skip empty lines
                        continue

                    # Parse template type and name from folder name
                    parsed_type, template_name = self._parse_template_name_and_type(folder_name)

                    if not parsed_type:  # Skip if not a valid template folder name
                        continue

                    # Filter by type if specified
                    if template_type and parsed_type != template_type:
                        continue

                    # Verify it has config.yaml
                    if self._verify_template_has_config(repo_dir, folder_name):
                        templates.append({'name': template_name, 'type': parsed_type})

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to list repository contents: {e.stderr}")
                return templates
            except Exception as e:
                logger.error(f"Error listing templates: {e}")
                return templates

        return sorted(templates, key=lambda x: (x['type'], x['name']))

    def _verify_template_has_config(self, repo_dir: str, folder_name: str) -> bool:
        """Verify that a template folder has config.yaml.

        Args:
            repo_dir: Repository directory
            folder_name: Template folder name

        Returns:
            True if config.yaml exists, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    'git',
                    '-C',
                    repo_dir,
                    'ls-tree',
                    '--name-only',
                    'main',
                    f'{folder_name}/config.yaml',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            return result.returncode == 0 and result.stdout.strip()

        except Exception as e:
            logger.debug(f"Error checking config.yaml for {folder_name}: {e}")
            return False

    def _parse_template_name_and_type(self, folder_name: str) -> Tuple[str, str]:
        """Parse template type and name from folder name.

        Args:
            folder_name: Folder name like 'classifier-pipeline-resnet'

        Returns:
            Tuple of (type, name) e.g., ('classifier', 'classifier-pipeline-resnet')
        """
        if '-' not in folder_name:
            return "", ""  # Invalid template name

        parts = folder_name.split('-', 1)
        template_type = parts[0]
        template_name = folder_name  # Whole name as specified

        return template_type, template_name

    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """Get detailed information about a specific template from Git repository.

        Args:
            template_name: Name of the template

        Returns:
            Template information dictionary or None if not found
        """
        # Create temporary directory for git operations
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, 'repo')

            # Shallow clone the repository
            if not self._shallow_clone_repo(repo_dir):
                logger.error("Failed to clone template repository")
                return None

            # Use sparse checkout to get only the specific template
            if not self._checkout_specific_folder(repo_dir, template_name):
                logger.error(f"Failed to checkout template: {template_name}")
                return None

            template_path = os.path.join(repo_dir, template_name)
            config_path = os.path.join(template_path, "config.yaml")

            if not os.path.exists(config_path):
                logger.error(f"config.yaml not found for template: {template_name}")
                return None

            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Extract parameters from config.yaml
                parameters = self._extract_parameters_from_config(config)

                # Parse template type from folder name
                template_type, _ = self._parse_template_name_and_type(template_name)

                return {
                    'name': template_name,
                    'type': template_type,
                    'path': template_path,
                    'config': config,
                    'parameters': parameters,
                    'step_directories': config.get('pipeline', {}).get('step_directories', []),
                }

            except Exception as e:
                logger.error(f"Error reading template {template_name}: {e}")
                return None

    def _extract_parameters_from_config(self, config: Dict) -> List[Dict[str, Any]]:
        """Extract parameter definitions from config.yaml.

        Args:
            config: Parsed YAML config dictionary

        Returns:
            List of parameter dictionaries with name and default_value
        """
        parameters = []

        # Extract parameters from orchestration_spec.arguments.parameters
        try:
            argo_spec = config.get('pipeline', {}).get('orchestration_spec', {})
            if isinstance(argo_spec, dict) and 'argo_orchestration_spec' in argo_spec:
                argo_yaml = yaml.safe_load(argo_spec['argo_orchestration_spec'])
                param_list = argo_yaml.get('spec', {}).get('arguments', {}).get('parameters', [])

                parameters = [
                    {'name': param.get('name', ''), 'default_value': param.get('value')}
                    for param in param_list
                ]

        except Exception as e:
            logger.warning(f"Error extracting parameters from config: {e}")

        return parameters

    def copy_template(
        self, template_name: str, destination: str, substitutions: Dict[str, str]
    ) -> bool:
        """Copy template from Git repository to destination with parameter substitution.

        Args:
            template_name: Name of the template to copy
            destination: Destination directory path
            substitutions: Dictionary of parameter substitutions

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create temporary directory for git operations
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_dir = os.path.join(temp_dir, 'repo')

                # Shallow clone the repository
                if not self._shallow_clone_repo(repo_dir):
                    logger.error("Failed to clone template repository")
                    return False

                # Use sparse checkout to get only the specific template
                if not self._checkout_specific_folder(repo_dir, template_name):
                    logger.error(f"Failed to checkout template: {template_name}")
                    return False

                template_path = os.path.join(repo_dir, template_name)

                if not os.path.exists(template_path):
                    logger.error(f"Template directory not found: {template_name}")
                    return False

                # Copy the template directory to destination
                shutil.copytree(template_path, destination, dirs_exist_ok=True)

                # Apply parameter substitutions to all copied files
                self._apply_substitutions(destination, substitutions)

                logger.info(f"Template '{template_name}' copied to {destination}")
                return True

        except Exception as e:
            logger.error(f"Error copying template {template_name}: {e}")
            return False

    def _apply_substitutions(self, directory: str, substitutions: Dict[str, str]):
        """Apply parameter substitutions to config.yaml files only.

        Args:
            directory: Directory to process
            substitutions: Dictionary of substitutions to apply
        """
        # Process config.yaml in root directory (pipeline config)
        root_config_path = os.path.join(directory, 'config.yaml')
        if os.path.exists(root_config_path):
            self._apply_config_substitutions(root_config_path, substitutions)

        # Process config.yaml in direct subdirectories only (pipeline step configs)
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                step_config_path = os.path.join(item_path, 'config.yaml')
                if os.path.exists(step_config_path):
                    self._apply_pipeline_step_substitutions(step_config_path, substitutions)

    def _apply_config_substitutions(self, config_path: str, substitutions: Dict[str, str]):
        """Apply specific configuration substitutions to config.yaml file using YAML traversal."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or 'pipeline' not in config:
                return

            pipeline_config = config['pipeline']

            # Set pipeline fields
            pipeline_config['id'] = substitutions['id']
            pipeline_config['user_id'] = substitutions['user_id']
            pipeline_config['app_id'] = substitutions['app_id']

            # Handle argo_orchestration_spec
            orch_spec = pipeline_config.get('orchestration_spec', {})
            if 'argo_orchestration_spec' in orch_spec:
                argo_config = yaml.safe_load(orch_spec['argo_orchestration_spec'])

                # Set generateName
                if 'metadata' not in argo_config:
                    argo_config['metadata'] = {}
                argo_config['metadata']['generateName'] = f"{substitutions['id']}-"

                # Update parameter values
                spec = argo_config.get('spec', {})
                parameters = spec.get('arguments', {}).get('parameters', [])
                for param in parameters:
                    param_name = param.get('name', '')
                    if param_name in substitutions:
                        param['value'] = substitutions[param_name]

                # Update templateRef paths
                for template in spec.get('templates', []):
                    for step_group in template.get('steps', []):
                        if isinstance(step_group, list):
                            for step in step_group:
                                template_ref = step.get('templateRef', {})
                                for field in ['name', 'template']:
                                    if field in template_ref:
                                        current_value = template_ref[field]
                                        if (
                                            isinstance(current_value, str)
                                            and '/pipeline_steps/' in current_value
                                        ):
                                            parts = current_value.split('/pipeline_steps/')
                                            if len(parts) == 2:
                                                step_name = parts[1]
                                                template_ref[field] = (
                                                    f"users/{substitutions['user_id']}/apps/{substitutions['app_id']}/pipeline_steps/{step_name}"
                                                )

                # Convert back to YAML string with literal block scalar style
                orch_spec['argo_orchestration_spec'] = yaml.safe_dump(
                    argo_config, default_flow_style=False, sort_keys=False
                )

            # Write the updated config back
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config, f, Dumper=LiteralBlockDumper, default_flow_style=False, sort_keys=False
                )

        except Exception as e:
            logger.warning(f"Could not apply config substitutions to {config_path}: {e}")

    def _apply_pipeline_step_substitutions(self, config_path: str, substitutions: Dict[str, str]):
        """Apply substitutions to pipeline step config.yaml file using YAML traversal."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or 'pipeline_step' not in config:
                return

            step_config = config['pipeline_step']
            step_config['user_id'] = substitutions['user_id']
            step_config['app_id'] = substitutions['app_id']

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config, f, Dumper=LiteralBlockDumper, default_flow_style=False, sort_keys=False
                )

        except Exception as e:
            logger.warning(f"Could not apply pipeline step substitutions to {config_path}: {e}")
