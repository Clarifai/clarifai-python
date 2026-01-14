"""Template management utilities for pipeline templates."""

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import yaml

from clarifai.utils.logging import logger


class TemplateManager:
    """Manages pipeline templates from remote Git repository."""

    def __init__(self, git_repo_url: str = "git@github.com:Clarifai/clarifai-train.git"):
        """Initialize template manager with Git repository.

        Args:
            git_repo_url: URL of the Git repository containing templates
        """
        self.git_repo_url = git_repo_url
        self.git_pat = os.getenv('CLARIFAI_GIT_PAT')

        if not self.git_pat:
            logger.warning(
                "CLARIFAI_GIT_PAT environment variable not set. Git operations may fail."
            )

    def _setup_git_auth_url(self) -> str:
        """Setup Git URL with PAT authentication for HTTPS."""
        if self.git_pat and self.git_repo_url.startswith('git@github.com:'):
            # Convert SSH URL to HTTPS with PAT
            repo_path = self.git_repo_url.replace('git@github.com:', '').replace('.git', '')
            return f"https://{self.git_pat}@github.com/{repo_path}.git"
        elif self.git_pat and self.git_repo_url.startswith('https://github.com/'):
            # Add PAT to existing HTTPS URL
            return self.git_repo_url.replace(
                'https://github.com/', f'https://{self.git_pat}@github.com/'
            )
        else:
            # Return original URL (may work for SSH with keys or public repos)
            return self.git_repo_url

    def _shallow_clone_repo(self, target_dir: str) -> bool:
        """Perform a shallow clone of the repository.

        Args:
            target_dir: Directory to clone into

        Returns:
            True if successful, False otherwise
        """
        try:
            auth_url = self._setup_git_auth_url()

            # Shallow clone with depth 1, targeting the main branch
            result = subprocess.run(
                [
                    'git',
                    'clone',
                    '--depth=1',
                    '--branch=main',
                    '--no-checkout',
                    auth_url,
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

    def _extract_parameters_from_config(self, config: Dict) -> List[Dict[str, any]]:
        """Extract parameter definitions from config.yaml.

        Args:
            config: Parsed YAML config dictionary

        Returns:
            List of parameter dictionaries with name, default_value, and type
        """
        parameters = []

        # Extract parameters from orchestration_spec.arguments.parameters
        try:
            argo_spec = config.get('pipeline', {}).get('orchestration_spec', {})
            if isinstance(argo_spec, dict) and 'argo_orchestration_spec' in argo_spec:
                # Parse the YAML string within argo_orchestration_spec
                argo_yaml = yaml.safe_load(argo_spec['argo_orchestration_spec'])
                param_list = argo_yaml.get('spec', {}).get('arguments', {}).get('parameters', [])

                for param in param_list:
                    param_name = param.get('name', '')
                    param_value = param.get('value')

                    # Skip special parameters that are handled separately
                    if param_name in ['user_id', 'app_id']:
                        continue

                    # Determine parameter type from value
                    param_type = self._infer_parameter_type(param_value)

                    parameters.append(
                        {'name': param_name, 'default_value': param_value, 'type': param_type}
                    )

        except Exception as e:
            logger.warning(f"Error extracting parameters from config: {e}")

        return parameters

    def _infer_parameter_type(self, value) -> str:
        """Infer parameter type from its value."""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, list):
            return 'array'
        else:
            return 'str'

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
        """Apply parameter substitutions to all files in directory.

        Args:
            directory: Directory to process
            substitutions: Dictionary of substitutions to apply
        """
        # Directories to skip during substitution
        skip_dirs = {
            '.git',
            '.venv',
            '__pycache__',
            '.pytest_cache',
            '.mypy_cache',
            'node_modules',
            '.env',
        }

        for root, dirs, files in os.walk(directory):
            # Remove skip directories from the dirs list to avoid walking into them
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file_name in files:
                file_path = os.path.join(root, file_name)

                # Only process text files (skip binary files)
                if self._is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Apply substitutions for both placeholder and parameter formats
                        original_file_content = content

                        for placeholder, value in substitutions.items():
                            # Handle standardized placeholders like <YOUR_USER_ID>
                            placeholder_with_brackets = f"<{placeholder}>"

                            if placeholder_with_brackets in content:
                                content = content.replace(placeholder_with_brackets, value)
                            elif placeholder in content:
                                # Use targeted YAML patterns for parameter values
                                patterns = [
                                    rf'(value:\s*)({re.escape(placeholder)})(\s*$|\s*\n)',  # value: placeholder
                                    rf'(value:\s*")({re.escape(placeholder)})(")',  # value: "placeholder"
                                ]

                                pattern_matched = False
                                for pattern in patterns:
                                    try:
                                        if re.search(pattern, content, re.MULTILINE):
                                            content = re.sub(
                                                pattern,
                                                rf'\g<1>{value}\g<3>',
                                                content,
                                                flags=re.MULTILINE,
                                            )
                                            pattern_matched = True
                                            break
                                    except re.error:
                                        continue

                                # Safe fallback for long, unique string values only
                                if (
                                    not pattern_matched
                                    and not placeholder.isupper()
                                    and len(placeholder) > 8
                                    and not placeholder.isdigit()
                                    and not any(char.isdigit() for char in placeholder)
                                ):
                                    content = content.replace(placeholder, value)

                        # Write back if any changes were made
                        if content != original_file_content:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)

                    except Exception as e:
                        logger.warning(f"Could not process file {file_path}: {e}")

    def _is_text_file(self, file_path: str) -> bool:
        """Check if file is a text file that can be processed for substitutions.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be processed for substitutions
        """
        # Check file extension
        text_extensions = {'.py', '.yaml', '.yml', '.txt', '.md', '.json', '.cfg', '.ini', '.sh'}
        _, ext = os.path.splitext(file_path.lower())

        if ext in text_extensions:
            return True

        # For files without extension, try to detect if it's text
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' not in chunk  # Simple binary detection
        except:
            return False
