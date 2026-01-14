"""Template management utilities for pipeline templates."""

import os
import shutil
from typing import Dict, List, Optional, Tuple

import yaml

from clarifai.utils.logging import logger


class TemplateManager:
    """Manages pipeline templates including discovery, analysis, and copying."""

    def __init__(self, template_root: str = "/Users/yashdixit/work/clarifai-train"):
        """Initialize template manager with template repository root.

        Args:
            template_root: Path to the template repository root directory
        """
        self.template_root = template_root

    def list_templates(self, template_type: Optional[str] = None) -> List[Dict[str, str]]:
        """List available pipeline templates.

        Args:
            template_type: Optional filter by template type (classifier, detector, etc.)

        Returns:
            List of template dictionaries with name and type
        """
        templates = []

        if not os.path.exists(self.template_root):
            logger.warning(f"Template root directory not found: {self.template_root}")
            return templates

        # Scan template directories directly in root
        for folder_name in os.listdir(self.template_root):
            folder_path = os.path.join(self.template_root, folder_name)
            
            if not os.path.isdir(folder_path) or folder_name.startswith('.'):
                continue
                
            # Parse template type and name from folder name
            parsed_type, template_name = self._parse_template_name_and_type(folder_name)
            
            if not parsed_type:  # Skip if not a valid template folder name
                continue
                
            # Filter by type if specified
            if template_type and parsed_type != template_type:
                continue
                
            if not self._is_valid_template(folder_path):
                continue
                
            templates.append({
                'name': template_name,
                'type': parsed_type
            })

        return sorted(templates, key=lambda x: (x['type'], x['name']))

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

    def _is_valid_template(self, template_path: str) -> bool:
        """Check if a directory is a valid template (has config.yaml)."""
        config_path = os.path.join(template_path, "config.yaml")
        return os.path.exists(config_path)

    def get_template_info(self, template_name: str) -> Optional[Dict]:
        """Get detailed information about a specific template.

        Args:
            template_name: Name of the template

        Returns:
            Template information dictionary or None if not found
        """
        template_path = self._find_template_path(template_name)
        if not template_path:
            return None

        config_path = os.path.join(template_path, "config.yaml")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Extract parameters from config.yaml
            parameters = self._extract_parameters_from_config(config)

            # Parse template type from folder name
            folder_name = os.path.basename(template_path)
            template_type, _ = self._parse_template_name_and_type(folder_name)

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
                    
                    parameters.append({
                        'name': param_name,
                        'default_value': param_value,
                        'type': param_type
                    })
                    
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
        """Copy template to destination with parameter substitution.

        Args:
            template_name: Name of the template to copy
            destination: Destination directory path
            substitutions: Dictionary of parameter substitutions

        Returns:
            True if successful, False otherwise
        """
        template_path = self._find_template_path(template_name)
        if not template_path:
            logger.error(f"Template not found: {template_name}")
            return False

        try:
            # Simply copy the entire template directory to destination
            shutil.copytree(template_path, destination, dirs_exist_ok=True)

            # Apply parameter substitutions to all copied files
            self._apply_substitutions(destination, substitutions)

            logger.info(f"Template '{template_name}' copied to {destination}")
            return True

        except Exception as e:
            logger.error(f"Error copying template {template_name}: {e}")
            return False

    def _find_template_path(self, template_name: str) -> Optional[str]:
        """Find the full path for a template by name.

        Args:
            template_name: Name of the template

        Returns:
            Full path to template directory or None if not found
        """
        if not os.path.exists(self.template_root):
            return None

        # Look for template directly in root directory
        template_path = os.path.join(self.template_root, template_name)
        if os.path.exists(template_path) and self._is_valid_template(template_path):
            return template_path

        return None

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

                        # Apply substitutions - handle both placeholder and parameter formats
                        original_file_content = content
                        
                        for placeholder, value in substitutions.items():
                            # For placeholders like YOUR_USER_ID, YOUR_APP_ID, etc., 
                            # look for the <PLACEHOLDER> format in templates
                            placeholder_with_brackets = f"<{placeholder}>"
                            
                            if placeholder_with_brackets in content:
                                # Direct substitution for standardized placeholders
                                content = content.replace(placeholder_with_brackets, value)
                            elif placeholder in content:
                                # For raw parameter values, use targeted YAML pattern substitution
                                # to avoid accidentally replacing other occurrences
                                import re
                                
                                # Use targeted YAML value patterns for parameter substitution
                                patterns = [
                                    rf'(value:\s*)({re.escape(placeholder)})(\s*$|\s*\n)',  # value: placeholder
                                    rf'(value:\s*")({re.escape(placeholder)})(")',  # value: "placeholder"
                                ]
                                
                                pattern_matched = False
                                for pattern in patterns:
                                    try:
                                        if re.search(pattern, content, re.MULTILINE):
                                            content = re.sub(pattern, rf'\g<1>{value}\g<3>', content, flags=re.MULTILINE)
                                            pattern_matched = True
                                            break
                                    except re.error:
                                        continue
                                
                                # Disable fallback for parameter values to prevent accidental substitutions
                                # Only use fallback for long string values that are unlikely to appear elsewhere
                                if (not pattern_matched and 
                                    not placeholder.isupper() and 
                                    len(placeholder) > 8 and 
                                    not placeholder.isdigit() and
                                    not any(char.isdigit() for char in placeholder)):
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
