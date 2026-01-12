"""Template management utilities for pipeline templates."""

import os
import re
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
            template_type: Optional filter by template type (train, data, agent, etc.)

        Returns:
            List of template dictionaries with name, type, and description
        """
        templates = []

        if not os.path.exists(self.template_root):
            logger.warning(f"Template root directory not found: {self.template_root}")
            return templates

        # Scan each type directory (train, data, agent, etc.)
        for type_dir in self._get_template_type_directories():
            if template_type and type_dir != template_type:
                continue

            type_path = os.path.join(self.template_root, type_dir)
            templates.extend(self._scan_templates_in_directory(type_path, type_dir))

        return sorted(templates, key=lambda x: (x['type'], x['name']))

    def _get_template_type_directories(self) -> List[str]:
        """Get list of template type directories."""
        return [
            d
            for d in os.listdir(self.template_root)
            if os.path.isdir(os.path.join(self.template_root, d))
        ]

    def _scan_templates_in_directory(self, type_path: str, type_name: str) -> List[Dict[str, str]]:
        """Scan templates within a type directory."""
        templates = []

        for template_name in os.listdir(type_path):
            template_path = os.path.join(type_path, template_name)

            if not os.path.isdir(template_path):
                continue

            if not self._is_valid_template(template_path):
                continue

            template_info = self._extract_template_info(template_path, template_name, type_name)
            if template_info:
                templates.append(template_info)

        return templates

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

            # Extract parameters and description from README
            parameters, use_case_description, step_descriptions = self.extract_info_from_readme(
                template_path
            )

            # Get template type from path
            template_type = os.path.basename(os.path.dirname(template_path))

            return {
                'name': template_name,
                'type': template_type,
                'path': template_path,
                'config': config,
                'parameters': parameters,
                'use_case': use_case_description,
                'step_descriptions': step_descriptions,
                'step_directories': config.get('pipeline', {}).get('step_directories', []),
            }

        except Exception as e:
            logger.error(f"Error reading template {template_name}: {e}")
            return None

    def extract_info_from_readme(
        self, template_path: str
    ) -> Tuple[List[Dict[str, str]], str, Dict[str, str]]:
        """Extract parameter definitions and use case from template README.

        Args:
            template_path: Path to template directory

        Returns:
            Tuple of (parameters list, use case description, step descriptions dict)
        """
        readme_path = os.path.join(template_path, 'README.md')
        if not os.path.exists(readme_path):
            return [], "Pipeline template", {}

        try:
            with open(readme_path, 'r') as f:
                content = f.read()

            parameters = []
            use_case = "Pipeline template"
            step_descriptions = {}

            # Extract use case from the "## Use Case" section
            use_case = self._extract_use_case(content)

            # Extract parameters and step descriptions from Pipeline Steps section
            parameters, step_descriptions = self._extract_pipeline_steps_info(content)

            # Remove duplicates and sort
            unique_parameters = self._deduplicate_parameters(parameters)
            return sorted(unique_parameters, key=lambda x: x['name']), use_case, step_descriptions

        except Exception as e:
            logger.warning(f"Error reading README {readme_path}: {e}")
            return [], "Pipeline template", {}

    def _extract_use_case(self, content: str) -> str:
        """Extract use case description from README content."""
        use_case_match = re.search(r'## Use Case\s*\n(.+?)(?=\n##|\n\n|\Z)', content, re.DOTALL)
        if not use_case_match:
            return "Pipeline template"

        use_case = use_case_match.group(1).strip()
        # Take only the first sentence/line for concise description
        if '\n' in use_case:
            use_case = use_case.split('\n')[0].strip()
        # Remove markdown formatting
        use_case = re.sub(r'\*\*(.+?)\*\*', r'\1', use_case)  # Remove bold
        return use_case

    def _extract_pipeline_steps_info(
        self, content: str
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Extract parameters and step descriptions from Pipeline Steps section."""
        parameters = []
        step_descriptions = {}

        # Find the Pipeline Steps section
        steps_content = self._extract_steps_section(content)
        if not steps_content:
            return parameters, step_descriptions

        # Find step subsections (### StepName)
        step_sections = re.findall(
            r'### (\w+)\s*\n(.+?)(?=\n###|\n##|\Z)', steps_content, re.DOTALL
        )

        for step_name, step_content in step_sections:
            # Extract step description
            step_description = self._extract_step_description(step_content)
            if step_description:
                step_descriptions[step_name] = step_description

            # Extract parameters from this step
            step_parameters = self._extract_step_parameters(step_content)
            parameters.extend(step_parameters)

        return parameters, step_descriptions

    def _extract_steps_section(self, content: str) -> str:
        """Extract the Pipeline Steps section content."""
        steps_start = content.find('## Pipeline Steps')
        if steps_start == -1:
            return ""

        use_case_start = content.find('## Use Case')
        if use_case_start != -1 and use_case_start > steps_start:
            return content[steps_start:use_case_start]

        # If no Use Case section, look for next ## section or end
        next_section = re.search(r'## Pipeline Steps\s*\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
        return next_section.group(0) if next_section else content[steps_start:]

    def _extract_step_description(self, step_content: str) -> Optional[str]:
        """Extract step description from step content."""
        step_desc_match = re.match(r'([^-\n]+?)(?=\n|:|\n-)', step_content.strip())
        if not step_desc_match:
            return None

        step_description = step_desc_match.group(1).strip()
        # Remove "with the following parameters" suffix if present
        return re.sub(r'\s+with the following parameters:?$', '', step_description)

    def _extract_step_parameters(self, step_content: str) -> List[Dict[str, str]]:
        """Extract parameters from a single step's content."""
        parameters = []

        # Extract parameters from bullet points like "- `param_name` (type): description"
        param_matches = re.findall(
            r'-\s+`([^`]+)`\s+\([^)]+\):\s*(.+?)(?=\n-|\n\n|\n###|\Z)',
            step_content,
            re.DOTALL,
        )

        for param_name, param_description in param_matches:
            # Clean up the description
            param_description = param_description.strip().replace('\n', ' ')
            param_description = re.sub(r'\s+', ' ', param_description)

            parameters.append(
                {
                    'name': param_name,
                    'description': param_description,
                    'placeholder': f'<{param_name.upper()}_VALUE>',
                }
            )

        return parameters

    def _deduplicate_parameters(self, parameters: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate parameters based on name."""
        seen = set()
        unique_parameters = []
        for param in parameters:
            if param['name'] not in seen:
                seen.add(param['name'])
                unique_parameters.append(param)
        return unique_parameters

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

        # Search in all type directories
        for type_dir in self._get_template_type_directories():
            template_path = os.path.join(self.template_root, type_dir, template_name)
            if os.path.exists(template_path) and self._is_valid_template(template_path):
                return template_path

        return None

    def _extract_template_info(
        self, template_path: str, template_name: str, template_type: str
    ) -> Optional[Dict[str, str]]:
        """Extract basic info about a template.

        Args:
            template_path: Path to template directory
            template_name: Name of the template
            template_type: Type of the template

        Returns:
            Template info dictionary or None if invalid
        """
        try:
            # Get description from README use case section
            _, description, _ = self.extract_info_from_readme(template_path)
            return {'name': template_name, 'type': template_type, 'description': description}

        except Exception as e:
            logger.warning(f"Invalid template at {template_path}: {e}")
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

                        # Apply all substitutions
                        modified = False
                        for placeholder, value in substitutions.items():
                            if placeholder in content:
                                content = content.replace(placeholder, value)
                                modified = True

                        # Write back if modified
                        if modified:
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
