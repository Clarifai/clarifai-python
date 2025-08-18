"""Pipeline configuration validation utilities."""

import re
from typing import Any, Dict, List

import yaml


class PipelineConfigValidator:
    """Validator for pipeline configuration files."""

    # Regex patterns for templateRef validation
    TEMPLATE_REF_WITH_VERSION_PATTERN = re.compile(
        r'^users/([^/]+)/apps/([^/]+)/pipeline_steps/([^/]+)/versions/([^/]+)$'
    )
    TEMPLATE_REF_WITHOUT_VERSION_PATTERN = re.compile(
        r'^users/([^/]+)/apps/([^/]+)/pipeline_steps/([^/]+)$'
    )

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate the pipeline configuration."""
        cls._validate_pipeline_section(config)
        cls._validate_orchestration_spec(config)

    @classmethod
    def _validate_pipeline_section(cls, config: Dict[str, Any]) -> None:
        """Validate the pipeline section of the config."""
        if "pipeline" not in config:
            raise ValueError("'pipeline' section not found in config.yaml")

        pipeline = config["pipeline"]
        required_fields = ["id", "user_id", "app_id"]

        for field in required_fields:
            if field not in pipeline:
                raise ValueError(f"'{field}' not found in pipeline section of config.yaml")
            if not pipeline[field]:
                raise ValueError(f"'{field}' cannot be empty in config.yaml")

        # Validate step_directories if present
        if "step_directories" in pipeline:
            if not isinstance(pipeline["step_directories"], list):
                raise ValueError("'step_directories' must be a list")

        # Validate orchestration_spec is present
        if "orchestration_spec" not in pipeline:
            raise ValueError("'orchestration_spec' not found in pipeline section")

    @classmethod
    def _validate_orchestration_spec(cls, config: Dict[str, Any]) -> None:
        """Validate the orchestration spec contains valid Argo workflow."""
        pipeline = config["pipeline"]
        orchestration_spec = pipeline["orchestration_spec"]

        if "argo_orchestration_spec" not in orchestration_spec:
            raise ValueError("'argo_orchestration_spec' not found in orchestration_spec")

        argo_spec_str = orchestration_spec["argo_orchestration_spec"]

        try:
            argo_spec = yaml.safe_load(argo_spec_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in argo_orchestration_spec: {e}")

        cls._validate_argo_workflow(argo_spec)

    @classmethod
    def _validate_argo_workflow(cls, argo_spec: Dict[str, Any]) -> None:
        """Validate the Argo workflow structure."""
        # Basic Argo workflow validation
        if not isinstance(argo_spec, dict):
            raise ValueError("argo_orchestration_spec must be a valid YAML object")

        required_fields = ["apiVersion", "kind", "spec"]
        for field in required_fields:
            if field not in argo_spec:
                raise ValueError(f"'{field}' not found in argo_orchestration_spec")

        if argo_spec["apiVersion"] != "argoproj.io/v1alpha1":
            raise ValueError("argo_orchestration_spec must have apiVersion 'argoproj.io/v1alpha1'")

        if argo_spec["kind"] != "Workflow":
            raise ValueError("argo_orchestration_spec must have kind 'Workflow'")

        # Validate templates and steps
        spec = argo_spec["spec"]
        if "templates" not in spec:
            raise ValueError("'templates' not found in argo_orchestration_spec.spec")

        cls._validate_argo_templates(spec["templates"])

    @classmethod
    def _validate_argo_templates(cls, templates: List[Dict[str, Any]]) -> None:
        """Validate Argo workflow templates."""
        for template in templates:
            if "steps" in template:
                for step_group in template["steps"]:
                    for step in step_group:
                        if "templateRef" in step:
                            template_ref = step["templateRef"]
                            cls._validate_template_ref(template_ref)

    @classmethod
    def _validate_template_ref(cls, template_ref: Dict[str, Any]) -> None:
        """Validate a templateRef in the Argo workflow."""
        if "name" not in template_ref or "template" not in template_ref:
            raise ValueError("templateRef must have both 'name' and 'template' fields")

        name = template_ref["name"]
        template = template_ref["template"]

        if name != template:
            raise ValueError(f"templateRef name '{name}' must match template '{template}'")

        # Check if it matches either pattern
        if not (
            cls.TEMPLATE_REF_WITH_VERSION_PATTERN.match(name)
            or cls.TEMPLATE_REF_WITHOUT_VERSION_PATTERN.match(name)
        ):
            raise ValueError(
                f"templateRef name '{name}' must match either pattern:\n"
                f"  - users/{{user_id}}/apps/{{app_id}}/pipeline_steps/{{step_id}}\n"
                f"  - users/{{user_id}}/apps/{{app_id}}/pipeline_steps/{{step_id}}/versions/{{version_id}}"
            )

    @classmethod
    def get_pipeline_steps_without_versions(cls, config: Dict[str, Any]) -> List[str]:
        """Get list of pipeline step names that don't have versions in templateRef."""
        pipeline = config["pipeline"]
        orchestration_spec = pipeline["orchestration_spec"]
        argo_spec_str = orchestration_spec["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)

        steps_without_versions = []

        for template in argo_spec["spec"]["templates"]:
            if "steps" in template:
                for step_group in template["steps"]:
                    for step in step_group:
                        if "templateRef" in step:
                            template_ref = step["templateRef"]
                            name = template_ref["name"]

                            # Check if it's without version
                            if cls.TEMPLATE_REF_WITHOUT_VERSION_PATTERN.match(name):
                                # Extract step name
                                parts = name.split('/')
                                step_name = parts[-1]  # Last part is the step name
                                if step_name not in steps_without_versions:
                                    steps_without_versions.append(step_name)

        return steps_without_versions
