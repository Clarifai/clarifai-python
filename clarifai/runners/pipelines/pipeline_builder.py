import os
import re
import sys
from typing import List, Dict, Any

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.utils.logging import logger


class PipelineConfigValidator:
    """Validator for pipeline configuration files."""
    
    # Regex patterns for templateRef validation
    TEMPLATE_REF_WITH_VERSION_PATTERN = re.compile(
        r'^users/([^/]+)/apps/([^/]+)/pipeline-steps/([^/]+)/versions/([^/]+)$'
    )
    TEMPLATE_REF_WITHOUT_VERSION_PATTERN = re.compile(
        r'^users/([^/]+)/apps/([^/]+)/pipeline-steps/([^/]+)$'
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
        if not (cls.TEMPLATE_REF_WITH_VERSION_PATTERN.match(name) or 
                cls.TEMPLATE_REF_WITHOUT_VERSION_PATTERN.match(name)):
            raise ValueError(
                f"templateRef name '{name}' must match either pattern:\n"
                f"  - users/{{user_id}}/apps/{{app_id}}/pipeline-steps/{{step_id}}\n"
                f"  - users/{{user_id}}/apps/{{app_id}}/pipeline-steps/{{step_id}}/versions/{{version_id}}"
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


class PipelineBuilder:
    """Pipeline Builder class for managing pipeline upload to Clarifai."""
    
    def __init__(self, config_path: str):
        """
        Initialize PipelineBuilder.
        
        :param config_path: Path to the pipeline configuration file
        """
        self._client = None
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        self.config = self._load_config()
        self.validator = PipelineConfigValidator()
        self.validator.validate_config(self.config)
        
        # Extract pipeline info
        pipeline_config = self.config["pipeline"]
        self.pipeline_id = pipeline_config["id"]
        self.user_id = pipeline_config["user_id"]
        self.app_id = pipeline_config["app_id"]
        
        # Track uploaded pipeline step versions
        self.uploaded_step_versions = {}
    
    @property
    def client(self):
        """Get or create the Clarifai client."""
        if self._client is None:
            self._client = BaseClient(user_id=self.user_id, app_id=self.app_id)
        return self._client
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and return the configuration from config file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config file {self.config_path}: {e}")
    
    def _save_config(self) -> None:
        """Save the updated configuration back to the file."""
        try:
            with open(self.config_path, 'w') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ValueError(f"Error saving config file {self.config_path}: {e}")
    
    def upload_pipeline_steps(self) -> bool:
        """Upload all pipeline steps listed in step_directories."""
        pipeline_config = self.config["pipeline"]
        step_directories = pipeline_config.get("step_directories", [])
        
        if not step_directories:
            logger.info("No pipeline steps to upload (step_directories is empty)")
            return True
        
        logger.info(f"Uploading {len(step_directories)} pipeline steps...")
        
        for step_dir in step_directories:
            step_path = os.path.join(self.config_dir, step_dir)
            
            if not os.path.exists(step_path):
                logger.error(f"Pipeline step directory not found: {step_path}")
                return False
            
            logger.info(f"Uploading pipeline step from directory: {step_dir}")
            
            # Create a backup of the original upload function to capture the version
            # We'll need to modify the upload process to capture the version ID
            success, version_id = self._upload_pipeline_step_with_version_capture(step_path)
            
            if not success:
                logger.error(f"Failed to upload pipeline step from directory: {step_dir}")
                return False
            
            # Store the version ID for later use
            self.uploaded_step_versions[step_dir] = version_id
            logger.info(f"Successfully uploaded pipeline step {step_dir} with version {version_id}")
        
        return True
    
    def _upload_pipeline_step_with_version_capture(self, step_path: str) -> tuple[bool, str]:
        """Upload a pipeline step and capture its version ID."""
        try:
            # Use the existing pipeline step builder
            from clarifai.runners.pipeline_steps.pipeline_step_builder import PipelineStepBuilder
            
            builder = PipelineStepBuilder(step_path)
            
            # Create dockerfile if needed
            builder.create_dockerfile()
            
            # Check if step exists
            exists = builder.check_pipeline_step_exists()
            if exists:
                logger.info(f"Pipeline step {builder.pipeline_step_id} already exists, creating new version")
            else:
                logger.info(f"Creating new pipeline step {builder.pipeline_step_id}")
            
            # Upload the pipeline step version directly without the interactive prompt
            success = builder.upload_pipeline_step_version()
            
            if success and builder.pipeline_step_version_id:
                return True, builder.pipeline_step_version_id
            else:
                logger.error("Failed to get pipeline step version ID after upload")
                return False, ""
                
        except Exception as e:
            logger.error(f"Error uploading pipeline step: {e}")
            return False, ""
    
    def update_config_with_versions(self) -> None:
        """Update the config.yaml with uploaded pipeline step versions."""
        if not self.uploaded_step_versions:
            logger.info("No pipeline step versions to update in config")
            return
        
        logger.info("Updating config.yaml with pipeline step versions...")
        
        # Update the orchestration spec
        pipeline_config = self.config["pipeline"]
        orchestration_spec = pipeline_config["orchestration_spec"]
        argo_spec_str = orchestration_spec["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)
        
        # Update templateRef names to include versions
        self._update_template_refs_with_versions(argo_spec)
        
        # Update the config
        orchestration_spec["argo_orchestration_spec"] = yaml.dump(argo_spec, default_flow_style=False)
        
        # Remove uploaded directories from step_directories
        remaining_dirs = []
        for step_dir in pipeline_config.get("step_directories", []):
            if step_dir not in self.uploaded_step_versions:
                remaining_dirs.append(step_dir)
        
        pipeline_config["step_directories"] = remaining_dirs
        
        # Save the updated config
        self._save_config()
        logger.info("Updated config.yaml with pipeline step versions")
    
    def _update_template_refs_with_versions(self, argo_spec: Dict[str, Any]) -> None:
        """Update templateRef names in Argo spec to include version information."""
        for template in argo_spec["spec"]["templates"]:
            if "steps" in template:
                for step_group in template["steps"]:
                    for step in step_group:
                        if "templateRef" in step:
                            template_ref = step["templateRef"]
                            name = template_ref["name"]
                            
                            # Check if this is a templateRef without version that we uploaded
                            if self.validator.TEMPLATE_REF_WITHOUT_VERSION_PATTERN.match(name):
                                # Extract step name
                                parts = name.split('/')
                                step_name = parts[-1]
                                
                                # Find the corresponding directory and version
                                for step_dir, version_id in self.uploaded_step_versions.items():
                                    # The step name should match the directory name or be derivable from it
                                    if step_name == step_dir:
                                        # Update the templateRef to include version
                                        new_name = f"{name}/versions/{version_id}"
                                        template_ref["name"] = new_name
                                        template_ref["template"] = new_name
                                        logger.info(f"Updated templateRef from {name} to {new_name}")
                                        break
    
    def create_pipeline(self) -> bool:
        """Create the pipeline using PostPipelines RPC."""
        logger.info(f"Creating pipeline {self.pipeline_id}...")
        
        try:
            # Create pipeline proto
            pipeline = resources_pb2.Pipeline(
                id=self.pipeline_id,
                user_id=self.user_id,
                app_id=self.app_id
            )
            
            # Add orchestration spec
            pipeline_config = self.config["pipeline"]
            orchestration_spec = pipeline_config["orchestration_spec"]
            argo_spec_str = orchestration_spec["argo_orchestration_spec"]
            
            # Parse the Argo spec to get API version
            argo_spec = yaml.safe_load(argo_spec_str)
            api_version = argo_spec.get("apiVersion", "argoproj.io/v1alpha1")
            
            # Create pipeline version with orchestration spec
            pipeline_version = resources_pb2.PipelineVersion()
            # Create orchestration spec proto
            orchestration_spec_proto = resources_pb2.OrchestrationSpec()
            # Create Argo orchestration spec proto
            argo_orchestration_spec_proto = resources_pb2.ArgoOrchestrationSpec()
            argo_orchestration_spec_proto.api_version = api_version
            argo_orchestration_spec_proto.spec_json = argo_spec_str
            
            orchestration_spec_proto.argo_orchestration_spec.CopyFrom(argo_orchestration_spec_proto)
            pipeline_version.orchestration_spec.CopyFrom(orchestration_spec_proto)
            
            pipeline.pipeline_version.CopyFrom(pipeline_version)
            
            # Make the RPC call
            response = self.client.STUB.PostPipelines(
                service_pb2.PostPipelinesRequest(
                    user_app_id=self.client.user_app_id,
                    pipelines=[pipeline]
                )
            )
            
            if response.status.code == status_code_pb2.SUCCESS:
                logger.info(f"Successfully created pipeline {self.pipeline_id}")
                return True
            else:
                logger.error(f"Failed to create pipeline: {response.status.description}")
                logger.error(f"Details: {response.status.details}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return False


def upload_pipeline(config_path: str):
    """
    Upload a pipeline with associated pipeline steps to Clarifai.
    
    :param config_path: Path to the pipeline configuration file
    """
    try:
        builder = PipelineBuilder(config_path)
        
        logger.info(f"Starting pipeline upload from config: {config_path}")
        
        # Step 1: Upload pipeline steps
        if not builder.upload_pipeline_steps():
            logger.error("Failed to upload pipeline steps")
            sys.exit(1)
        
        # Step 2: Update config with version information
        builder.update_config_with_versions()
        
        # Step 3: Create the pipeline
        if not builder.create_pipeline():
            logger.error("Failed to create pipeline")
            sys.exit(1)
        
        logger.info("Pipeline upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline upload failed: {e}")
        sys.exit(1)