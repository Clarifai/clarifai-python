import json
import os
import sys
from typing import Any, Dict

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.runners.utils.pipeline_validation import PipelineConfigValidator
from clarifai.utils.logging import logger


class LiteralBlockDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses literal block style for multi-line strings."""

    def represent_str(self, data):
        if '\n' in data:
            return self.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return self.represent_scalar('tag:yaml.org,2002:str', data)


LiteralBlockDumper.add_representer(str, LiteralBlockDumper.represent_str)


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
            with open(self.config_path, 'w', encoding="utf-8") as file:
                yaml.dump(
                    self.config,
                    file,
                    Dumper=LiteralBlockDumper,
                    default_flow_style=False,
                    sort_keys=False,
                )
        except Exception as e:
            raise ValueError(f"Error saving config file {self.config_path}: {e}")

    def upload_pipeline_steps(self) -> bool:
        """Upload all pipeline steps listed in step_directories."""
        pipeline_config = self.config["pipeline"]
        step_directories = pipeline_config.get("step_directories", [])

        if not step_directories:
            logger.info("No pipeline steps to upload (step_directories is empty)")
            return False  # treat this as an error.

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
            logger.info(
                f"Successfully uploaded pipeline step {step_dir} with version {version_id}"
            )

        return True

    def _upload_pipeline_step_with_version_capture(self, step_path: str) -> tuple[bool, str]:
        """
        Upload a pipeline step and capture its version ID.
        Implements hash-based optimization to only upload modified steps.
        """
        try:
            # Use the existing pipeline step builder
            from clarifai.runners.pipeline_steps.pipeline_step_builder import PipelineStepBuilder

            builder = PipelineStepBuilder(step_path)

            # Check if we should upload based on hash comparison
            should_upload = builder.should_upload_step()

            if not should_upload:
                # Load existing version ID from config-lock.yaml
                config_lock = builder.load_config_lock()
                if config_lock and config_lock.get("id"):
                    version_id = config_lock["id"]
                    logger.info(
                        f"Using existing pipeline step version {version_id} (no changes detected)"
                    )
                    return True, version_id
                else:
                    logger.warning(
                        "Hash indicates no upload needed, but no version ID found in config-lock.yaml. Proceeding with upload."
                    )

            # Create dockerfile if needed
            builder.create_dockerfile()

            # Check if step exists
            exists = builder.check_pipeline_step_exists()
            if exists:
                logger.info(
                    f"Pipeline step {builder.pipeline_step_id} already exists, creating new version"
                )
            else:
                logger.info(f"Creating new pipeline step {builder.pipeline_step_id}")

            # Upload the pipeline step version directly without the interactive prompt
            success = builder.upload_pipeline_step_version()

            if success and builder.pipeline_step_version_id:
                # Generate config-lock.yaml with the new version ID
                builder.save_config_lock(builder.pipeline_step_version_id)
                logger.info(
                    f"Generated config-lock.yaml for pipeline step with version {builder.pipeline_step_version_id}"
                )
                return True, builder.pipeline_step_version_id
            else:
                logger.error("Failed to get pipeline step version ID after upload")
                return False, ""

        except Exception as e:
            logger.error(f"Error uploading pipeline step: {e}")
            return False, ""

    def prepare_lockfile_with_step_versions(self) -> Dict[str, Any]:
        """Prepare lockfile data with step versions after pipeline step upload."""
        if not self.uploaded_step_versions:
            logger.info("No pipeline step versions for lockfile")

        # Create a copy of the orchestration spec to modify
        pipeline_config = self.config["pipeline"]
        orchestration_spec = pipeline_config["orchestration_spec"].copy()
        argo_spec_str = orchestration_spec["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)

        # Update templateRef names to include versions
        self._update_template_refs_with_versions(argo_spec)

        # Create the partial lockfile data structure (without pipeline info)
        lockfile_data = {
            "pipeline": {
                "id": self.pipeline_id,
                "user_id": self.user_id,
                "app_id": self.app_id,
                "version_id": None,  # Will be filled in later
                "orchestration_spec": {
                    "argo_orchestration_spec": yaml.dump(
                        argo_spec, Dumper=LiteralBlockDumper, default_flow_style=False
                    )
                },
            }
        }

        # Include step_version_secrets if present in pipeline config (not orchestration_spec)
        step_version_secrets = pipeline_config.get("config", {}).get("step_version_secrets", {})
        if step_version_secrets:
            if "config" not in lockfile_data["pipeline"]:
                lockfile_data["pipeline"]["config"] = {}
            lockfile_data["pipeline"]["config"]["step_version_secrets"] = step_version_secrets

        return lockfile_data

    def update_lockfile_with_pipeline_info(
        self, lockfile_data: Dict[str, Any], pipeline_version_id: str
    ) -> Dict[str, Any]:
        """Update the prepared lockfile data with pipeline version information."""
        lockfile_data["pipeline"]["version_id"] = pipeline_version_id
        return lockfile_data

    def generate_lockfile_data(
        self, pipeline_id: str = None, pipeline_version_id: str = None
    ) -> Dict[str, Any]:
        """Generate the complete lockfile data structure without modifying config.yaml.

        This method is kept for backward compatibility. The recommended approach is to use
        prepare_lockfile_with_step_versions() followed by update_lockfile_with_pipeline_info().
        """
        if not self.uploaded_step_versions:
            logger.info("No pipeline step versions for lockfile")

        # Create a copy of the orchestration spec to modify
        pipeline_config = self.config["pipeline"]
        orchestration_spec = pipeline_config["orchestration_spec"].copy()
        argo_spec_str = orchestration_spec["argo_orchestration_spec"]
        argo_spec = yaml.safe_load(argo_spec_str)

        # Update templateRef names to include versions
        self._update_template_refs_with_versions(argo_spec)

        # Create the lockfile data structure
        lockfile_data = {
            "pipeline": {
                "id": pipeline_id or self.pipeline_id,
                "user_id": self.user_id,
                "app_id": self.app_id,
                "version_id": pipeline_version_id,
                "orchestration_spec": {
                    "argo_orchestration_spec": yaml.dump(
                        argo_spec, Dumper=LiteralBlockDumper, default_flow_style=False
                    )
                },
            }
        }

        # Include step_version_secrets if present in pipeline config (not orchestration_spec)
        step_version_secrets = pipeline_config.get("config", {}).get("step_version_secrets", {})
        if step_version_secrets:
            if "config" not in lockfile_data["pipeline"]:
                lockfile_data["pipeline"]["config"] = {}
            lockfile_data["pipeline"]["config"]["step_version_secrets"] = step_version_secrets

        return lockfile_data

    def save_lockfile(self, lockfile_data: Dict[str, Any], lockfile_path: str = None) -> None:
        """Save lockfile data to config-lock.yaml."""
        if lockfile_path is None:
            lockfile_path = os.path.join(self.config_dir, "config-lock.yaml")

        try:
            with open(lockfile_path, 'w', encoding="utf-8") as file:
                yaml.dump(
                    lockfile_data,
                    file,
                    Dumper=LiteralBlockDumper,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"Generated lockfile: {lockfile_path}")
        except Exception as e:
            raise ValueError(f"Error saving lockfile {lockfile_path}: {e}")

    def _update_template_refs_with_versions(self, argo_spec: Dict[str, Any]) -> None:
        """
        Update templateRef names in Argo spec to include version information.
        The step versions should be resolved from the corresponding config-lock.yaml
        file of each pipeline-step, located in the step_directories.
        """
        for template in argo_spec["spec"]["templates"]:
            if "steps" in template:
                for step_group in template["steps"]:
                    for step in step_group:
                        if "templateRef" in step:
                            template_ref = step["templateRef"]
                            name = template_ref["name"]
                            # Extract step name
                            parts = name.split('/')

                            # Check if this is a templateRef without version that we uploaded
                            if self.validator.TEMPLATE_REF_WITHOUT_VERSION_PATTERN.match(name):
                                step_name = parts[-1]
                                # The step name should match the directory name or be derivable from it
                                version_id = self.uploaded_step_versions.get(step_name, None)

                                # If not found in uploaded_step_versions, try to get from config-lock.yaml
                                if version_id is None:
                                    version_id = self._get_version_from_config_lock(step_name)

                                if version_id is not None:
                                    # Update the templateRef to include version
                                    new_name = f"{name}/versions/{version_id}"
                                    template_ref["name"] = new_name
                                    template_ref["template"] = new_name
                                    logger.info(f"Updated templateRef from {name} to {new_name}")
                                else:
                                    logger.warning(f"Could not find version for step: {step_name}")
                            elif self.validator.TEMPLATE_REF_WITH_VERSION_PATTERN.match(name):
                                # strip the /versions/{version_id} from the end of name
                                # to get the name like above
                                orig_name = name
                                name = orig_name.rsplit('/versions/', 1)[0]
                                step_name = parts[-3]  # Get the step name from the path

                                # if it already has a version, make sure it matches the uploaded
                                # version
                                version_id = self.uploaded_step_versions.get(step_name, None)

                                # If not found in uploaded_step_versions, try to get from config-lock.yaml
                                if version_id is None:
                                    version_id = self._get_version_from_config_lock(step_name)

                                if version_id is not None:
                                    # Update the templateRef to include version
                                    new_name = f"{name}/versions/{version_id}"
                                    template_ref["name"] = new_name
                                    template_ref["template"] = new_name
                                    logger.info(
                                        f"Updated templateRef from {orig_name} to {new_name}"
                                    )
                                else:
                                    logger.warning(f"Could not find version for step: {step_name}")

    def _get_version_from_config_lock(self, step_name: str) -> str:
        """
        Get version ID from config-lock.yaml file in the corresponding step directory.

        :param step_name: Name of the pipeline step
        :return: Version ID if found, None otherwise
        """
        pipeline_config = self.config["pipeline"]
        step_directories = pipeline_config.get("step_directories", [])

        for step_dir in step_directories:
            # Check if step_dir matches step_name (handle both exact match and derivable cases)
            if (
                step_dir == step_name
                or step_dir.endswith(f"/{step_name}")
                or step_name in step_dir
            ):
                config_lock_path = os.path.join(self.config_dir, step_dir, "config-lock.yaml")

                if os.path.exists(config_lock_path):
                    try:
                        with open(config_lock_path, 'r', encoding='utf-8') as f:
                            config_lock = yaml.safe_load(f)
                            version_id = config_lock.get("id")
                            if version_id:
                                logger.info(
                                    f"Found version {version_id} for step {step_name} in {config_lock_path}"
                                )
                                return version_id
                    except Exception as e:
                        logger.warning(
                            f"Failed to read config-lock.yaml at {config_lock_path}: {e}"
                        )

        return None

    def _add_step_version_secrets(
        self, pipeline_version: resources_pb2.PipelineVersion, step_version_secrets: Dict[str, Any]
    ) -> None:
        """Add step_version_secrets to the pipeline version config.

        Args:
            pipeline_version: The PipelineVersion proto to update
            step_version_secrets: Dictionary mapping step references to their secret configs
                                 Format: {step_ref: {secret_name: secret_path}}
        """
        for step_ref, step_config in step_version_secrets.items():
            # Note: 'step_config' contains the secret mappings directly (not nested under 'secrets')
            # Secret references are like "users/user123/secrets/my-api-key"
            if not step_config:
                continue

            # Create Struct for the step secrets (new proto format)
            # Using google.protobuf.Struct to create flat JSON structure: {secretName: secretPath}
            from google.protobuf.struct_pb2 import Struct

            step_secrets_struct = Struct()
            step_secrets_struct.update(step_config)

            # Add to pipeline version config using the new proto format
            pipeline_version.config.step_version_secrets[step_ref].CopyFrom(step_secrets_struct)

    def create_pipeline(self) -> tuple[bool, str]:
        """Create the pipeline using PostPipelines RPC.

        Returns:
            tuple[bool, str]: (success, pipeline_version_id)
        """
        logger.info(f"Creating pipeline {self.pipeline_id}...")

        try:
            # Create pipeline proto
            pipeline = resources_pb2.Pipeline(
                id=self.pipeline_id, user_id=self.user_id, app_id=self.app_id
            )

            # Add orchestration spec
            pipeline_config = self.config["pipeline"]
            orchestration_spec = pipeline_config["orchestration_spec"]
            argo_spec_str = orchestration_spec["argo_orchestration_spec"]

            # Parse the Argo spec to get API version
            argo_spec = yaml.safe_load(argo_spec_str)
            api_version = argo_spec.get("apiVersion", "argoproj.io/v1alpha1")

            # Ensure that pipeline_config.argo_orchestration_spec_proto has the updated spec.templates.steps.templateRef values
            # For each step, if the templateRef is missing a version, append the correct version at the end
            # The step versions should be resolved from the corresponding config-lock.yaml file of each pipeline-step, located in the step_directories
            self._update_template_refs_with_versions(argo_spec)

            # Create pipeline version with orchestration spec
            pipeline_version = resources_pb2.PipelineVersion()
            # Create orchestration spec proto
            orchestration_spec_proto = resources_pb2.OrchestrationSpec()
            # Create Argo orchestration spec proto
            argo_orchestration_spec_proto = resources_pb2.ArgoOrchestrationSpec()
            argo_orchestration_spec_proto.api_version = api_version

            argo_spec_json_str = json.dumps(argo_spec)

            # Validate JSON string before setting
            try:
                # Test that we can parse it back
                json.loads(argo_spec_json_str)
            except json.JSONDecodeError as json_error:
                logger.error(f"Argo spec JSON validation failed: {json_error}")
                raise

            argo_orchestration_spec_proto.spec_json = argo_spec_json_str

            orchestration_spec_proto.argo_orchestration_spec.CopyFrom(
                argo_orchestration_spec_proto
            )
            pipeline_version.orchestration_spec.CopyFrom(orchestration_spec_proto)

            # Add step_version_secrets if present in pipeline config (not orchestration_spec)
            step_version_secrets = pipeline_config.get("config", {}).get(
                "step_version_secrets", {}
            )
            if step_version_secrets:
                self._add_step_version_secrets(pipeline_version, step_version_secrets)

            pipeline.pipeline_version.CopyFrom(pipeline_version)

            # Make the RPC call
            response = self.client.STUB.PostPipelines(
                service_pb2.PostPipelinesRequest(
                    user_app_id=self.client.user_app_id, pipelines=[pipeline]
                )
            )

            if response.status.code == status_code_pb2.SUCCESS:
                logger.info(f"Successfully created pipeline {self.pipeline_id}")

                pipeline_version_id = ""
                # Log pipeline and version IDs if available in response
                if response.pipelines:
                    created_pipeline = response.pipelines[0]
                    logger.info(f"Pipeline ID: {created_pipeline.id}")
                    if created_pipeline.pipeline_version and created_pipeline.pipeline_version.id:
                        pipeline_version_id = created_pipeline.pipeline_version.id
                        logger.info(f"Pipeline version ID: {pipeline_version_id}")

                return True, pipeline_version_id
            else:
                logger.error(f"Failed to create pipeline: {response.status.description}")
                logger.error(f"Details: {response.status.details}")
                return False, ""

        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return False, ""


def upload_pipeline(path: str, no_lockfile: bool = False):
    """
    Upload a pipeline with associated pipeline steps to Clarifai.

    :param path: Path to the pipeline configuration file or directory containing config.yaml
    :param no_lockfile: If True, skip creating config-lock.yaml
    """
    try:
        # Determine if path is a directory or file
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.yaml not found in directory: {path}")
        else:
            config_path = path

        builder = PipelineBuilder(config_path)

        logger.info(f"Starting pipeline upload from config: {config_path}")

        # Step 1: Upload pipeline steps
        if not builder.upload_pipeline_steps():
            logger.error("Failed to upload pipeline steps")
            sys.exit(1)

        # Step 2: Generate lockfile (unless --no-lockfile is specified)
        # This will be used to update the versions of pipeline-steps that just got uploaded in Step 1
        lockfile_data = None
        if not no_lockfile:
            lockfile_data = builder.prepare_lockfile_with_step_versions()

        # Step 3: Create the pipeline
        success, pipeline_version_id = builder.create_pipeline()
        if not success:
            logger.error("Failed to create pipeline")
            sys.exit(1)

        # Step 4: Update lockfile (unless --no-lockfile is specified)
        if not no_lockfile and lockfile_data:
            lockfile_data = builder.update_lockfile_with_pipeline_info(
                lockfile_data, pipeline_version_id
            )
            builder.save_lockfile(lockfile_data)
            logger.info("Pipeline upload completed successfully with lockfile!")
        else:
            logger.info("Pipeline upload completed successfully (lockfile skipped)!")

    except Exception as e:
        logger.error(f"Pipeline upload failed: {e}")
        sys.exit(1)
