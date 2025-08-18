import os
import re
import sys
import tarfile
import time
from string import Template

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.client.base import BaseClient
from clarifai.utils.logging import logger
from clarifai.utils.misc import get_uuid
from clarifai.versions import CLIENT_VERSION

# Upload chunk size for pipeline step versions (14MB)
UPLOAD_CHUNK_SIZE = 14 * 1024 * 1024


class PipelineStepBuilder:
    """Pipeline Step Builder class for managing pipeline step upload to Clarifai."""

    def __init__(self, folder: str):
        """
        Initialize PipelineStepBuilder.

        :param folder: The folder containing the pipeline step files (config.yaml, requirements.txt,
                      dockerfile, and pipeline_step.py in 1/ subdirectory)
        """
        self._client = None
        self.folder = self._validate_folder(folder)
        self.config = self._load_config(os.path.join(self.folder, 'config.yaml'))
        self._validate_config()
        self.pipeline_step_proto = self._get_pipeline_step_proto()
        self.pipeline_step_id = self.pipeline_step_proto.id
        self.pipeline_step_version_id = None
        self.pipeline_step_compute_info = self._get_pipeline_step_compute_info()

    @property
    def client(self):
        """Get or create the Clarifai client."""
        if self._client is None:
            pipeline_step_config = self.config["pipeline_step"]
            user_id = pipeline_step_config["user_id"]
            app_id = pipeline_step_config["app_id"]
            self._client = BaseClient(user_id=user_id, app_id=app_id)
        return self._client

    def _validate_folder(self, folder):
        """Validate that the folder contains required files."""
        folder = os.path.abspath(folder)

        # Check for required files
        required_files = ['config.yaml']
        for file in required_files:
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file '{file}' not found in {folder}")

        # Check for pipeline_step.py in 1/ subdirectory
        pipeline_step_file = os.path.join(folder, '1', 'pipeline_step.py')
        if not os.path.exists(pipeline_step_file):
            raise FileNotFoundError(f"Required file '1/pipeline_step.py' not found in {folder}")

        return folder

    @staticmethod
    def _load_config(config_path):
        """Load and return the configuration from config.yaml."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config.yaml: {e}")

    def _validate_config(self):
        """Validate the configuration."""
        # Validate pipeline_step section
        if "pipeline_step" not in self.config:
            raise ValueError("pipeline_step section not found in config.yaml")

        pipeline_step = self.config["pipeline_step"]
        required_fields = ["id", "user_id", "app_id"]

        for field in required_fields:
            if field not in pipeline_step:
                raise ValueError(f"{field} not found in pipeline_step section of config.yaml")
            if not pipeline_step[field]:
                raise ValueError(f"{field} cannot be empty in config.yaml")

        # Validate pipeline_step_compute_info section
        if "pipeline_step_compute_info" not in self.config:
            raise ValueError("pipeline_step_compute_info section not found in config.yaml")

    def _get_pipeline_step_proto(self):
        """Create pipeline step proto from config."""
        pipeline_step_config = self.config["pipeline_step"]

        pipeline_step_proto = resources_pb2.PipelineStep(
            id=pipeline_step_config["id"], user_id=pipeline_step_config["user_id"]
        )

        return pipeline_step_proto

    def _get_pipeline_step_compute_info(self):
        """Get pipeline step compute info from config."""
        assert "pipeline_step_compute_info" in self.config, (
            "pipeline_step_compute_info not found in the config file"
        )
        compute_config = self.config.get("pipeline_step_compute_info", {})

        # Ensure cpu_limit is a string if it exists and is an int
        if 'cpu_limit' in compute_config and isinstance(compute_config['cpu_limit'], int):
            compute_config['cpu_limit'] = str(compute_config['cpu_limit'])

        compute_info = json_format.ParseDict(compute_config, resources_pb2.ComputeInfo())

        return compute_info

    def check_pipeline_step_exists(self):
        """Check if pipeline step already exists."""
        try:
            resp = self.client.STUB.GetPipelineStep(
                service_pb2.GetPipelineStepRequest(
                    user_app_id=self.client.user_app_id, pipeline_step_id=self.pipeline_step_id
                )
            )
            return resp.status.code == status_code_pb2.SUCCESS
        except Exception:
            return False

    def create_pipeline_step(self):
        """Create a new pipeline step if it doesn't exist."""
        if self.check_pipeline_step_exists():
            logger.info(f"Pipeline step {self.pipeline_step_id} already exists")
            return True

        try:
            # Build pipeline step input params
            input_params = []
            if "pipeline_step_input_params" in self.config:
                for param_config in self.config["pipeline_step_input_params"]:
                    param = resources_pb2.PipelineStepInputParam(name=param_config["name"])
                    if "default" in param_config:
                        param.default_value = param_config["default"]
                    if "description" in param_config:
                        param.description = param_config["description"]
                    if "accepted_values" in param_config:
                        param.accepted_values.extend(param_config["accepted_values"])
                    input_params.append(param)

            pipeline_step = resources_pb2.PipelineStep(
                id=self.pipeline_step_id, user_id=self.pipeline_step_proto.user_id
            )

            resp = self.client.STUB.PostPipelineSteps(
                service_pb2.PostPipelineStepsRequest(
                    user_app_id=self.client.user_app_id, pipeline_steps=[pipeline_step]
                )
            )

            if resp.status.code == status_code_pb2.SUCCESS:
                logger.info(f"Successfully created pipeline step {self.pipeline_step_id}")
                return True
            else:
                logger.error(f"Failed to create pipeline step: {resp.status}")
                return False

        except Exception as e:
            logger.error(f"Error creating pipeline step: {e}")
            return False

    def create_dockerfile(self):
        """Create a Dockerfile for the pipeline step."""
        # Use similar logic to model builder for dockerfile creation
        dockerfile_template = """FROM --platform=$TARGETPLATFORM public.ecr.aws/clarifai-models/python-base:$PYTHON_VERSION-df565436eea93efb3e8d1eb558a0a46df29523ec as final

COPY --link requirements.txt /home/nonroot/requirements.txt

# Update clarifai package so we always have latest protocol to the API. Everything should land in /venv
RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]

# Copy in the actual files like config.yaml, requirements.txt, and most importantly 1/pipeline_step.py for the actual pipeline step.
COPY --link=true 1 /home/nonroot/main/1
# At this point we only need these for validation in the SDK.
COPY --link=true requirements.txt config.yaml /home/nonroot/main/
"""

        # Get Python version from config or use default
        build_info = self.config.get('build_info', {})
        python_version = build_info.get('python_version', '3.12')

        # Ensure requirements.txt has clarifai
        self._ensure_clarifai_requirement()

        # Replace placeholders
        dockerfile_content = Template(dockerfile_template).safe_substitute(
            PYTHON_VERSION=python_version
        )

        # Write Dockerfile if it doesn't exist
        dockerfile_path = os.path.join(self.folder, 'Dockerfile')
        if os.path.exists(dockerfile_path):
            logger.info(f"Dockerfile already exists at {dockerfile_path}, skipping creation.")
            return
        with open(dockerfile_path, 'w') as dockerfile:
            dockerfile.write(dockerfile_content)

        logger.info(f"Created Dockerfile at {dockerfile_path}")

    def _ensure_clarifai_requirement(self):
        """Ensure clarifai is in requirements.txt with proper version."""
        requirements_path = os.path.join(self.folder, 'requirements.txt')

        # Read existing requirements
        requirements = []
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                requirements = f.readlines()

        # Check if clarifai is already present
        has_clarifai = any('clarifai' in line for line in requirements)

        if not has_clarifai:
            requirements.append(f'clarifai=={CLIENT_VERSION}\n')
            with open(requirements_path, 'w') as f:
                f.writelines(requirements)
            logger.info(f"Added clarifai=={CLIENT_VERSION} to requirements.txt")

    @property
    def tar_file(self):
        """Get the tar file path."""
        return f"{self.folder}.tar.gz"

    def upload_pipeline_step_version(self):
        """Upload a new version of the pipeline step."""
        # Ensure pipeline step exists
        if not self.create_pipeline_step():
            logger.error("Failed to create pipeline step")
            return False

        # Create tar file
        file_path = self.tar_file
        logger.debug(f"Creating tar file: {file_path}")

        def filter_func(tarinfo):
            name = tarinfo.name
            exclude = [os.path.basename(self.tar_file), "*~", "*.pyc", "*.pyo", "__pycache__"]
            return None if any(name.endswith(ex) for ex in exclude) else tarinfo

        with tarfile.open(file_path, "w:gz") as tar:
            tar.add(self.folder, arcname=".", filter=filter_func)

        logger.debug("Tar file creation complete")

        file_size = os.path.getsize(file_path)
        logger.debug(f"Tar file size: {file_size} bytes")

        try:
            # Upload pipeline step version with client-side progress tracking
            uploaded_bytes = 0
            chunk_count = 0
            total_chunks = (
                file_size + UPLOAD_CHUNK_SIZE - 1
            ) // UPLOAD_CHUNK_SIZE  # Ceiling division

            for response in self.client.STUB.PostPipelineStepVersionsUpload(
                self._pipeline_step_version_upload_iterator(file_path)
            ):
                # Calculate progress based on chunks uploaded
                if chunk_count == 0:
                    # First response is config upload, no progress yet
                    percent_completed = 0
                else:
                    # Calculate progress based on completed chunks
                    uploaded_bytes = min(chunk_count * UPLOAD_CHUNK_SIZE, file_size)
                    percent_completed = min(100, int((uploaded_bytes / file_size) * 100))

                chunk_count += 1

                print(
                    f"Status: {response.status.description}, Upload Progress: {percent_completed}%, Details: {response.status.details}",
                    f"request_id: {response.status.req_id}",
                    end='\r',
                    flush=True,
                )

            if response.status.code != status_code_pb2.PIPELINE_STEP_BUILDING:
                logger.error(f"Failed to upload pipeline step version: {response}")
                return False

            self.pipeline_step_version_id = response.pipeline_step_version_id
            logger.info(f"\nCreated Pipeline Step Version ID: {self.pipeline_step_version_id}")

            # Monitor build progress
            return self._monitor_pipeline_step_build()

        finally:
            # Clean up tar file
            if os.path.exists(file_path):
                logger.debug(f"Cleaning up tar file: {file_path}")
                os.remove(file_path)

    def _pipeline_step_version_upload_iterator(self, file_path):
        """Iterator for uploading pipeline step version in chunks."""
        # First yield the config
        yield self._init_upload_pipeline_step_version(file_path)

        # Then yield file content in chunks
        with open(file_path, "rb") as f:
            file_size = os.path.getsize(file_path)
            chunk_size = UPLOAD_CHUNK_SIZE
            logger.info("Uploading pipeline step content...")
            logger.debug(f"File size: {file_size}")
            logger.debug(f"Chunk size: {chunk_size}")

            offset = 0
            part_id = 1
            while offset < file_size:
                try:
                    current_chunk_size = min(chunk_size, file_size - offset)
                    chunk = f.read(current_chunk_size)
                    if not chunk:
                        break
                    yield service_pb2.PostPipelineStepVersionsUploadRequest(
                        content_part=resources_pb2.UploadContentPart(
                            data=chunk,
                            part_number=part_id,
                            range_start=offset,
                        )
                    )
                    offset += len(chunk)
                    part_id += 1
                except Exception as e:
                    logger.exception(f"\nError uploading file: {e}")
                    break

        if offset == file_size:
            logger.info("Upload complete!")

    def _get_tar_file_content_size(self, tar_file_path):
        """
        Calculates the total size of the contents of a tar file.

        Args:
          tar_file_path (str): The path to the tar file.

        Returns:
          int: The total size of the contents in bytes.
        """
        total_size = 0
        with tarfile.open(tar_file_path, 'r') as tar:
            for member in tar:
                if member.isfile():
                    total_size += member.size
        return total_size

    def _init_upload_pipeline_step_version(self, file_path):
        """Initialize the pipeline step version upload."""
        file_size = os.path.getsize(file_path)
        storage_request_size = self._get_tar_file_content_size(file_path)
        logger.debug(f"Uploading pipeline step version of pipeline step {self.pipeline_step_id}")
        logger.debug(f"Using file '{os.path.basename(file_path)}' of size: {file_size} bytes")
        logger.debug(f"Storage request size: {storage_request_size} bytes")

        # Build pipeline step input params
        input_params = []
        if "pipeline_step_input_params" in self.config:
            for param_config in self.config["pipeline_step_input_params"]:
                param = resources_pb2.PipelineStepInputParam(name=param_config["name"])
                if "default" in param_config:
                    param.default_value = param_config["default"]
                if "description" in param_config:
                    param.description = param_config["description"]
                if "accepted_values" in param_config:
                    param.accepted_values.extend(param_config["accepted_values"])
                input_params.append(param)

        # Create pipeline step version proto with generated ID
        version_id = get_uuid(16)  # Generate a 16-character UUID
        pipeline_step_version = resources_pb2.PipelineStepVersion(
            id=version_id,
            description="Pipeline step version",
            pipeline_step_input_params=input_params,
            pipeline_step_compute_info=self.pipeline_step_compute_info,
        )

        return service_pb2.PostPipelineStepVersionsUploadRequest(
            upload_config=service_pb2.PostPipelineStepVersionsUploadConfig(
                user_app_id=self.client.user_app_id,
                pipeline_step_id=self.pipeline_step_id,
                pipeline_step_version=pipeline_step_version,
                total_size=file_size,
                storage_request_size=storage_request_size,
            )
        )

    def _monitor_pipeline_step_build(self, timeout_sec=300, interval_sec=1):
        """
        Monitor the pipeline step build process with timeout and log display.

        :param timeout_sec: Maximum time to wait for build completion (default 300 seconds)
        :param interval_sec: Interval between status checks (default 1 second)
        :return: True if build successful, False otherwise
        """
        max_checks = timeout_sec // interval_sec
        seen_logs = set()  # To avoid duplicate log messages
        st = time.time()

        for _ in range(max_checks):
            print(
                f"Pipeline Step is building... (elapsed {time.time() - st:.1f}s)",
                end='\r',
                flush=True,
            )

            try:
                response = self.client.STUB.GetPipelineStepVersion(
                    service_pb2.GetPipelineStepVersionRequest(
                        user_app_id=self.client.user_app_id,
                        pipeline_step_id=self.pipeline_step_id,
                        pipeline_step_version_id=self.pipeline_step_version_id,
                    ),
                    metadata=self.client.auth_helper.metadata,
                )
                logger.debug(f"GetPipelineStepVersion Response: {response}")

                # Fetch and display build logs
                logs_request = service_pb2.ListLogEntriesRequest(
                    log_type="builder",
                    user_app_id=self.client.user_app_id,
                    pipeline_step_id=self.pipeline_step_id,
                    pipeline_step_version_id=self.pipeline_step_version_id,
                    page=1,
                    per_page=50,
                )
                logs = self.client.STUB.ListLogEntries(
                    logs_request, metadata=self.client.auth_helper.metadata
                )

                for log_entry in logs.log_entries:
                    if log_entry.url not in seen_logs:
                        seen_logs.add(log_entry.url)
                        log_entry_msg = re.sub(
                            r"(\\*)(\[[a-z#/@][^[]*?])",
                            lambda m: f"{m.group(1)}{m.group(1)}\\{m.group(2)}",
                            log_entry.message.strip(),
                        )
                        logger.info(log_entry_msg)

                status = response.pipeline_step_version.status.code
                if status in {
                    status_code_pb2.StatusCode.PIPELINE_STEP_READY,
                    status_code_pb2.StatusCode.PIPELINE_STEP_BUILDING_FAILED,
                    status_code_pb2.StatusCode.PIPELINE_STEP_BUILD_UNEXPECTED_ERROR,
                    status_code_pb2.StatusCode.INTERNAL_UNCATEGORIZED,
                }:
                    if status == status_code_pb2.StatusCode.PIPELINE_STEP_READY:
                        logger.info("\nPipeline step build complete!")
                        logger.info(f"Build time elapsed {time.time() - st:.1f}s")
                        return True
                    else:
                        logger.error(
                            f"\nPipeline step build failed with status: {response.pipeline_step_version.status}"
                        )
                        return False
                elif status != status_code_pb2.StatusCode.PIPELINE_STEP_BUILDING:
                    logger.error(
                        f"\nUnexpected status during pipeline step build: {response.pipeline_step_version.status}"
                    )
                    return False

                time.sleep(interval_sec)

            except Exception as e:
                logger.error(f"Error monitoring pipeline step build: {e}")
                return False

        raise TimeoutError("Pipeline step build did not finish in time")


def upload_pipeline_step(folder, skip_dockerfile=False):
    """
    Upload a pipeline step to Clarifai.

    :param folder: The folder containing the pipeline step files.
    :param skip_dockerfile: If True, will not create a Dockerfile.
    """
    builder = PipelineStepBuilder(folder)

    if not skip_dockerfile:
        builder.create_dockerfile()

    exists = builder.check_pipeline_step_exists()
    if exists:
        logger.info(
            f"Pipeline step {builder.pipeline_step_id} already exists, this upload will create a new version for it."
        )
    else:
        logger.info(
            f"New pipeline step {builder.pipeline_step_id} will be created with its first version."
        )

    input("Press Enter to continue...")

    success = builder.upload_pipeline_step_version()
    if success:
        logger.info("Pipeline step upload completed successfully!")
    else:
        logger.error("Pipeline step upload failed!")
        sys.exit(1)
