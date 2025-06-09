import os
import sys
import tarfile
import time
from string import Template

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.utils.logging import logger
from clarifai.versions import CLIENT_VERSION


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
            self._client = BaseClient()
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
            id=pipeline_step_config["id"],
            user_id=pipeline_step_config["user_id"]
        )

        return pipeline_step_proto

    def _get_pipeline_step_compute_info(self):
        """Get pipeline step compute info from config."""
        compute_config = self.config.get("pipeline_step_compute_info", {})

        compute_info = resources_pb2.ComputeInfo()

        if "cpu_limit" in compute_config:
            compute_info.cpu_limit = compute_config["cpu_limit"]
        if "cpu_memory" in compute_config:
            compute_info.cpu_memory = compute_config["cpu_memory"]
        if "num_accelerators" in compute_config:
            compute_info.num_accelerators = compute_config["num_accelerators"]

        return compute_info

    def check_pipeline_step_exists(self):
        """Check if pipeline step already exists."""
        try:
            resp = self.client.STUB.GetPipelineStep(
                service_pb2.GetPipelineStepRequest(
                    user_app_id=self.client.user_app_id,
                    pipeline_step_id=self.pipeline_step_id
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
                    param = resources_pb2.PipelineStepInputParam(
                        name=param_config["name"]
                    )
                    if "default" in param_config:
                        param.default_value = param_config["default"]
                    if "description" in param_config:
                        param.description = param_config["description"]
                    if "accepted_values" in param_config:
                        param.accepted_values.extend(param_config["accepted_values"])
                    input_params.append(param)

            pipeline_step = resources_pb2.PipelineStep(
                id=self.pipeline_step_id,
                user_id=self.pipeline_step_proto.user_id
            )

            resp = self.client.STUB.PostPipelineSteps(
                service_pb2.PostPipelineStepsRequest(
                    user_app_id=self.client.user_app_id,
                    pipeline_steps=[pipeline_step]
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

        # Write Dockerfile
        dockerfile_path = os.path.join(self.folder, 'Dockerfile')
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
            # Upload pipeline step version
            for response in self.client.STUB.PostPipelineStepVersionsUpload(
                self._pipeline_step_version_upload_iterator(file_path)
            ):
                percent_completed = getattr(response.status, 'percent_completed', 0)
                details = response.status.details

                print(
                    f"Status: {response.status.description}, Progress: {percent_completed}% - {details}",
                    f"request_id: {response.status.req_id}",
                    end='\r',
                    flush=True
                )

            if response.status.code != status_code_pb2.PIPELINE_STEP_BUILDING:
                logger.error(f"Failed to upload pipeline step version: {response}")
                return False

            self.pipeline_step_version_id = response.pipeline_step_version.id
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
            chunk_size = int(127 * 1024 * 1024)  # 127MB chunk size
            num_chunks = (file_size // chunk_size) + 1
            logger.info("Uploading pipeline step content...")
            logger.debug(f"File size: {file_size}")
            logger.debug(f"Chunk size: {chunk_size}")
            logger.debug(f"Number of chunks: {num_chunks}")

            read_so_far = 0
            for part_id in range(num_chunks):
                try:
                    chunk_size = min(chunk_size, file_size - read_so_far)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    read_so_far += len(chunk)
                    yield service_pb2.PostPipelineStepVersionsUploadRequest(
                        content_part=resources_pb2.UploadContentPart(
                            data=chunk,
                            part_number=part_id + 1,
                            range_start=read_so_far,
                        )
                    )
                except Exception as e:
                    logger.exception(f"\nError uploading file: {e}")
                    break

        if read_so_far == file_size:
            logger.info("Upload complete!")

    def _init_upload_pipeline_step_version(self, file_path):
        """Initialize the pipeline step version upload."""
        file_size = os.path.getsize(file_path)
        logger.debug(f"Uploading pipeline step version of pipeline step {self.pipeline_step_id}")
        logger.debug(f"Using file '{os.path.basename(file_path)}' of size: {file_size} bytes")

        # Build pipeline step input params
        input_params = []
        if "pipeline_step_input_params" in self.config:
            for param_config in self.config["pipeline_step_input_params"]:
                param = resources_pb2.PipelineStepInputParam(
                    name=param_config["name"]
                )
                if "default" in param_config:
                    param.default_value = param_config["default"]
                if "description" in param_config:
                    param.description = param_config["description"]
                if "accepted_values" in param_config:
                    param.accepted_values.extend(param_config["accepted_values"])
                input_params.append(param)

        # Create pipeline step version proto
        pipeline_step_config = self.config["pipeline_step"]
        pipeline_step_version = resources_pb2.PipelineStepVersion(
            pipeline_step=self.pipeline_step_proto,
            user_id=pipeline_step_config["user_id"],
            app_id=pipeline_step_config["app_id"],
            pipeline_step_input_params=input_params,
            pipeline_step_compute_info=self.pipeline_step_compute_info
        )

        # Build info from config
        build_info = self.config.get('build_info', {})
        if build_info:
            version_build_info = resources_pb2.BuildInfo()
            if 'python_version' in build_info:
                version_build_info.python_version = build_info['python_version']
            pipeline_step_version.build_info.CopyFrom(version_build_info)

        return service_pb2.PostPipelineStepVersionsUploadRequest(
            config=service_pb2.PostPipelineStepVersionsUploadConfig(
                user_app_id=self.client.user_app_id,
                pipeline_step_version=pipeline_step_version,
                total_size=file_size
            )
        )

    def _monitor_pipeline_step_build(self):
        """Monitor the pipeline step build process."""
        st = time.time()
        while True:
            try:
                resp = self.client.STUB.GetPipelineStepVersion(
                    service_pb2.GetPipelineStepVersionRequest(
                        user_app_id=self.client.user_app_id,
                        pipeline_step_id=self.pipeline_step_id,
                        version_id=self.pipeline_step_version_id,
                    )
                )

                status_code = resp.pipeline_step_version.status.code

                if status_code == status_code_pb2.PIPELINE_STEP_BUILDING:
                    print(
                        f"Pipeline step is building... (elapsed {time.time() - st:.1f}s)", 
                        end='\r', 
                        flush=True
                    )
                    time.sleep(1)
                elif status_code == status_code_pb2.PIPELINE_STEP_READY:
                    logger.info("\nPipeline step build complete!")
                    logger.info(f"Build time elapsed {time.time() - st:.1f}s")
                    return True
                else:
                    logger.error(
                        f"\nPipeline step build failed with status: {resp.pipeline_step_version.status}"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error monitoring pipeline step build: {e}")
                return False


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
