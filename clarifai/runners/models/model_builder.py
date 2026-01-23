import builtins
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from string import Template
from typing import Any, Dict, Literal, Optional
from unittest.mock import MagicMock

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.client import Model, Nodepool
from clarifai.client.base import BaseClient
from clarifai.client.user import User
from clarifai.errors import UserError
from clarifai.runners.models.agentic_class import AgenticModelClass
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils import code_script
from clarifai.runners.utils.const import (
    AMD_PYTHON_BASE_IMAGE,
    AMD_TORCH_BASE_IMAGE,
    AMD_VLLM_BASE_IMAGE,
    AVAILABLE_PYTHON_IMAGES,
    AVAILABLE_TORCH_IMAGES,
    CONCEPTS_REQUIRED_MODEL_TYPE,
    DEFAULT_AMD_GPU_VERSION,
    DEFAULT_AMD_TORCH_VERSION,
    DEFAULT_DOWNLOAD_CHECKPOINT_WHEN,
    DEFAULT_PYTHON_VERSION,
    DEFAULT_RUNTIME_DOWNLOAD_PATH,
    PYTHON_BASE_IMAGE,
    TORCH_BASE_IMAGE,
)
from clarifai.runners.utils.loader import HuggingFaceLoader
from clarifai.runners.utils.method_signatures import signatures_to_yaml
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import logger
from clarifai.versions import get_latest_version_from_pypi

CLARIFAI_LATEST_VERSION = get_latest_version_from_pypi()

# Additional package installation if the model will be used w/ a streaming video runner:
# Dockerfile: Install ffmpeg and av
#
# Our base images are distroless, so we do not have apt-get or other package managers
# available; however, we will also not be able to use those package repositories on-prem.
# As a result, we build our own static ffmpeg image to serve as the source of these deps.
# See: https://github.com/Clarifai/models-images/tree/main/static_streaming
#
# TODO: before we make this public, we need to figure out how to distribute the src;
# line to copy in src commented out because it's 500MB
STREAMING_VIDEO_ADDITIONAL_PACKAGE_INSTALLATION = """
COPY --from=public.ecr.aws/clarifai-models/static-streaming:5.1.8 /ffmpeg /usr/local/bin/
COPY --from=public.ecr.aws/clarifai-models/static-streaming:5.1.8 /ffprobe /usr/local/bin/
# COPY --from=public.ecr.aws/clarifai-models/static-streaming:5.1.8 /src /usr/local/src/
RUN uv pip install --no-cache-dir av
"""

# parse the user's requirements.txt to determine the proper base image to build on top of, based on the torch and other large dependencies and it's versions
# List of dependencies to look for
dependencies = [
    'torch',
    'clarifai',
    'vllm',
]


def is_related(object_class, main_class):
    # Check if the object_class is a subclass of main_class
    if issubclass(object_class, main_class):
        return True

    # Check if the object_class is a subclass of any of the parent classes of main_class
    parent_classes = object_class.__bases__
    for parent in parent_classes:
        if main_class in parent.__bases__:
            return True
    return False


def get_user_input(prompt, required=True, default=None):
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        value = input(prompt).strip()
        if not value and default:
            return default
        if not value and required:
            print("❌ This field is required. Please enter a value.")
            continue
        return value


def get_yes_no_input(prompt, default=None):
    """Get yes/no input from user."""
    if default is not None:
        prompt = f"{prompt} [{'Y/n' if default else 'y/N'}]: "
    else:
        prompt = f"{prompt} [y/n]: "

    while True:
        response = input(prompt).strip().lower()
        if not response and default is not None:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'.")


def select_compute_option(user_id: str):
    """
    Dynamically list compute-clusters and node-pools that belong to `user_id`
    and return a dict with nodepool_id, compute_cluster_id, cluster_user_id.
    """
    user = User(user_id=user_id)  # PAT / BASE URL are picked from env-vars
    clusters = list(user.list_compute_clusters())
    if not clusters:
        print("❌ No compute clusters found for this user.")
        return None
    print("\n🖥️  Available Compute Clusters:")
    for idx, cc in enumerate(clusters, 1):
        desc = getattr(cc, "description", "") or "No description"
        print(f"{idx}. {cc.id}  –  {desc}")
    while True:
        try:
            sel = int(input("Select compute cluster (number): ")) - 1
            if 0 <= sel < len(clusters):
                cluster = clusters[sel]
                break
            print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a number.")
    nodepools = list(cluster.list_nodepools())
    if not nodepools:
        print("❌ No nodepools in selected cluster.")
        return None
    print("\n📦  Available Nodepools:")
    for idx, np in enumerate(nodepools, 1):
        desc = getattr(np, "description", "") or "No description"
        print(f"{idx}. {np.id}  –  {desc}")
    while True:
        try:
            sel = int(input("Select nodepool (number): ")) - 1
            if 0 <= sel < len(nodepools):
                nodepool = nodepools[sel]
                break
            print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a number.")
    return {
        "nodepool_id": nodepool.id,
        "compute_cluster_id": cluster.id,
        "cluster_user_id": getattr(cluster, "user_id", user_id),
    }


class ModelBuilder:
    DEFAULT_CHECKPOINT_SIZE = 50 * 1024**3  # 50 GiB

    def __init__(
        self,
        folder: str,
        validate_api_ids: bool = True,
        download_validation_only: bool = False,
        app_not_found_action: Literal["auto_create", "prompt", "error"] = "error",
        platform: Optional[str] = None,
        pat: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        :param folder: The folder containing the model.py, config.yaml, requirements.txt and
        checkpoints.
        :param validate_api_ids: Whether to validate the user_id and app_id in the config file. TODO(zeiler):
        deprecate in favor of download_validation_only.
        :param download_validation_only: Whether to skip the API config validation. Set to True if
        just downloading a checkpoint.
        :param app_not_found_action: Defines how to handle the case when the app is not found.
        Options: 'auto_create' - create automatically, 'prompt' - ask user, 'error' - raise exception.
        :param platform: Target platform(s) for Docker image build (e.g., "linux/amd64" or "linux/amd64,linux/arm64"). This overrides the platform specified in config.yaml.
        :param pat: Personal access token for authentication. If None, will use environment variables.
        :param base_url: Base URL for the API. If None, will use environment variables.
        """
        assert app_not_found_action in ["auto_create", "prompt", "error"], ValueError(
            f"Expected one of {['auto_create', 'prompt', 'error']}, got {app_not_found_action=}"
        )
        self.app_not_found_action = app_not_found_action
        self._client = None
        self._pat = pat
        self._base_url = base_url
        self._cli_platform = platform
        if not validate_api_ids:  # for backwards compatibility
            download_validation_only = True
        self.download_validation_only = download_validation_only
        self.folder = self._validate_folder(folder)
        self.config = self._load_config(os.path.join(self.folder, 'config.yaml'))
        self._validate_config()
        self._validate_config_secrets()
        self._validate_stream_options()
        self.model_proto = self._get_model_proto()
        self.model_id = self.model_proto.id
        self.model_version_id = None
        self.inference_compute_info = self._get_inference_compute_info()
        self.is_v3 = True  # Do model build for v3

    def create_model_instance(self, load_model=True, mocking=False) -> ModelClass:
        """
        Create an instance of the model class, as specified in the config file.
        """
        model_class = self.load_model_class(mocking=mocking)

        # initialize the model
        model = model_class()
        if load_model:
            model.load_model()
        return model

    def get_model_proto(self) -> resources_pb2.Model:
        """
        Retrieve the model and model version proto using self.model_id and self.model_version_id.

        Args:
            None

        Returns:
            resources_pb2.Model: The retrieved model proto.

        Raises:
            UserError: If the model or model version cannot be retrieved.
        """
        request = service_pb2.GetModelRequest(
            user_app_id=self.client.user_app_id,
            model_id=self.model_id,
        )
        # Add secrets to additional_fields to get request-type secrets
        request.additional_fields.append("secrets")
        if self.model_version_id is not None:
            request.version_id = self.model_version_id
        resp: service_pb2.SingleModelResponse = self.client.STUB.GetModel(request)
        if resp.status.code != status_code_pb2.SUCCESS:
            if self.model_version_id is None:
                raise UserError(f"Failed to get model '{self.model_id}': {resp.status.details}")
            else:
                raise UserError(
                    f"Failed to get model '{self.model_id}'"
                    f" version '{self.model_version_id}': {resp.status.details}"
                )
        return resp.model

    def load_model_class(self, mocking=False):
        """
        Import the model class from the model.py file, dynamically handling missing dependencies
        """
        # look for default model.py file location
        for loc in ["model.py", "1/model.py"]:
            model_file = os.path.join(self.folder, loc)
            if os.path.exists(model_file):
                break
        if not os.path.exists(model_file):
            raise Exception("Model file not found.")

        module_name = os.path.basename(model_file).replace(".py", "")

        spec = importlib.util.spec_from_file_location(module_name, model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        original_import = builtins.__import__
        # Prevent __pycache__ folder generation during module execution
        original_dont_write_bytecode = sys.dont_write_bytecode

        def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Allow standard libraries and clarifai
            if self._is_standard_or_clarifai(name):
                return original_import(name, globals, locals, fromlist, level)

            # Mock all third-party imports to avoid ImportErrors or other issues
            return MagicMock()

        if mocking:
            # Replace the built-in __import__ function with our custom one
            builtins.__import__ = custom_import

        try:
            # Set sys.dont_write_bytecode to prevent __pycache__ folder generation
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Error loading model.py: {e}")
            raise
        finally:
            # Restore the original __import__ function and bytecode setting
            builtins.__import__ = original_import
            sys.dont_write_bytecode = original_dont_write_bytecode

        # Find all classes in the model.py file that are subclasses of ModelClass
        classes = [
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if is_related(cls, ModelClass) and cls.__module__ == module.__name__
        ]
        #  Ensure there is exactly one subclass of BaseRunner in the model.py file
        if len(classes) != 1:
            # check for old inheritence structure, ModelRunner used to be a ModelClass
            runner_classes = [
                cls
                for _, cls in inspect.getmembers(module, inspect.isclass)
                if cls.__module__ == module.__name__
                and any(c.__name__ == 'ModelRunner' for c in cls.__bases__)
            ]
            if runner_classes and len(runner_classes) == 1:
                raise Exception(
                    f'Could not determine model class.'
                    f' Models should now inherit from {ModelClass.__module__}.ModelClass, not ModelRunner.'
                    f' Please update your model "{runner_classes[0].__name__}" to inherit from ModelClass.'
                )
            raise Exception(
                "Could not determine model class. There should be exactly one model inheriting from ModelClass defined in the model.py"
            )
        model_class = classes[0]
        return model_class

    def _is_standard_or_clarifai(self, name):
        """Check if import is from standard library or clarifai"""
        if name.startswith("clarifai"):
            return True

        # Handle Python <3.10 compatibility
        stdlib_names = getattr(sys, "stdlib_module_names", sys.builtin_module_names)
        if name in stdlib_names:
            return True

        # Handle submodules (e.g., os.path)
        parts = name.split(".")
        for i in range(1, len(parts)):
            if ".".join(parts[:i]) in stdlib_names:
                return True
        return False

    def _validate_folder(self, folder):
        if folder == ".":
            folder = ""  # will getcwd() next which ends with /
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)
        logger.debug(f"Validating folder: {folder}")
        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"Folder {folder} not found, please provide a valid folder path"
            )
        files = os.listdir(folder)
        assert "config.yaml" in files, "config.yaml not found in the folder"
        # If just downloading we don't need requirements.txt or the python code, we do need the
        # 1/ folder to put 1/checkpoints into.
        assert "1" in files, "Subfolder '1' not found in the folder"
        if not self.download_validation_only:
            assert "requirements.txt" in files, "requirements.txt not found in the folder"
            subfolder_files = os.listdir(os.path.join(folder, '1'))
            assert 'model.py' in subfolder_files, "model.py not found in the folder"
        return folder

    @staticmethod
    def _load_config(config_file: str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def _backup_config(config_file: str):
        if not os.path.exists(config_file):
            return
        backup_file = config_file + ".bak"
        if os.path.exists(backup_file):
            raise FileExistsError(
                f"Backup file {backup_file} already exists. Please remove it before proceeding."
            )
        shutil.copy(config_file, backup_file)

    @staticmethod
    def _save_config(config_file: str, config: dict):
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f)

    def _validate_config_checkpoints(self):
        """
        Validates the checkpoints section in the config file.
          return loader_type, repo_id, hf_token, when, allowed_file_patterns, ignore_file_patterns
        :return: loader_type the type of loader or None if no checkpoints.
        :return: repo_id location of checkpoint.
        :return: hf_token token to access checkpoint.
        :return: when one of ['upload', 'build', 'runtime'] to download checkpoint
        :return: allowed_file_patterns patterns to allow in downloaded checkpoint
        :return: ignore_file_patterns patterns to ignore in downloaded checkpoint
        """
        if "checkpoints" not in self.config:
            return None, None, None, DEFAULT_DOWNLOAD_CHECKPOINT_WHEN, None, None
        assert "type" in self.config.get("checkpoints"), (
            "No loader type specified in the config file"
        )
        loader_type = self.config.get("checkpoints").get("type")
        if not loader_type:
            logger.info("No loader type specified in the config file for checkpoints")
            return None, None, None, DEFAULT_DOWNLOAD_CHECKPOINT_WHEN, None, None
        checkpoints = self.config.get("checkpoints")
        if 'when' not in checkpoints:
            logger.warn(
                f"No 'when' specified in the config file for checkpoints, defaulting to download at {DEFAULT_DOWNLOAD_CHECKPOINT_WHEN}"
            )
        when = checkpoints.get("when", DEFAULT_DOWNLOAD_CHECKPOINT_WHEN)
        assert when in [
            "upload",
            "build",
            "runtime",
        ], (
            "Invalid value for when in the checkpoint loader when, needs to be one of ['upload', 'build', 'runtime']"
        )
        assert loader_type == "huggingface", "Only huggingface loader supported for now"
        if loader_type == "huggingface":
            assert "repo_id" in self.config.get("checkpoints"), (
                "No repo_id specified in the config file"
            )
            repo_id = self.config.get("checkpoints").get("repo_id")

            # get from config.yaml otherwise fall back to HF_TOKEN env var.
            hf_token = self.config.get("checkpoints").get(
                "hf_token", os.environ.get("HF_TOKEN", None)
            )

            allowed_file_patterns = self.config.get("checkpoints").get(
                'allowed_file_patterns', None
            )
            if isinstance(allowed_file_patterns, str):
                allowed_file_patterns = [allowed_file_patterns]
            ignore_file_patterns = self.config.get("checkpoints").get('ignore_file_patterns', None)
            if isinstance(ignore_file_patterns, str):
                ignore_file_patterns = [ignore_file_patterns]
            return (
                loader_type,
                repo_id,
                hf_token,
                when,
                allowed_file_patterns,
                ignore_file_patterns,
            )

    def _check_app_exists(self):
        resp = self.client.STUB.GetApp(
            service_pb2.GetAppRequest(user_app_id=self.client.user_app_id)
        )
        if resp.status.code == status_code_pb2.SUCCESS:
            return True
        if resp.status.code == status_code_pb2.CONN_KEY_INVALID:
            logger.error(
                f"Invalid PAT provided for user {self.client.user_app_id.user_id}. Please check your PAT and try again."
            )
            return False

        user_id = self.client.user_app_id.user_id
        app_id = self.client.user_app_id.app_id

        if self.app_not_found_action == "error":
            logger.error(
                f"Error checking API {self._base_api} for user app `{user_id}/{app_id}`. Error code: {resp.status.code}"
            )
            logger.error(
                f"App `{app_id}` not found for user `{user_id}`. Please create the app first and try again."
            )
            return False
        else:
            user = User(
                user_id=user_id,
                pat=self.client.pat,
                token=self.client.token,
                base_url=self.client.base,
            )

            def create_app():
                logger.info(f"Creating App `{app_id}` user `{user_id}`.")
                user.create_app(app_id=app_id)

            logger.info(f"App {app_id} not found for user {user_id}.")

            if self.app_not_found_action == "prompt":
                create_app_prompt = input(f"Do you want to create App `{app_id}`? (y/n): ")
                if create_app_prompt.lower() == 'y':
                    create_app()
                    return True
                else:
                    logger.error(
                        f"App `{app_id}` has not been created for user `{user_id}`. Please create the app first or switch to an existing one, then try again."
                    )
                    return False

            elif self.app_not_found_action == "auto_create":
                create_app()
                return True

    def _validate_config_model(self):
        assert "model" in self.config, "model section not found in the config file"
        model = self.config.get('model')
        assert "user_id" in model, "user_id not found in the config file"
        assert "app_id" in model, "app_id not found in the config file"
        assert "model_type_id" in model, "model_type_id not found in the config file"
        assert "id" in model, "model_id not found in the config file"
        if '.' in model.get('id'):
            logger.error(
                "Model ID cannot contain '.', please remove it from the model_id in the config file"
            )
            sys.exit(1)

        assert model.get('user_id') != "", "user_id cannot be empty in the config file"
        assert model.get('app_id') != "", "app_id cannot be empty in the config file"
        assert model.get('model_type_id') != "", "model_type_id cannot be empty in the config file"
        assert model.get('id') != "", "model_id cannot be empty in the config file"

        if not self._check_app_exists():
            sys.exit(1)

    @staticmethod
    def _set_local_runner_model(config, user_id, app_id, model_id, model_type_id):
        """
        Sets the model configuration for local development.
        This is used when running the model locally without uploading it to Clarifai.
        """
        if 'model' not in config:
            config['model'] = {}
        config["model"]["user_id"] = user_id
        config["model"]["app_id"] = app_id
        config["model"]["id"] = model_id
        config["model"]["model_type_id"] = model_type_id
        return config

    def _validate_config(self):
        if not self.download_validation_only:
            self._validate_config_model()

            assert "inference_compute_info" in self.config, (
                "inference_compute_info not found in the config file"
            )

            if self.config.get("concepts"):
                model_type_id = self.config.get('model').get('model_type_id')
                assert model_type_id in CONCEPTS_REQUIRED_MODEL_TYPE, (
                    f"Model type {model_type_id} not supported for concepts"
                )

        if self.config.get("checkpoints"):
            loader_type, _, hf_token, _, _, _ = self._validate_config_checkpoints()

            if loader_type == "huggingface":
                is_valid_token = hf_token and HuggingFaceLoader.validate_hftoken(hf_token)
                if not is_valid_token and hf_token:
                    logger.info(
                        "Continuing without Hugging Face token for validating config in model builder."
                    )

                has_repo_access = HuggingFaceLoader.validate_hf_repo_access(
                    repo_id=self.config.get("checkpoints", {}).get("repo_id"),
                    token=hf_token if is_valid_token else None,
                )

                if not has_repo_access:
                    logger.error(
                        f"Invalid Hugging Face repo access for repo {self.config.get('checkpoints').get('repo_id')}. Please check your repo and try again."
                    )
                    sys.exit("Token does not have access to HuggingFace repo , exiting.")

        num_threads = self.config.get("num_threads")
        if num_threads or num_threads == 0:
            assert isinstance(num_threads, int) and num_threads >= 1, ValueError(
                f"`num_threads` must be an integer greater than or equal to 1. Received type {type(num_threads)} with value {num_threads}."
            )
        else:
            num_threads = int(os.environ.get("CLARIFAI_NUM_THREADS", 16))
            self.config["num_threads"] = num_threads

        # Validate AgenticModelClass requirements
        if not self.download_validation_only:
            self._validate_agentic_model_requirements()

    def _validate_agentic_model_requirements(self):
        """
        Validate that AgenticModelClass models have required dependencies (fastmcp and mcp) in requirements.txt.
        """
        try:
            # Load the model class with mocking to avoid import errors
            model_class = self.load_model_class(mocking=True)

            # Check if the model class is a subclass of AgenticModelClass
            if issubclass(model_class, AgenticModelClass):
                # Parse requirements.txt to check for required packages
                dependencies = self._parse_requirements()

                missing_packages = []
                if 'fastmcp' not in dependencies:
                    missing_packages.append('fastmcp')
                if 'mcp' not in dependencies:
                    missing_packages.append('mcp')

                if missing_packages:
                    logger.error(
                        f"Model class '{model_class.__name__}' inherits from AgenticModelClass, "
                        f"but the following required packages are missing from requirements.txt: {', '.join(missing_packages)}, which are required for agentic models. "
                        f"Please add these packages to your requirements.txt file."
                    )
                    sys.exit(1)
        except Exception as e:
            # If we can't load the model class, log a warning but don't fail
            # This could happen if there are import errors, but we don't want to block
            # non-agentic models from being uploaded
            logger.debug(f"Could not validate AgenticModelClass requirements: {e}")

    def _validate_stream_options(self):
        """
        Validate OpenAI streaming configuration for Clarifai models.
        """
        if self.download_validation_only:
            return

        if not self._is_clarifai_internal():
            return  # Skip validation for non-clarifai models

        # Parse all Python files once
        all_python_content = self._get_all_python_content()

        if self._uses_openai_streaming(all_python_content):
            logger.info(
                "Detected OpenAI chat completions for Clarifai model streaming - validating stream_options..."
            )

            if not self.has_proper_usage_tracking(all_python_content):
                logger.error(
                    "Missing configuration to track usage for OpenAI chat completion calls. "
                    "Go to your model scripts and make sure to set both: "
                    "1) stream_options={'include_usage': True}"
                    "2) set_output_context"
                )

    def _validate_config_secrets(self):
        """
        Validate the secrets section in the config file.
        """
        if "secrets" not in self.config:
            return

        secrets = self.config.get("secrets", [])
        if not isinstance(secrets, list):
            raise ValueError("The 'secrets' field must be an array.")

        for i, secret in enumerate(secrets):
            if not isinstance(secret, dict):
                raise ValueError(f"Secret at index {i} must be a dictionary.")

            # Validate required fields
            if "id" not in secret or not secret["id"]:
                raise ValueError(f"Secret at index {i} must have a non-empty 'id' field.")

            if "type" not in secret or not secret["type"]:
                secret["type"] = "env"

            if "env_var" not in secret or not secret["env_var"]:
                raise ValueError(f"Secret at index {i} must have a non-empty 'env_var' field.")
            # Validate secret type
            if secret["type"] != "env":
                raise ValueError(
                    f"Secret at index {i} has invalid type '{secret['type']}'. Must be 'env'."
                )

        logger.info(f"Validated {len(secrets)} secrets in config file.")

    def _process_secrets(self):
        """
        Process secrets from config file and create/validate them using the User client.
        Returns the processed secrets array for inclusion in ModelVersion.OutputInfo.Params.
        """
        if "secrets" not in self.config:
            return []

        secrets = self.config.get("secrets", [])
        if not secrets:
            return []

        # Get user client for secret operations
        user = User(
            user_id=self.config.get('model').get('user_id'),
            pat=self.client.pat,
            token=self.client.token,
            base_url=self.client.base,
        )

        processed_secrets = []
        secrets_to_create = []

        for secret in secrets:
            secret_id = secret["id"]
            secret_type = secret.get("type", "env")
            env_var = secret["env_var"]
            secret_value = secret.get("value")  # Optional for existing secrets

            # Check if secret already exists
            try:
                existing_secret = user.get_secret(secret_id)
                logger.info(f"Secret '{secret_id}' already exists, using existing secret.")

                # Add to processed secrets without the value
                processed_secret = {
                    "id": secret_id,
                    "type": secret_type,
                    "env_var": env_var,
                }
                processed_secrets.append(processed_secret)

            except Exception:
                # Secret doesn't exist, need to create it
                if secret_value:
                    logger.info(f"Secret '{secret_id}' does not exist, will create it.")
                    secrets_to_create.append(
                        {
                            "id": secret_id,
                            "value": secret_value,
                            "description": secret.get("description", f"Secret for {env_var}"),
                        }
                    )

                    # Add to processed secrets
                    processed_secret = {
                        "id": secret_id,
                        "type": secret_type,
                        "env_var": env_var,
                    }
                    processed_secrets.append(processed_secret)
                else:
                    raise ValueError(
                        f"Secret '{secret_id}' does not exist and no value provided for creation."
                    )

        # Create new secrets if any
        if secrets_to_create:
            try:
                created_secrets = user.create_secrets(secrets_to_create)
                logger.info(f"Successfully created {len(created_secrets)} new secrets.")
            except Exception as e:
                logger.error(f"Failed to create secrets: {e}")
                raise

        return processed_secrets

    def _is_clarifai_internal(self):
        """
        Check if the current user is a Clarifai internal user based on email domain.

        Returns:
            bool: True if user is a Clarifai internal user, False otherwise
        """
        try:
            # Get user info from Clarifai API
            user_client = User(
                pat=self.client.pat, user_id=self.config.get('model').get('user_id')
            )
            user_response = user_client.get_user_info()

            user = user_response.user

            # Check primary email domain
            if hasattr(user, 'primary_email') and user.primary_email:
                return user.primary_email.endswith('@clarifai.com')

            return False

        except Exception as e:
            # Gracefully handle insufficient scopes (dev environment) or any other errors
            error_msg = str(e)
            if "CONN_INSUFFICIENT_SCOPES" in error_msg:
                logger.debug("Skipping user validation due to insufficient scopes")
            else:
                logger.debug(f"User validation failed (skip validation and continue): {e}")
            return False

    def _get_all_python_content(self):
        """
        Parse and concatenate all Python files in the model's 1/ subfolder.
        """
        model_folder = os.path.join(self.folder, '1')
        if not os.path.exists(model_folder):
            return ""

        all_content = []
        for root, _, files in os.walk(model_folder):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            all_content.append(f.read())
                    except Exception:
                        continue
        return "\n".join(all_content)

    def _uses_openai_streaming(self, python_content):
        return 'chat.completions.create' in python_content and 'generate(' in python_content

    def has_proper_usage_tracking(self, python_content):
        include_usage_patterns = ["'include_usage': True", '"include_usage": True']
        has_include_usage = any(pattern in python_content for pattern in include_usage_patterns)
        has_set_output_context = 'set_output_context' in python_content

        return has_include_usage and has_set_output_context

    @staticmethod
    def _get_tar_file_content_size(tar_file_path):
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

    def method_signatures_yaml(self):
        """
        Returns the method signatures for the model class in YAML format.
        """
        model_class = self.load_model_class(mocking=True)
        method_infos = model_class._get_method_infos()
        signatures = {method.name: method.signature for method in method_infos.values()}
        return signatures_to_yaml(signatures)

    def get_method_signatures(self, mocking=True):
        """
        Returns the method signatures for the model class.

        Args:
          mocking (bool): Whether to mock the model class or not. Defaults to False.

        Returns:
          list: A list of method signatures for the model class.
        """
        model_class = self.load_model_class(mocking=mocking)
        method_infos = model_class._get_method_infos()
        signatures = [
            method.signature for method in method_infos.values() if method.signature is not None
        ]
        return signatures

    @property
    def client(self):
        if self._client is None:
            assert "model" in self.config, "model info not found in the config file"
            model = self.config.get('model')
            assert "user_id" in model, "user_id not found in the config file"
            assert "app_id" in model, "app_id not found in the config file"
            # The owner of the model and the app.
            user_id = model.get('user_id')
            app_id = model.get('app_id')

            # Use context parameters if provided, otherwise fall back to environment variables
            self._base_api = (
                self._base_url
                if self._base_url
                else os.environ.get('CLARIFAI_API_BASE', 'https://api.clarifai.com')
            )

            # Create BaseClient with explicit pat parameter if provided
            if self._pat:
                self._client = BaseClient(
                    user_id=user_id, app_id=app_id, base=self._base_api, pat=self._pat
                )
            else:
                self._client = BaseClient(user_id=user_id, app_id=app_id, base=self._base_api)

        return self._client

    @property
    def model_ui_url(self):
        url_helper = ClarifaiUrlHelper(self._client.auth_helper)
        # Note(zeiler): the UI experience isn't the best when including version id right now.
        # if self.model_version_id is not None:
        #     return url_helper.clarifai_url(
        #         self.client.user_app_id.user_id,
        #         self.client.user_app_id.app_id,
        #         "models",
        #         self.model_id,
        #         self.model_version_id,
        #     )
        # else:
        return url_helper.clarifai_url(
            self.client.user_app_id.user_id,
            self.client.user_app_id.app_id,
            "models",
            self.model_id,
        )

    @property
    def model_api_url(self):
        url_helper = ClarifaiUrlHelper(self._client.auth_helper)
        if self.model_version_id is not None:
            return url_helper.api_url(
                self.client.user_app_id.user_id,
                self.client.user_app_id.app_id,
                "models",
                self.model_id,
                self.model_version_id,
            )
        else:
            return url_helper.api_url(
                self.client.user_app_id.user_id,
                self.client.user_app_id.app_id,
                "models",
                self.model_id,
            )

    def _get_model_proto(self):
        assert "model" in self.config, "model info not found in the config file"
        model = self.config.get('model')

        assert "model_type_id" in model, "model_type_id not found in the config file"
        assert "id" in model, "model_id not found in the config file"
        if not self.download_validation_only:
            assert "user_id" in model, "user_id not found in the config file"
            assert "app_id" in model, "app_id not found in the config file"

        model_proto = json_format.ParseDict(model, resources_pb2.Model())

        return model_proto

    def _get_inference_compute_info(self):
        assert "inference_compute_info" in self.config, (
            "inference_compute_info not found in the config file"
        )
        inference_compute_info = self.config.get('inference_compute_info')
        # Ensure cpu_limit is a string if it exists and is an int
        if 'cpu_limit' in inference_compute_info and isinstance(
            inference_compute_info['cpu_limit'], int
        ):
            inference_compute_info['cpu_limit'] = str(inference_compute_info['cpu_limit'])
        return json_format.ParseDict(inference_compute_info, resources_pb2.ComputeInfo())

    def check_model_exists(self):
        resp = self.client.STUB.GetModel(
            service_pb2.GetModelRequest(
                user_app_id=self.client.user_app_id, model_id=self.model_proto.id
            )
        )
        if resp.status.code == status_code_pb2.SUCCESS:
            return True
        return False

    def maybe_create_model(self):
        if self.check_model_exists():
            logger.info(
                f"Model '{self.client.user_app_id.user_id}/{self.client.user_app_id.app_id}/models/{self.model_proto.id}' already exists, "
                f"will create a new version for it."
            )
            return

        request = service_pb2.PostModelsRequest(
            user_app_id=self.client.user_app_id,
            models=[self.model_proto],
        )
        return self.client.STUB.PostModels(request)

    def _match_req_line(self, line):
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None
        # split on whitespace followed by #
        line = re.split(r'\s+#', line)[0]
        if "==" in line:
            pkg, version = line.split("==")
        elif ">=" in line:
            pkg, version = line.split(">=")
        elif ">" in line:
            pkg, version = line.split(">")
        elif "<=" in line:
            pkg, version = line.split("<=")
        elif "<" in line:
            pkg, version = line.split("<")
        elif "@" in line:
            pkg, version = line.split("@", 1)
        else:
            pkg, version = line, None  # No version specified
        pkg = pkg.strip()
        for dep in dependencies:
            if dep == pkg:
                if (
                    dep == 'torch' and line.find('whl/cpu') > 0
                ):  # Ignore torch-cpu whl files, use base mage.
                    return None, None
                return pkg, version.strip() if version else None
        return None, None

    def _parse_requirements(self):
        dependencies_version = {}
        with open(os.path.join(self.folder, 'requirements.txt'), 'r') as file:
            for line in file:
                # Skip empty lines and comments
                dependency, version = self._match_req_line(line)
                if dependency is None:
                    continue
                dependencies_version[dependency] = version if version else None
        return dependencies_version

    def _validate_requirements(self, python_version):
        """here we use uv pip compile to validate the requirements.txt file
        and ensure that the dependencies are compatible with each other prior to uploading
        """
        if not os.path.exists(os.path.join(self.folder, 'requirements.txt')):
            raise FileNotFoundError(
                "requirements.txt not found in the folder, please provide a valid requirements.txt file"
            )
        path = os.path.join(self.folder, 'requirements.txt')
        # run the f"uv pip compile {path} --universal" command to validate the requirements.txt file
        if not shutil.which('uv'):
            raise Exception(
                "uv command not found, please install uv to validate the requirements.txt file"
            )
        logger.info(f"Setup: Validating requirements.txt file at {path} using uv pip compile")
        # Don't log the output of the comment unless it errors.
        result = subprocess.run(
            f"uv pip compile {path} --universal --python {python_version} --no-header --no-emit-index-url --no-cache-dir",
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Error validating requirements.txt file: {result.stderr}")
            logger.error(
                "Failed to validate the requirements.txt file, please check the file for errors. Note this can happen if the machine you're upload from has different python version, accelerator, etc. from the desired machine you want to upload to."
            )
            logger.error("Output: " + result.stdout)
            # If we have an error, raise an exception.
            return False
        else:
            logger.info("Setup: Requirements.txt file validated successfully")
            # If we have no error, we can just return.
            return True

    def _is_amd(self):
        """
        Check if the model is AMD or not.
        """
        is_amd_gpu = False
        is_nvidia_gpu = False
        if "inference_compute_info" in self.config:
            inference_compute_info = self.config.get('inference_compute_info')
            if 'accelerator_type' in inference_compute_info:
                accelerator_type = inference_compute_info['accelerator_type']
                if accelerator_type:  # Check if not None or empty
                    for accelerator in accelerator_type:
                        if 'amd' in accelerator.lower():
                            is_amd_gpu = True
                        elif 'nvidia' in accelerator.lower():
                            is_nvidia_gpu = True
        if is_amd_gpu and is_nvidia_gpu:
            raise Exception(
                "Both AMD and NVIDIA GPUs are specified in the config file, please use only one type of GPU."
            )
        if is_amd_gpu:
            logger.info(
                "Setup: Using AMD base image to build the Docker image and upload the model"
            )
        elif is_nvidia_gpu:
            logger.info(
                "Setup: Using NVIDIA base image to build the Docker image and upload the model"
            )
        return is_amd_gpu

    def _lint_python_code(self):
        """
        Lint the python code in the model.py file using flake8.
        This will help catch any simple bugs in the code before uploading it to the API.
        """
        if not shutil.which('ruff'):
            raise Exception("ruff command not found, please install ruff to lint the python code")
        # List all the python files in the /1/ folder recursively and lint them.
        python_files = []
        for root, _, files in os.walk(os.path.join(self.folder, '1')):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        if not python_files:
            logger.info("No Python files found to lint, skipping linting step.")
        elif len(python_files) > 10:
            logger.info(f"Setup: Linting {len(python_files)} Python files.")
        else:
            logger.info(f"Setup: Linting Python files: {python_files}")

        # Run ruff to lint the python code.
        # Use --no-cache to prevent .ruff_cache folder generation in model directories
        command = "ruff check --select=F --no-cache"

        # Batch linting to avoid "Argument list too long" error
        batch_size = 100
        all_success = True
        for i in range(0, len(python_files), batch_size):
            batch = python_files[i : i + batch_size]
            result = subprocess.run(
                f"{command} {' '.join(batch)}",
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                all_success = False
                logger.error(f"Error linting Python code: {result.stderr}")
                logger.error("Output: " + result.stdout)

        if not all_success:
            logger.error(
                f"Failed to lint the Python code, please check the code for errors using '{command}' so you don't have simple errors in your code prior to upload."
            )
        else:
            logger.info("Setup: Python code linted successfully, no errors found.")

    def _normalize_dockerfile_content(self, content):
        """
        Normalize Dockerfile content for comparison by standardizing whitespace and indentation.
        This handles differences in spacing, indentation, and line endings.
        """
        lines = []
        for line in content.splitlines():
            # Strip leading/trailing whitespace from each line
            normalized_line = line.strip()
            # Skip empty lines for comparison
            if normalized_line:
                lines.append(normalized_line)
        # Join with consistent line endings
        return '\n'.join(lines)

    def _generate_dockerfile_content(self):
        """
        Generate the Dockerfile content based on the model configuration.
        This is a helper method that returns the content without writing to file.
        """

        additional_packages = ""
        streaming_video_consumer = self.config.get('streaming_video_consumer', False)
        if streaming_video_consumer:
            additional_packages = STREAMING_VIDEO_ADDITIONAL_PACKAGE_INSTALLATION

        # Get the Python version from the config file
        build_info = self.config.get('build_info', {})

        # Check if node_version is specified - if so, use the Node.js Dockerfile template
        node_version = build_info.get('node_version', '') or ''
        use_node_template = bool(node_version and str(node_version).strip())

        if use_node_template:
            dockerfile_template_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'dockerfile_template',
                'Dockerfile.node.template',
            )
            logger.info(
                f"Setup: Node version {node_version} specified in config.yaml, using Node.js Dockerfile template"
            )
        else:
            dockerfile_template_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'dockerfile_template',
                'Dockerfile.template',
            )

        with open(dockerfile_template_path, 'r') as template_file:
            dockerfile_template = template_file.read()

        dockerfile_template = Template(dockerfile_template)

        if 'python_version' in build_info:
            python_version = build_info['python_version']
            if python_version not in AVAILABLE_PYTHON_IMAGES:
                raise Exception(
                    f"Python version {python_version} not supported, please use one of the following versions: {AVAILABLE_PYTHON_IMAGES} in your config.yaml"
                )

            logger.info(
                f"Setup: Using Python version {python_version} from the config file to build the Dockerfile"
            )
        else:
            logger.info(
                f"Setup: Python version not found in the config file, using default Python version: {DEFAULT_PYTHON_VERSION}"
            )
            python_version = DEFAULT_PYTHON_VERSION

        # Before we bother even picking the right base image, let's use uv to validate
        # that the requirements.txt file is valid and compatible.
        self._validate_requirements(python_version)

        # Make sure any python code will not have simple bugs by linting it first.
        self._lint_python_code()

        # Parse the requirements.txt file to determine the base image
        dependencies = self._parse_requirements()

        # If using Node.js template, use simpler substitution
        if use_node_template:
            if 'clarifai' not in dependencies:
                raise Exception(
                    f"clarifai not found in requirements.txt, please add clarifai to the requirements.txt file with a fixed version. Current version is clarifai=={CLARIFAI_LATEST_VERSION}"
                )
            clarifai_version = dependencies['clarifai']
            if not clarifai_version:
                logger.warn(
                    f"clarifai version not found in requirements.txt, using the latest version {CLARIFAI_LATEST_VERSION}"
                )
                clarifai_version = CLARIFAI_LATEST_VERSION
                lines = []
                with open(os.path.join(self.folder, 'requirements.txt'), 'r') as file:
                    for line in file:
                        # if the line without whitespace is "clarifai"
                        dependency, version = self._match_req_line(line)
                        if dependency and dependency == "clarifai":
                            lines.append(
                                line.replace("clarifai", f"clarifai=={CLARIFAI_LATEST_VERSION}")
                            )
                        else:
                            lines.append(line)
                with open(os.path.join(self.folder, 'requirements.txt'), 'w') as file:
                    file.writelines(lines)
                logger.warn(
                    f"Updated requirements.txt to have clarifai=={CLARIFAI_LATEST_VERSION}"
                )

            # Replace placeholders with actual values for Node.js template
            dockerfile_content = dockerfile_template.safe_substitute(
                PYTHON_VERSION=python_version,
                NODE_VERSION=str(node_version).strip(),
            )
            return dockerfile_content

        # Standard template logic (multi-stage build)
        is_amd_gpu = self._is_amd()
        if is_amd_gpu:
            final_image = AMD_PYTHON_BASE_IMAGE.format(python_version=python_version)
            downloader_image = AMD_PYTHON_BASE_IMAGE.format(python_version=python_version)
            if 'vllm' in dependencies:
                if python_version != DEFAULT_PYTHON_VERSION:
                    raise Exception(
                        f"vLLM is not supported with Python version {python_version}, please use Python version {DEFAULT_PYTHON_VERSION} in your config.yaml"
                    )
                torch_version = dependencies.get('torch', None)
                if 'torch' in dependencies:
                    if not torch_version:
                        logger.info(
                            f"Setup: torch version not found in requirements.txt, using the default version {DEFAULT_AMD_TORCH_VERSION}"
                        )
                        torch_version = DEFAULT_AMD_TORCH_VERSION
                    elif torch_version not in [DEFAULT_AMD_TORCH_VERSION]:
                        # Currently, we have only one vLLM image built with the DEFAULT_AMD_TORCH_VERSION.
                        # If the user requests a different PyTorch version, that specific version will be
                        # installed during the requirements.txt installation step
                        torch_version = DEFAULT_AMD_TORCH_VERSION
                else:
                    logger.info(
                        f"`torch` not found in requirements.txt, using the default torch=={DEFAULT_AMD_TORCH_VERSION}"
                    )
                    torch_version = DEFAULT_AMD_TORCH_VERSION
                python_version = DEFAULT_PYTHON_VERSION
                gpu_version = DEFAULT_AMD_GPU_VERSION
                final_image = AMD_VLLM_BASE_IMAGE.format(
                    torch_version=torch_version,
                    python_version=python_version,
                    gpu_version=gpu_version,
                )
                logger.info("Setup: Using vLLM base image to build the Docker image")
            elif (
                'torch' in dependencies
                and (dependencies['torch'] in [None, DEFAULT_AMD_TORCH_VERSION])
                and python_version == DEFAULT_PYTHON_VERSION
            ):
                torch_version = dependencies['torch']
                if not torch_version:
                    logger.info(
                        f"torch version not found in requirements.txt, using the default version {DEFAULT_AMD_TORCH_VERSION}"
                    )
                    torch_version = DEFAULT_AMD_TORCH_VERSION
                python_version = DEFAULT_PYTHON_VERSION
                gpu_version = DEFAULT_AMD_GPU_VERSION
                final_image = AMD_TORCH_BASE_IMAGE.format(
                    torch_version=torch_version,
                    python_version=python_version,
                    gpu_version=gpu_version,
                )
                logger.info(
                    f"Setup: Using Torch version {torch_version} base image to build the Docker image"
                )
        else:
            final_image = PYTHON_BASE_IMAGE.format(python_version=python_version)
            downloader_image = PYTHON_BASE_IMAGE.format(python_version=python_version)
            if 'torch' in dependencies and dependencies['torch']:
                torch_version = dependencies['torch']
                # Sort in reverse so that newer cuda versions come first and are preferred.
                for image in sorted(AVAILABLE_TORCH_IMAGES, reverse=True):
                    if image.find('rocm') >= 0:
                        continue  # skip ROCm images as those are handled above.
                    if torch_version in image and f'py{python_version}' in image:
                        # like cu124, rocm6.3, etc.
                        gpu_version = image.split('-')[-1]
                        final_image = TORCH_BASE_IMAGE.format(
                            torch_version=torch_version,
                            python_version=python_version,
                            gpu_version=gpu_version,
                        )
                        logger.info(
                            f"Setup: Using Torch version {torch_version} base image to build the Docker image"
                        )
                        break
        if 'clarifai' not in dependencies:
            raise Exception(
                f"clarifai not found in requirements.txt, please add clarifai to the requirements.txt file with a fixed version. Current version is clarifai=={CLARIFAI_LATEST_VERSION}"
            )
        clarifai_version = dependencies['clarifai']
        if not clarifai_version:
            logger.warn(
                f"clarifai version not found in requirements.txt, using the latest version {CLARIFAI_LATEST_VERSION}"
            )
            clarifai_version = CLARIFAI_LATEST_VERSION
            lines = []
            with open(os.path.join(self.folder, 'requirements.txt'), 'r') as file:
                for line in file:
                    # if the line without whitespace is "clarifai"
                    dependency, version = self._match_req_line(line)
                    if dependency and dependency == "clarifai":
                        lines.append(
                            line.replace("clarifai", f"clarifai=={CLARIFAI_LATEST_VERSION}")
                        )
                    else:
                        lines.append(line)
            with open(os.path.join(self.folder, 'requirements.txt'), 'w') as file:
                file.writelines(lines)
            logger.warn(f"Updated requirements.txt to have clarifai=={CLARIFAI_LATEST_VERSION}")

        # Replace placeholders with actual values
        dockerfile_content = dockerfile_template.safe_substitute(
            name='main',
            FINAL_IMAGE=final_image,  # for pip requirements
            DOWNLOADER_IMAGE=downloader_image,  # for downloading checkpoints
            CLARIFAI_VERSION=clarifai_version,  # for clarifai
            ADDITIONAL_PACKAGES=additional_packages,
        )

        return dockerfile_content

    def create_dockerfile(self, generate_dockerfile=False):
        """
        Create a Dockerfile for the model based on its configuration.
        """
        generated_content = self._generate_dockerfile_content()

        if generate_dockerfile:
            should_create_dockerfile = True
        else:
            # Always handle Dockerfile creation with user interaction when content differs
            dockerfile_path = os.path.join(self.folder, 'Dockerfile')
            should_create_dockerfile = True

            if os.path.exists(dockerfile_path):
                # Read existing Dockerfile content
                with open(dockerfile_path, 'r') as existing_dockerfile:
                    existing_content = existing_dockerfile.read()

                # Compare content (normalize for robust comparison that handles indentation differences)
                if self._normalize_dockerfile_content(
                    existing_content
                ) == self._normalize_dockerfile_content(generated_content):
                    logger.info(
                        "Dockerfile already exists with identical content, skipping creation."
                    )
                    should_create_dockerfile = False
                else:
                    logger.info("Dockerfile already exists with different content.")
                    response = input(
                        "A different Dockerfile already exists. Do you want to overwrite it with the generated one? "
                        "Type 'y' to overwrite, 'n' to keep your custom Dockerfile: "
                    )
                    if response.lower() != 'y':
                        logger.info("Keeping existing custom Dockerfile.")
                        should_create_dockerfile = False
                    else:
                        logger.info("Overwriting existing Dockerfile with generated content.")

        if should_create_dockerfile:
            # Write Dockerfile
            dockerfile_path = os.path.join(self.folder, 'Dockerfile')
            with open(dockerfile_path, 'w') as dockerfile:
                dockerfile.write(generated_content)

    @property
    def checkpoint_path(self):
        return self._checkpoint_path(self.folder)

    def _checkpoint_path(self, folder):
        return os.path.join(folder, self.checkpoint_suffix)

    @property
    def checkpoint_suffix(self):
        return os.path.join('1', 'checkpoints')

    @property
    def tar_file(self):
        return f"{self.folder}.tar.gz"

    def default_runtime_checkpoint_path(self):
        return DEFAULT_RUNTIME_DOWNLOAD_PATH

    def download_checkpoints(
        self, stage: str = DEFAULT_DOWNLOAD_CHECKPOINT_WHEN, checkpoint_path_override: str = None
    ):
        """
        Downloads the checkpoints specified in the config file.

        :param stage: The stage of the build process. This is used to determine when to download the
        checkpoints. The stage can be one of ['build', 'upload', 'runtime']. If you want to force
        downloading now then set stage to match e when field of the checkpoints section of you config.yaml.
        :param checkpoint_path_override: The path to download the checkpoints to (with 1/checkpoints added as suffix). If not provided, the
        default path is used based on the folder ModelUploader was initialized with. The checkpoint_suffix will be appended to the path.
        If stage is 'runtime' and checkpoint_path_override is None, the default runtime path will be used.

        :return: The path to the downloaded checkpoints. Even if it doesn't download anything, it will return the default path.
        """
        path = self.checkpoint_path  # default checkpoint path.
        if not self.config.get("checkpoints"):
            logger.info("No checkpoints specified in the config file")
            return path
        clarifai_model_type_id = self.config.get('model').get('model_type_id')

        loader_type, repo_id, hf_token, when, allowed_file_patterns, ignore_file_patterns = (
            self._validate_config_checkpoints()
        )
        if stage not in ["build", "upload", "runtime"]:
            raise Exception(
                "Invalid stage provided, must be one of ['build', 'upload', 'runtime']"
            )
        if when != stage:
            logger.info(
                f"Skipping downloading checkpoints for stage {stage} since config.yaml says to download them at stage {when}"
            )
            return path

        success = False
        if loader_type == "huggingface":
            loader = HuggingFaceLoader(
                repo_id=repo_id, token=hf_token, model_type_id=clarifai_model_type_id
            )
            # for runtime default to /tmp path
            if stage == "runtime" and checkpoint_path_override is None:
                checkpoint_path_override = self.default_runtime_checkpoint_path()
            path = checkpoint_path_override if checkpoint_path_override else self.checkpoint_path
            success = loader.download_checkpoints(
                path,
                allowed_file_patterns=allowed_file_patterns,
                ignore_file_patterns=ignore_file_patterns,
            )

        if loader_type:
            if not success:
                logger.error(f"Failed to download checkpoints for model {repo_id}")
                sys.exit(1)
            else:
                logger.info(f"Downloaded checkpoints for model {repo_id} successfully to {path}")
        return path

    def _concepts_protos_from_concepts(self, concepts):
        concept_protos = []
        for concept in concepts:
            concept_protos.append(
                resources_pb2.Concept(
                    id=str(concept[0]),
                    name=concept[1],
                )
            )
        return concept_protos

    def hf_labels_to_config(self, labels, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        model = config.get('model')
        model_type_id = model.get('model_type_id')
        assert model_type_id in CONCEPTS_REQUIRED_MODEL_TYPE, (
            f"Model type {model_type_id} not supported for concepts"
        )
        concept_protos = self._concepts_protos_from_concepts(labels)

        config['concepts'] = [
            {'id': concept.id, 'name': concept.name} for concept in concept_protos
        ]

        with open(config_file, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
        concepts = config.get('concepts')
        logger.info(f"Updated config.yaml with {len(concepts)} concepts.")

    def _get_git_info(self) -> Optional[Dict[str, Any]]:
        """
        Get git repository information for the model path.

        Returns:
            Dict with git info (url, commit, branch) or None if not a git repository
        """
        try:
            # Check if the folder is within a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get git remote URL
            remote_result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get current commit hash
            commit_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=True,
            )

            # Get current branch
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=False,
            )

            git_info = {
                'commit': commit_result.stdout.strip(),
                'branch': branch_result.stdout.strip()
                if branch_result.returncode == 0
                else 'HEAD',
            }

            if remote_result.returncode == 0:
                git_info['url'] = remote_result.stdout.strip()

            return git_info

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not a git repository or git not available
            return None

    def _check_git_status_and_prompt(self) -> bool:
        """
        Check for uncommitted changes in git repository within the model path and prompt user.

        Returns:
            True if should continue with upload, False if should abort
        """
        try:
            # Check for uncommitted changes within the model path only
            status_result = subprocess.run(
                ['git', 'status', '--porcelain', '.'],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=True,
            )

            if status_result.stdout.strip():
                logger.warning("Uncommitted changes detected in model path:")
                logger.warning(status_result.stdout)

                response = input(
                    "\nDo you want to continue upload with uncommitted changes? (y/N): "
                )
                return response.lower() in ['y', 'yes']
            else:
                logger.info("Model path has no uncommitted changes.")
                return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Error checking git status, but we already know it's a git repo
            logger.warning("Could not check git status, continuing with upload.")
            return True

    def get_model_version_proto(self, git_info: Optional[Dict[str, Any]] = None):
        """
        Create a ModelVersion protobuf message for the model.
        Args:
          git_info (Optional[Dict[str, Any]]): Git repository information to include in metadata.
        Returns:
          resources_pb2.ModelVersion: The ModelVersion protobuf message.
        """

        signatures = self.get_method_signatures()
        model_version_proto = resources_pb2.ModelVersion(
            pretrained_model_config=resources_pb2.PretrainedModelConfig(),
            inference_compute_info=self.inference_compute_info,
            method_signatures=signatures,
        )

        # Add build_info with platform if specified in CLI or config
        # CLI platform takes precedence over config platform
        platform = None
        if self._cli_platform:
            platform = self._cli_platform
            logger.info(f"Using platform from CLI: {platform}")
        else:
            build_info_config = self.config.get('build_info', {})
            if 'platform' in build_info_config:
                platform = build_info_config['platform']
                if platform:
                    logger.info(f"Using platform from config.yaml: {platform}")

        # Check if platform is not None and not an empty string
        if platform:
            # Create BuildInfo and set platform if the field is available
            build_info = resources_pb2.BuildInfo()
            if hasattr(build_info, 'platform'):
                build_info.platform = platform
                model_version_proto.build_info.CopyFrom(build_info)
                logger.info(f"Set build platform to: {platform}")
            else:
                logger.warning(
                    f"Platform '{platform}' specified but not supported "
                    "in current clarifai-grpc version. Please update clarifai-grpc to use this feature."
                )

        # Add git information to metadata if available
        if git_info:
            from google.protobuf.struct_pb2 import Struct

            metadata_struct = Struct()
            metadata_struct.update({'git_registry': git_info})
            model_version_proto.metadata.CopyFrom(metadata_struct)

        # Process and add secrets to output_info.params
        try:
            processed_secrets = self._process_secrets()
            if processed_secrets:
                # Initialize output_info.params if not already present
                if not model_version_proto.HasField("output_info"):
                    model_version_proto.output_info.CopyFrom(resources_pb2.OutputInfo())

                # Initialize params if not already present
                if not model_version_proto.output_info.HasField("params"):
                    from google.protobuf.struct_pb2 import Struct

                    model_version_proto.output_info.params.CopyFrom(Struct())

                # Add secrets to params
                model_version_proto.output_info.params.update({"secrets": processed_secrets})
                logger.info(
                    f"Added {len(processed_secrets)} secrets to model version output_info.params"
                )
        except Exception as e:
            logger.error(f"Failed to process secrets: {e}")
            raise

        model_type_id = self.config.get('model').get('model_type_id')
        if model_type_id in CONCEPTS_REQUIRED_MODEL_TYPE:
            if 'concepts' in self.config:
                labels = self.config.get('concepts')
                logger.info(f"Found {len(labels)} concepts in the config file.")
                for concept in labels:
                    concept_proto = json_format.ParseDict(concept, resources_pb2.Concept())
                    model_version_proto.output_info.data.concepts.append(concept_proto)
            elif self.config.get("checkpoints") and HuggingFaceLoader.validate_concept(
                self.checkpoint_path
            ):
                labels = HuggingFaceLoader.fetch_labels(self.checkpoint_path)
                logger.info(f"Found {len(labels)} concepts from the model checkpoints.")
                # sort the concepts by id and then update the config file
                labels = sorted(labels.items(), key=lambda x: int(x[0]))

                config_file = os.path.join(self.folder, 'config.yaml')
                try:
                    self.hf_labels_to_config(labels, config_file)
                except Exception as e:
                    logger.error(f"Failed to update the config.yaml file with the concepts: {e}")

                model_version_proto.output_info.data.concepts.extend(
                    self._concepts_protos_from_concepts(labels)
                )
        return model_version_proto

    def upload_model_version(self, git_info=None):
        file_path = f"{self.folder}.tar.gz"
        logger.debug(f"Will tar it into file: {file_path}")

        model_type_id = self.config.get('model').get('model_type_id')
        loader_type, repo_id, hf_token, when, _, _ = self._validate_config_checkpoints()

        if (model_type_id in CONCEPTS_REQUIRED_MODEL_TYPE) and 'concepts' not in self.config:
            logger.info(
                f"Model type {model_type_id} requires concepts to be specified in the config.yaml file.."
            )
            if self.config.get("checkpoints"):
                logger.info(
                    "Checkpoints specified in the config.yaml file, will download the HF model's config.json file to infer the concepts."
                )
                # If we don't already have the concepts, download the config.json file from HuggingFace
                if loader_type == "huggingface":
                    # If the config.yaml says we'll download in the future (build time or runtime) then we need to get this config now.
                    if when != "upload" and not HuggingFaceLoader.validate_config(
                        self.checkpoint_path
                    ):
                        input(
                            "Press Enter to download the HuggingFace model's config.json file to infer the concepts and continue..."
                        )
                        loader = HuggingFaceLoader(repo_id=repo_id, token=hf_token)
                        loader.download_config(self.checkpoint_path)

            else:
                logger.error(
                    "No checkpoints specified in the config.yaml file to infer the concepts. Please either specify the concepts directly in the config.yaml file or include a checkpoints section to download the HF model's config.json file to infer the concepts."
                )
                return

        model_version_proto = self.get_model_version_proto(git_info)

        def filter_func(tarinfo):
            name = tarinfo.name
            exclude = [self.tar_file, "*~", "*.pyc", "*.pyo", "__pycache__", ".ruff_cache"]
            if when != "upload":
                exclude.append(self.checkpoint_suffix)
            return None if any(name.endswith(ex) for ex in exclude) else tarinfo

        with tarfile.open(self.tar_file, "w:gz") as tar:
            tar.add(self.folder, arcname=".", filter=filter_func)
        logger.debug("Tarring complete, about to start upload.")

        file_size = os.path.getsize(self.tar_file)
        logger.debug(f"Size of the tar is: {file_size} bytes")

        self.storage_request_size = self._get_tar_file_content_size(file_path)
        if when != "upload" and self.config.get("checkpoints"):
            # Get the checkpoint size to add to the storage request.
            # First check for the env variable, then try querying huggingface. If all else fails, use the default.
            checkpoint_size = int(os.environ.get('CHECKPOINT_SIZE_BYTES', 0))
            if not checkpoint_size:
                _, repo_id, _, _, _, _ = self._validate_config_checkpoints()
                checkpoint_size = HuggingFaceLoader.get_huggingface_checkpoint_total_size(repo_id)
            if not checkpoint_size:
                checkpoint_size = self.DEFAULT_CHECKPOINT_SIZE
            self.storage_request_size += checkpoint_size

        resp = self.maybe_create_model()
        if not self.check_model_exists():
            logger.error(f"Failed to create model: {self.model_proto.id}. Details: {resp}")
            sys.exit(1)

        for response in self.client.STUB.PostModelVersionsUpload(
            self.model_version_stream_upload_iterator(model_version_proto, file_path),
        ):
            percent_completed = 0
            if response.status.code == status_code_pb2.UPLOAD_IN_PROGRESS:
                percent_completed = response.status.percent_completed
            details = response.status.details

            print(
                f"Status: {response.status.description}, Progress: {percent_completed}% - {details} ",
                f"request_id: {response.status.req_id}",
                end='\r',
                flush=True,
            )
        if response.status.code != status_code_pb2.MODEL_BUILDING:
            logger.error(f"Failed to upload model version: {response}")
            return
        self.model_version_id = response.model_version_id
        logger.info(f"Created Model Version ID: {self.model_version_id}")
        logger.info(f"Full url to that version is: {self.model_ui_url}")
        is_uploaded = False
        try:
            is_uploaded = self.monitor_model_build()
            if is_uploaded:
                # python code to run the model.

                method_signatures = self.get_method_signatures()
                snippet = code_script.generate_client_script(
                    method_signatures,
                    user_id=self.client.user_app_id.user_id,
                    app_id=self.client.user_app_id.app_id,
                    model_id=self.model_proto.id,
                    colorize=True,
                )
                logger.info("""\n
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Here is a code snippet to use this model:
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                """)
                logger.info(snippet)
                logger.info("""\n
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                """)
        finally:
            if os.path.exists(self.tar_file):
                logger.debug(f"Cleaning up upload file: {self.tar_file}")
                os.remove(self.tar_file)
        if is_uploaded:
            return self.model_version_id

    def model_version_stream_upload_iterator(self, model_version_proto, file_path):
        yield self.init_upload_model_version(model_version_proto, file_path)
        with open(file_path, "rb") as f:
            file_size = os.path.getsize(file_path)
            chunk_size = int(127 * 1024 * 1024)  # 127MB chunk size
            num_chunks = (file_size // chunk_size) + 1
            logger.info("Uploading file...")
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
                    yield service_pb2.PostModelVersionsUploadRequest(
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

    def init_upload_model_version(self, model_version_proto, file_path):
        file_size = os.path.getsize(file_path)
        logger.debug(f"Uploading model version of model {self.model_proto.id}")
        logger.debug(f"Using file '{os.path.basename(file_path)}' of size: {file_size} bytes")
        result = service_pb2.PostModelVersionsUploadRequest(
            upload_config=service_pb2.PostModelVersionsUploadConfig(
                user_app_id=self.client.user_app_id,
                model_id=self.model_proto.id,
                model_version=model_version_proto,
                total_size=file_size,
                storage_request_size=self.storage_request_size,
                is_v3=self.is_v3,
            )
        )
        return result

    def get_model_build_logs(self, current_page=1):
        logs_request = service_pb2.ListLogEntriesRequest(
            log_type="builder",
            user_app_id=self.client.user_app_id,
            model_id=self.model_proto.id,
            model_version_id=self.model_version_id,
            page=current_page,
            per_page=50,
        )
        response = self.client.STUB.ListLogEntries(logs_request)
        return response

    def monitor_model_build(self):
        st = time.time()
        seen_logs = set()  # To avoid duplicate log messages
        current_page = 1  # Track current page for log pagination
        while True:
            resp = self.client.STUB.GetModelVersion(
                service_pb2.GetModelVersionRequest(
                    user_app_id=self.client.user_app_id,
                    model_id=self.model_proto.id,
                    version_id=self.model_version_id,
                )
            )

            status_code = resp.model_version.status.code
            logs = self.get_model_build_logs(current_page)
            entries_count = 0
            for log_entry in logs.log_entries:
                entries_count += 1
                if log_entry.url not in seen_logs:
                    seen_logs.add(log_entry.url)
                    log_entry_msg = re.sub(
                        r"(\\*)(\[[a-z#/@][^[]*?])",
                        lambda m: f"{m.group(1)}{m.group(1)}\\{m.group(2)}",
                        log_entry.message.strip(),
                    )
                    logger.info(log_entry_msg)

            # If we got a full page (50 entries), there might be more logs on the next page
            # If we got fewer than 50 entries, we've reached the end and should stay on current page
            if entries_count == 50:
                current_page += 1
            # else: stay on current_page
            if status_code == status_code_pb2.MODEL_BUILDING:
                print(
                    f"Model is building... (elapsed {time.time() - st:.1f}s)", end='\r', flush=True
                )

                # Fetch and display the logs
                time.sleep(1)
            elif status_code == status_code_pb2.MODEL_TRAINED:
                logger.info("Model build complete!")
                logger.info(f"Build time elapsed {time.time() - st:.1f}s)")
                logger.info(
                    f"Check out the model at {self.model_ui_url} version: {self.model_version_id}"
                )
                return True
            else:
                logger.info(
                    f"\nModel build failed with status: {resp.model_version.status} and response {resp}"
                )
                return False


def upload_model(
    folder,
    stage,
    skip_dockerfile,
    platform: Optional[str] = None,
    pat: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Uploads a model to Clarifai.

    :param folder: The folder containing the model files.
    :param stage: The stage we are calling download checkpoints from. Typically this would "upload" and will download checkpoints if config.yaml checkpoints section has when set to "upload". Other options include "runtime" to be used in load_model or "upload" to be used during model upload. Set this stage to whatever you have in config.yaml to force downloading now.
    :param skip_dockerfile: If True, will skip Dockerfile generation entirely. If False or not provided, intelligently handle existing Dockerfiles with user confirmation.
    :param platform: Target platform(s) for Docker image build (e.g., "linux/amd64" or "linux/amd64,linux/arm64"). This overrides the platform specified in config.yaml.
    :param pat: Personal access token for authentication. If None, will use environment variables.
    :param base_url: Base URL for the API. If None, will use environment variables.
    """
    builder = ModelBuilder(
        folder, app_not_found_action="prompt", platform=platform, pat=pat, base_url=base_url
    )
    builder.download_checkpoints(stage=stage)

    if not skip_dockerfile:
        builder.create_dockerfile()

    exists = builder.check_model_exists()
    if exists:
        logger.info(
            f"Model already exists at {builder.model_ui_url}, this upload will create a new version for it."
        )
    else:
        logger.info(
            f"New model will be created at {builder.model_ui_url} with it's first version."
        )

    # Check for git repository information
    git_info = builder._get_git_info()
    if git_info:
        logger.info(f"Detected git repository: {git_info.get('url', 'local repository')}")
        logger.info(f"Current commit: {git_info['commit']}")
        logger.info(f"Current branch: {git_info['branch']}")

        # Check for uncommitted changes and prompt user
        if not builder._check_git_status_and_prompt():
            logger.info("Upload cancelled by user due to uncommitted changes.")
            return
    input("Press Enter to continue...")

    model_version = builder.upload_model_version(git_info)

    # Ask user if they want to deploy the model
    if model_version is not None:  # if it comes back None then it failed.
        if get_yes_no_input("\n🔶 Do you want to deploy the model?", True):
            # Setup deployment for the uploaded model
            setup_deployment_for_model(builder)
        else:
            logger.info("Model uploaded successfully. Skipping deployment setup.")
            return


def deploy_model(
    model_url=None,
    model_id=None,
    app_id=None,
    user_id=None,
    deployment_id=None,
    model_version_id=None,
    nodepool_id=None,
    compute_cluster_id=None,
    cluster_user_id=None,
    min_replicas=0,
    max_replicas=5,
):
    """
    Deploy a model on Clarifai platform.
    Args:
        model_url (str): The full Clarifai model URL (optional if model_id is provided).
        model_id (str): The ID of the model to be deployed (optional if model_url is provided).
        app_id (str): The application ID where the model resides.
        user_id (str): The user ID who owns the model.
        deployment_id (str): The ID for the new deployment.
        model_version_id (str): The version ID of the model to deploy. If not provided, the latest version will be used.
        nodepool_id (str): The ID of the nodepool where the deployment will be created.
        compute_cluster_id (str): The ID of the compute cluster to use for deployment.
        cluster_user_id (str): The user ID that owns the compute cluster.
        min_replicas (int): Minimum number of replicas for autoscaling.
        max_replicas (int): Maximum number of replicas for autoscaling.
    """
    if model_url and model_id:
        raise UserError("You can only specify one of url or model_id.")
    if not model_url and not model_id:
        raise UserError("You must specify one of url or model_id.")
    if model_url:
        user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(model_url)
    if not model_version_id:
        model = Model(model_id=model_id, app_id=app_id, user_id=user_id)
        model_versions = [v for v in model.list_versions()]
        if not model_versions:
            raise UserError(f"No versions found for model {model_id}.")
        if len(model_versions) > 1:
            # model_version_id = model_versions[len(model_versions) - 1].model_version.id # Use the first version
            model_version_id = model_versions[0].model_version.id  # latest version

    # Construct the full deployment config
    deployment_config = {
        "deployment": {
            "id": deployment_id,
            "user_id": user_id,
            "description": "Model deployment created to test Model upload",
            "autoscale_config": {
                "min_replicas": min_replicas,
                "max_replicas": max_replicas,
                "traffic_history_seconds": 600,
                "scale_down_delay_seconds": 300,
                "scale_to_zero_delay_seconds": 3600,
                "scale_up_delay_seconds": 300,
            },
            "worker": {
                "model": {
                    "id": model_id,
                    "model_version": {
                        "id": model_version_id,
                    },
                    "user_id": user_id,
                    "app_id": app_id,
                }
            },
            "scheduling_choice": 4,  # "performance"
            "nodepools": [
                {
                    "id": nodepool_id,
                    "compute_cluster": {
                        "id": compute_cluster_id,
                        "user_id": cluster_user_id,
                    },
                }
            ],
            "visibility": {"gettable": 50},
        }
    }

    try:
        # Instantiate Nodepool and create the deployment
        nodepool = Nodepool(nodepool_id=nodepool_id, user_id=user_id)
        deployment = nodepool.create_deployment(
            deployment_id=deployment_id, deployment_config=deployment_config
        )

        print(
            f"✅ Deployment '{deployment_id}' successfully created for model '{model_id}' with version '{model_version_id}'."
        )
        return True
    except Exception as e:
        print(f"❌ Failed to create deployment '{deployment_id}': {e}")
        return False


def setup_deployment_for_model(builder):
    """
    Set up deployment for a model after upload.

    Args:
        builder: The ModelBuilder instance that has uploaded the model.
    """

    print("\n🚀 Model Deployment")
    state = {
        'uploaded': True,
        'deployed': False,
    }
    model = builder.config.get('model')
    state.update(
        {
            'user_id': model.get('user_id'),
            'app_id': model.get('app_id'),
            'model_id': model.get('id'),
            'model_version_id': builder.model_version_id,
        }
    )

    # Select compute options
    compute_config = select_compute_option(user_id=state['user_id'])

    # Get deployment configuration
    print("\n⌨️  Enter Deployment Configuration:")
    deployment_id = get_user_input(
        "Enter deployment ID", default=f"deploy-{state['model_id']}-{uuid.uuid4().hex[:6]}"
    )
    min_replicas = int(get_user_input("Enter minimum replicas", default="1"))
    max_replicas = int(get_user_input("Enter maximum replicas", default="5"))

    print("\n⏳ Deploying model...")
    success = deploy_model(
        model_id=state['model_id'],
        app_id=state['app_id'],
        user_id=state['user_id'],
        deployment_id=deployment_id,
        model_version_id=state['model_version_id'],
        nodepool_id=compute_config['nodepool_id'],
        compute_cluster_id=compute_config['compute_cluster_id'],
        cluster_user_id=compute_config['cluster_user_id'],
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    if success:
        state.update(
            {
                'deployed': True,
                'deployment_id': deployment_id,
                'nodepool_id': compute_config['nodepool_id'],
            }
        )
        print("Model deployed successfully! You can test it now.")
        time.sleep(2)  # Give some time for the deployment to stabilize
    else:
        logger.warning("Deployment failed. Initiating backtrack & cleanup.")
        backtrack_workflow(state)
        return

    """
    # NOTE: Backtrack & cleanup option for users is disabled.
    # Reason: The prompt is ambiguous and could unintentionally delete deployments or model versions.

    if get_yes_no_input("\n🗑️ Do you want to backtrack and clean up?", True):
        backtrack_workflow(state)

    """


def delete_model_deployment(deployment_id, user_id, nodepool_id=None):
    """
    Delete a model deployment on Clarifai platform.

    Args:
        deployment_id (str): The ID of the deployment to be deleted.
        nodepool_id (str): The ID of the nodepool where the deployment resides.
        user_id (str): The Clarifai user ID (usually owner of the deployment).
    """

    # Instantiate the Nodepool object with given IDs
    nodepool = Nodepool(nodepool_id=nodepool_id, user_id=user_id)
    # The delete_deployments method expects a list of deployment IDs
    try:
        nodepool.delete_deployments([deployment_id])
        print(f"✅ Deployment '{deployment_id}' has been successfully deleted.")
        return True
    except Exception as e:
        print(f"❌ Failed to delete deployment '{deployment_id}': {e}")
        return False


def delete_model_version(
    model_url=None, model_id=None, app_id=None, user_id=None, model_version_id=None
):
    """
    Delete a specific version of a model on Clarifai platform.
    Args:
        model_url (str): The full Clarifai model URL (optional if model_id is provided).
        model_id (str): The ID of the model (optional if model_url is provided).
        app_id (str): The ID of the application the model belongs to.
        user_id (str): The ID of the user who owns the model.
        model_version_id (str): The ID of the model version to be deleted.
    """
    if not model_version_id:
        raise UserError("You must specify a model_version_id to delete.")
    if model_url and model_id:
        raise UserError("You can only specify one of url or model_id.")
    if not model_url and not model_id:
        raise UserError("You must specify one of url or model_id.")
    if model_url:
        user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(model_url)
    model = Model(model_id=model_id, app_id=app_id, user_id=user_id)
    try:
        model.delete_version(version_id=model_version_id)
        print(f"✅ Model version '{model_version_id}' successfully deleted.")
        return True
    except Exception as e:
        print(f"❌ Failed to delete model version '{model_version_id}': {e}")
        return False


def backtrack_workflow(state):
    """Handle backtracking when operations fail."""
    print("\n🔄 Starting backtrack process...")

    # Delete deployment if it was created
    if state.get('deployed') and state.get('deployment_id'):
        if get_yes_no_input("Do you want to delete the deployment?", True):
            success = delete_model_deployment(
                deployment_id=state['deployment_id'],
                user_id=state['user_id'],
                nodepool_id=state.get('nodepool_id'),
            )
            if success:
                state['deployed'] = False

    # Delete model version if it was uploaded
    if state.get('uploaded') and state.get('model_version_id'):
        if get_yes_no_input("Do you want to delete the model version?", False):
            success = delete_model_version(
                model_id=state['model_id'],
                app_id=state['app_id'],
                user_id=state['user_id'],
                model_version_id=state['model_version_id'],
            )
            if success:
                state['uploaded'] = False
                state['model_version_id'] = None
