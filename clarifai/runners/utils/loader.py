import fnmatch
import importlib.util
import json
import os
import shutil

import requests

from clarifai.runners.utils.const import CONCEPTS_REQUIRED_MODEL_TYPE
from clarifai.utils.logging import logger


class HuggingFaceLoader:
    HF_DOWNLOAD_TEXT = "The 'huggingface_hub' package is not installed. Please install it using 'pip install huggingface_hub'."

    def __init__(self, repo_id=None, token=None, model_type_id=None):
        self.repo_id = repo_id
        self.token = token
        self.clarifai_model_type_id = model_type_id
        if token:
            if self.validate_hftoken(token):
                try:
                    from huggingface_hub import login
                except ImportError:
                    raise ImportError(self.HF_DOWNLOAD_TEXT)
                login(token=token)
                logger.info("Hugging Face token validated")
            else:
                self.token = None
                logger.info("Continuing without Hugging Face token")

    @classmethod
    def validate_hftoken(cls, hf_token: str):
        try:
            if importlib.util.find_spec("huggingface_hub") is None:
                raise ImportError(cls.HF_DOWNLOAD_TEXT)
            from huggingface_hub import HfApi

            api = HfApi()
            api.whoami(token=hf_token)
            return True
        except Exception as e:
            logger.error(
                f"Invalid Hugging Face token provided in the config file, this might cause issues with downloading the restricted model checkpoints. Failed reason: {e}"
            )
            return False

    def download_checkpoints(
        self, checkpoint_path: str, allowed_file_patterns=None, ignore_file_patterns=None
    ):
        # throw error if huggingface_hub wasn't installed
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(self.HF_DOWNLOAD_TEXT)
        if os.path.exists(checkpoint_path) and self.validate_download(
            checkpoint_path, allowed_file_patterns, ignore_file_patterns
        ):
            logger.info("Checkpoints already exist")
            return True
        else:
            os.makedirs(checkpoint_path, exist_ok=True)
            try:
                is_hf_model_exists = self.validate_hf_model()
                if not is_hf_model_exists:
                    return False

                self.ignore_patterns = self._get_ignore_patterns()
                if ignore_file_patterns:
                    if self.ignore_patterns:
                        self.ignore_patterns.extend(ignore_file_patterns)
                    else:
                        self.ignore_patterns = ignore_file_patterns

                snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=checkpoint_path,
                    local_dir_use_symlinks=False,
                    allow_patterns=allowed_file_patterns,
                    ignore_patterns=self.ignore_patterns,
                )
                # Remove the `.cache` folder if it exists
                cache_path = os.path.join(checkpoint_path, ".cache")
                if os.path.exists(cache_path) and os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)

            except Exception as e:
                logger.error(f"Error downloading model checkpoints {e}")
                return False
            finally:
                is_downloaded = self.validate_download(
                    checkpoint_path, allowed_file_patterns, ignore_file_patterns
                )
                if not is_downloaded:
                    logger.error("Error validating downloaded model checkpoints")
                    return False
            return True

    def download_config(self, checkpoint_path: str):
        # throw error if huggingface_hub wasn't installed
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(self.HF_DOWNLOAD_TEXT)
        if os.path.exists(checkpoint_path) and os.path.exists(
            os.path.join(checkpoint_path, 'config.json')
        ):
            logger.info("HF model's config.json already exists")
            return True
        os.makedirs(checkpoint_path, exist_ok=True)
        try:
            is_hf_model_exists = self.validate_hf_model()
            if not is_hf_model_exists:
                logger.error("Model %s not found on Hugging Face" % (self.repo_id))
                return False
            hf_hub_download(
                repo_id=self.repo_id, filename='config.json', local_dir=checkpoint_path
            )
        except Exception as e:
            logger.error(f"Error downloading model's config.json {e}")
            return False
        return True

    def validate_hf_model(
        self,
    ):
        # check if model exists on HF
        try:
            from huggingface_hub import file_exists, repo_exists
        except ImportError:
            raise ImportError(self.HF_DOWNLOAD_TEXT)
        if self.clarifai_model_type_id in CONCEPTS_REQUIRED_MODEL_TYPE:
            return repo_exists(self.repo_id) and file_exists(self.repo_id, 'config.json')
        else:
            return repo_exists(self.repo_id)

    def validate_download(
        self, checkpoint_path: str, allowed_file_patterns: list, ignore_file_patterns: list
    ):
        # check if model exists on HF
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            raise ImportError(self.HF_DOWNLOAD_TEXT)
        # Get the list of files on the repo
        repo_files = list_repo_files(self.repo_id, token=self.token)

        # Get the list of files on the repo that are allowed
        if allowed_file_patterns:

            def should_allow(file_path):
                return any(
                    fnmatch.fnmatch(file_path, pattern) for pattern in allowed_file_patterns
                )

            repo_files = [f for f in repo_files if should_allow(f)]

        self.ignore_patterns = self._get_ignore_patterns()
        if ignore_file_patterns:
            if self.ignore_patterns:
                self.ignore_patterns.extend(ignore_file_patterns)
            else:
                self.ignore_patterns = ignore_file_patterns
        # Get the list of files on the repo that are not ignored
        if getattr(self, "ignore_patterns", None):
            patterns = self.ignore_patterns

            def should_ignore(file_path):
                return any(fnmatch.fnmatch(file_path, pattern) for pattern in patterns)

            repo_files = [f for f in repo_files if not should_ignore(f)]

        # Check if downloaded files match the files we expect (ignoring ignored patterns)
        checkpoint_dir_files = []
        for dp, dn, fn in os.walk(os.path.expanduser(checkpoint_path)):
            checkpoint_dir_files.extend(
                [os.path.relpath(os.path.join(dp, f), checkpoint_path) for f in fn]
            )

        # Validate by comparing file lists
        return (
            len(checkpoint_dir_files) >= len(repo_files)
            and not (len(set(repo_files) - set(checkpoint_dir_files)) > 0)
            and len(repo_files) > 0
        )

    def _get_ignore_patterns(self):
        # check if model exists on HF
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            raise ImportError(self.HF_DOWNLOAD_TEXT)

        # Get the list of files on the repo that are not ignored
        repo_files = list_repo_files(self.repo_id, token=self.token)
        self.ignore_patterns = None
        if any(f.endswith(".safetensors") for f in repo_files):
            self.ignore_patterns = [
                "**/original/*",
                "original/*",
                "**/*.pth",
                "**/*.bin",
                "*.pth",
                "*.bin",
                "**/.cache/*",
            ]
        return self.ignore_patterns

    @classmethod
    def validate_hf_repo_access(cls, repo_id: str, token: str = None) -> bool:
        # check if model exists on HF
        try:
            from huggingface_hub import auth_check
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        except ImportError:
            raise ImportError(cls.HF_DOWNLOAD_TEXT)

        try:
            auth_check(repo_id, token=token)
            logger.info("Hugging Face repo access validated")
            return True
        except GatedRepoError:
            logger.error(
                "Hugging Face repo is gated. Please make sure you have access to the repo."
            )
            return False
        except RepositoryNotFoundError:
            logger.error("Hugging Face repo not found. Please make sure the repo exists.")
            return False

    @staticmethod
    def validate_config(checkpoint_path: str):
        # check if downloaded config.json exists
        return os.path.exists(checkpoint_path) and os.path.exists(
            os.path.join(checkpoint_path, 'config.json')
        )

    @staticmethod
    def validate_concept(checkpoint_path: str):
        # check if downloaded concept exists in hf model
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        labels = config.get('id2label', None)
        if labels:
            return True
        return False

    @staticmethod
    def fetch_labels(checkpoint_path: str):
        # Fetch labels for classification, detection and segmentation models
        config_path = os.path.join(checkpoint_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        labels = config['id2label']
        return labels

    @staticmethod
    def get_huggingface_checkpoint_total_size(repo_name):
        """
        Fetches the JSON data for a Hugging Face model using the API with `?blobs=true`.
        Calculates the total size from the JSON output.

        Args:
            repo_name (str): The name of the model on Hugging Face Hub. e.g. "casperhansen/llama-3-8b-instruct-awq"

        Returns:
            int: The total size in bytes.
        """
        try:
            url = f"https://huggingface.co/api/models/{repo_name}?blobs=true"
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            json_data = response.json()

            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data

            total_size = 0
            for file in data['siblings']:
                total_size += file['size']
            return total_size
        except Exception as e:
            logger.error(f"Error fetching checkpoint size from huggingface.co: {e}")
            return 0
