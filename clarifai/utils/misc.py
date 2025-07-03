import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import uuid
from typing import Any, Dict, List

from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.errors import UserError
from clarifai.utils.constants import HOME_PATH
from clarifai.utils.logging import logger

RETRYABLE_CODES = [
    status_code_pb2.MODEL_DEPLOYING,
    status_code_pb2.MODEL_LOADING,
    status_code_pb2.MODEL_BUSY_PLEASE_RETRY,
]

DEFAULT_CONFIG = HOME_PATH / '.config/clarifai/config'


def status_is_retryable(status_code: int) -> bool:
    """Check if a status code is retryable."""
    return status_code in RETRYABLE_CODES


class Chunker:
    """Split an input sequence into small chunks."""

    def __init__(self, seq: List, size: int) -> None:
        self.seq = seq
        self.size = size

    def chunk(self) -> List[List]:
        """Chunk input sequence."""
        return [self.seq[pos : pos + self.size] for pos in range(0, len(self.seq), self.size)]


class BackoffIterator:
    """Iterator that returns a sequence of backoff values."""

    def __init__(self, count=0):
        self.count = count

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return 0.1 * (1.3**self.count)


def get_from_dict_or_env(key: str, env_key: str, **data) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key)


def get_from_env(key: str, env_key: str) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        raise UserError(
            f"Did not find `{key}`, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


def concept_relations_accumulation(
    relations_dict: Dict[str, Any], subject_concept: str, object_concept: str, predicate: str
) -> Dict[str, Any]:
    """Append the concept relation to relations dict based on its predicate.

    Args:
        relations_dict (dict): A dict of concept relations info.
    """
    if predicate == 'hyponym':
        if object_concept in relations_dict:
            relations_dict[object_concept].append(subject_concept)
        else:
            relations_dict[object_concept] = [subject_concept]
    elif predicate == 'hypernym':
        if subject_concept in relations_dict:
            relations_dict[subject_concept].append(object_concept)
        else:
            relations_dict[subject_concept] = [object_concept]
    else:
        relations_dict[object_concept] = []
        relations_dict[subject_concept] = []
    return relations_dict


def get_uuid(val: int) -> str:
    """Generates a UUID."""
    return uuid.uuid4().hex[:val]


def clean_input_id(input_id: str) -> str:
    """Clean input_id string into a valid input ID"""
    input_id = re.sub('[., /]+', '_', input_id)
    input_id = re.sub('[_]+', '_', input_id)
    input_id = re.sub('[-]+', '-', input_id)
    input_id = input_id.lower().strip('_-')
    input_id = re.sub('[^a-z0-9-_]+', '', input_id)
    return input_id


def normalize_github_repo_url(github_repo):
    """Normalize GitHub repository URL to a standard format."""
    if github_repo.startswith('http'):
        return github_repo
    elif '/' in github_repo and not github_repo.startswith('git@'):
        # Handle "user/repo" format
        return f"https://github.com/{github_repo}.git"
    else:
        return github_repo


def clone_github_repo(repo_url, target_dir, pat=None, branch=None):
    """Clone a GitHub repository with optional PAT authentication and branch specification."""
    # Handle local file paths - just copy instead of cloning
    if os.path.exists(repo_url):
        try:
            shutil.copytree(repo_url, target_dir, ignore=shutil.ignore_patterns('.git'))
            logger.info(f"Successfully copied local repository from {repo_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy local repository: {e}")
            return False
    
    cmd = ["git", "clone"]
    
    # Add branch specification if provided
    if branch:
        cmd.extend(["-b", branch])
    
    # Handle authentication with PAT
    if pat:
        # Parse the URL and validate the hostname
        parsed_url = urllib.parse.urlparse(repo_url)
        if parsed_url.hostname == "github.com":
            # Insert PAT into the URL for authentication
            authenticated_url = f"https://{pat}@{parsed_url.netloc}{parsed_url.path}"
            cmd.append(authenticated_url)
        else:
            logger.error(f"Invalid repository URL: {repo_url}")
            return False
    else:
        cmd.append(repo_url)
    
    cmd.append(target_dir)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if branch:
            logger.info(f"Successfully cloned repository from {repo_url} (branch: {branch})")
        else:
            logger.info(f"Successfully cloned repository from {repo_url}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        return False


def find_model_structure(repo_dir):
    """Find a valid model structure in the cloned repository."""
    # Look for directories that contain model.py and config.yaml
    for root, dirs, files in os.walk(repo_dir):
        # Skip .git directory
        if '.git' in root:
            continue
            
        # Check if this directory has model structure
        has_model_py = False
        has_config_yaml = False
        has_version_dir = False
        
        # Check for model.py in subdirectories (like 1/model.py)
        for d in dirs:
            if d.isdigit():  # Version directory like "1"
                version_dir = os.path.join(root, d)
                if os.path.exists(os.path.join(version_dir, "model.py")):
                    has_model_py = True
                    has_version_dir = True
                    break
        
        # Check for config.yaml in the current directory
        if "config.yaml" in files:
            has_config_yaml = True
            
        if has_model_py and has_config_yaml and has_version_dir:
            return root
    
    return None


def copy_model_structure(source_dir, target_dir):
    """Copy model structure from source to target directory."""
    # Copy all relevant files and directories
    for item in os.listdir(source_dir):
        if item == '.git':
            continue
            
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)
        
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)
    
    logger.info(f"Copied model structure from {source_dir} to {target_dir}")


def install_requirements(requirements_path):
    """Install requirements from requirements.txt if it exists."""
    if os.path.exists(requirements_path):
        try:
            cmd = ["python", "-m", "pip", "install", "-r", requirements_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Successfully installed requirements from the GitHub repository")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install requirements: {e.stderr}")
            return False
    return False
