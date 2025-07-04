import os
import re
import shutil
import subprocess
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


def format_github_repo_url(github_repo):
    """Format GitHub repository URL to a standard format."""
    if github_repo.startswith('http'):
        return github_repo
    elif '/' in github_repo and not github_repo.startswith('git@'):
        # Handle "user/repo" format
        return f"https://github.com/{github_repo}.git"
    else:
        return github_repo


def clone_github_repo(repo_url, target_dir, github_pat=None, branch=None):
    """Clone a GitHub repository with optional GitHub PAT authentication and branch specification."""
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

    # Handle authentication with GitHub PAT
    if github_pat:
        # Parse the URL and validate the hostname
        parsed_url = urllib.parse.urlparse(repo_url)
        if parsed_url.hostname == "github.com":
            # Insert GitHub PAT into the URL for authentication
            authenticated_url = f"https://{github_pat}@{parsed_url.netloc}{parsed_url.path}"
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
