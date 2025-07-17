import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
from clarifai_grpc.grpc.api.status import status_code_pb2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def get_from_dict_env_or_config(key: str, env_key: str, **data) -> str:
    """Get a value from a dictionary, environment variable, or CLI config context."""
    # First try the provided data/kwargs
    if key in data and data[key]:
        return data[key]

    # Then try environment variables
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]

    # Finally try CLI config context as fallback
    try:
        from clarifai.utils.config import Config
        from clarifai.utils.constants import DEFAULT_CONFIG

        config = Config.from_yaml(filename=DEFAULT_CONFIG)
        current_context = config.current

        # Convert env_key to the attribute name expected by Context
        # e.g., CLARIFAI_PAT -> pat, CLARIFAI_USER_ID -> user_id, CLARIFAI_API_BASE -> api_base
        if env_key == "CLARIFAI_PAT":
            attr_name = "pat"
        elif env_key == "CLARIFAI_USER_ID":
            attr_name = "user_id"
        elif env_key == "CLARIFAI_API_BASE":
            attr_name = "api_base"
        else:
            # For other cases, convert CLARIFAI_SOMETHING to something
            attr_name = env_key.replace("CLARIFAI_", "").lower()

        if hasattr(current_context, attr_name):
            value = getattr(current_context, attr_name)
            if value:
                return value
    except Exception:
        # If CLI config loading fails, fall through to raise error
        pass

    # If all methods fail, raise an error suggesting clarifai login
    raise UserError(
        f"Configuration Required. Could not find '{key}'. Please provide it in one of the following ways:\n\n"
        f"- Pass '{key}' as a named parameter to your function.\n"
        f"- Set the {env_key} environment variable in your environment.\n"
        f"- Run `clarifai login` in your terminal to configure CLI authentication."
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


class GitHubDownloader:
    def __init__(
        self, max_retries: int = 3, backoff_factor: float = 0.3, github_token: str = None
    ):
        self.session = requests.Session()
        self.github_token = github_token

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update({'User-Agent': 'GitHub-Folder-Downloader/1.0'})

        if self.github_token:
            self.session.headers.update({'Authorization': f'token {self.github_token}'})

    def expected_folder_structure(self) -> List[Dict[str, Any]]:
        return [
            {"name": "1", "type": "dir", "children": [{"name": "model.py", "type": "file"}]},
            {"name": "config.yaml", "type": "file"},
            {"name": "requirements.txt", "type": "file"},
        ]

    def _format_expected_structure(self):
        """Format the expected structure as a nice tree view."""
        tree_str = ""
        tree_str += "Expected folder structure:\n"
        tree_str += "├── 1/\n"
        tree_str += "│   └── model.py\n"
        tree_str += "├── requirements.txt\n"
        tree_str += "└── config.yaml\n"
        return tree_str

    def parse_github_url(self, url: str) -> Tuple[str, str, str, str]:
        try:
            parsed = urlparse(url)

            if parsed.netloc not in ['github.com', 'www.github.com']:
                raise ValueError("URL must be a GitHub repository URL")

            path_parts = [p for p in parsed.path.strip('/').split('/') if p]

            if len(path_parts) < 2:
                raise ValueError("Invalid GitHub repository URL format")

            owner = path_parts[0]
            repo = path_parts[1]

            if len(path_parts) >= 4 and path_parts[2] in ['tree', 'blob']:
                branch = path_parts[3]
                folder_path = '/'.join(path_parts[4:]) if len(path_parts) > 4 else ''
            elif len(path_parts) >= 3:
                branch = path_parts[2]
                folder_path = '/'.join(path_parts[3:]) if len(path_parts) > 3 else ''
            else:
                branch = 'main'
                folder_path = ''

            return owner, repo, branch, folder_path

        except Exception as e:
            logger.error(f"Failed to parse GitHub URL: {e}")
            sys.exit(1)

    def get_folder_contents(self, owner: str, repo: str, path: str, branch: str = 'main') -> list:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        params = {'ref': branch} if branch else {}

        try:
            response = self.session.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise requests.RequestException("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise requests.RequestException(
                "Connection error. Please check your internet connection."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                token_msg = (
                    ""
                    if self.github_token
                    else " For private repositories, use the github_token parameter."
                )
                raise requests.RequestException(
                    f"Folder not found: {path}. Check if path exists or if the repository is private.{token_msg}"
                )
            elif e.response.status_code == 401 or e.response.status_code == 403:
                token_msg = (
                    " The provided GitHub token may be invalid or have insufficient permissions."
                    if self.github_token
                    else " For private repositories, use the github_token parameter."
                )
                raise requests.RequestException(f"Authentication error: {e}.{token_msg}")
            else:
                raise requests.RequestException(f"API request failed: {e}")
        except requests.exceptions.RequestException as e:
            token_msg = (
                ""
                if self.github_token
                else " For private repositories, use the github_token parameter."
            )
            raise requests.RequestException(f"API request failed: {e}.{token_msg}")

    def validate_remote_structure(
        self,
        owner: str,
        repo: str,
        path: str,
        branch: str,
        expected_structure: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        validation_result = {
            'valid': True,
            'missing_files': [],
            'missing_dirs': [],
            'warnings': [],
            'remote_contents': [],
        }

        try:
            remote_contents = self.get_folder_contents(owner, repo, path, branch)
            validation_result['remote_contents'] = remote_contents

            remote_items = {item['name']: item['type'] for item in remote_contents}

            for item in expected_structure:
                item_name = item['name']
                item_type = item.get('type', 'file')

                if item_name not in remote_items:
                    if item_type == 'file':
                        validation_result['missing_files'].append(item_name)
                    else:
                        validation_result['missing_dirs'].append(item_name)
                    validation_result['valid'] = False
                elif remote_items[item_name] != item_type:
                    validation_result['warnings'].append(
                        f"Item '{item_name}' exists but is a {remote_items[item_name]} instead of {item_type}"
                    )
                    validation_result['valid'] = False

            expected_names = {item['name'] for item in expected_structure}
            unexpected_items = [name for name in remote_items.keys() if name not in expected_names]
            if unexpected_items:
                validation_result['warnings'].append(
                    f"Unexpected items found: {', '.join(unexpected_items)}"
                )

        except requests.RequestException as e:
            validation_result['valid'] = False
            validation_result['warnings'].append(f"Failed to access remote repository: {e}")

        return validation_result

    def download_file(self, download_url: str, local_path: str) -> None:
        try:
            response = self.session.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0 and total_size > 1024 * 1024:
                            progress = (downloaded_size / total_size) * 100
                            logger.info(
                                f"\rDownloading: {os.path.basename(local_path)} - {progress:.1f}%",
                                end='',
                                flush=True,
                            )

            if total_size > 1024 * 1024:
                logger.info()

            logger.info(f"Downloaded: {local_path}")

        except requests.exceptions.Timeout:
            logger.info(f"Timeout downloading {local_path}. Skipping...")
        except requests.exceptions.ConnectionError:
            logger.info(f"Connection error downloading {local_path}. Skipping...")
        except Exception as e:
            logger.info(f"Failed to download {local_path}: {e}")

    def process_folder(
        self, owner: str, repo: str, path: str, local_base_path: str, branch: str = 'main'
    ) -> None:
        try:
            contents = self.get_folder_contents(owner, repo, path, branch)

            if not contents:
                logger.info(f"Info: Empty folder - {path}")
                return

            for item in contents:
                item_name = item['name']
                item_path = os.path.join(local_base_path, item_name)

                if item['type'] == 'file':
                    self.download_file(item['download_url'], item_path)

                elif item['type'] == 'dir':
                    os.makedirs(item_path, exist_ok=True)
                    logger.info(f"Created directory: {item_path}")

                    new_path = f"{path}/{item_name}" if path else item_name
                    self.process_folder(owner, repo, new_path, item_path, branch)

        except requests.exceptions.RequestException as e:
            if "Folder not found" in str(e):
                logger.error(f"Error: Folder not found - {path}")
                raise
            else:
                logger.error(f"Error accessing folder {path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error processing folder {path}: {e}")
            raise

    def validate_folder_structure(
        self, folder_path: str, expected_structure: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        validation_result = {
            'valid': True,
            'missing_files': [],
            'missing_dirs': [],
            'warnings': [],
        }

        if not os.path.exists(folder_path):
            validation_result['valid'] = False
            validation_result['warnings'].append(f"Folder {folder_path} does not exist")
            return validation_result

        for item in expected_structure:
            item_name = item['name']
            item_type = item.get('type', 'file')
            item_path = os.path.join(folder_path, item_name)

            if item_type == 'file':
                if not os.path.isfile(item_path):
                    validation_result['missing_files'].append(item_name)
                    validation_result['valid'] = False
            elif item_type == 'dir':
                if not os.path.isdir(item_path):
                    validation_result['missing_dirs'].append(item_name)
                    validation_result['valid'] = False

        return validation_result

    def download_github_folder(
        self,
        url: str,
        output_dir: str,
        github_token: str = None,
        validate_structure: bool = False,
        pre_validate: bool = True,
        strict_validation: bool = False,
    ) -> None:
        logger.info(f"Parsing GitHub URL: {url}")

        # Update token if provided as a parameter
        if github_token:
            self.github_token = github_token
            self.session.headers.update({'Authorization': f'token {github_token}'})

        try:
            owner, repo, branch, folder_path = self.parse_github_url(url)
            logger.info(f"Repository: {owner}/{repo}")
            logger.info(f"Branch: {branch}")
            logger.info(f"Folder: {folder_path or 'root'}")

            expected_structure = self.expected_folder_structure() if pre_validate else None

            if expected_structure:
                logger.info("\nValidating remote folder structure...")
                remote_validation = self.validate_remote_structure(
                    owner, repo, folder_path, branch, expected_structure
                )

                if not remote_validation['valid']:
                    logger.error("Remote structure validation failed!")

                    if remote_validation['missing_files']:
                        logger.error(
                            f"Missing files: {', '.join(remote_validation['missing_files'])}"
                        )

                    if remote_validation['missing_dirs']:
                        logger.error(
                            f"Missing directories: {', '.join(remote_validation['missing_dirs'])}"
                        )

                    if remote_validation['warnings']:
                        for warning in remote_validation['warnings']:
                            logger.error(f"Warning: {warning}")

                    # Print the expected structure in a nice format
                    tree_view = self._format_expected_structure()
                    logger.info("\nThe repository must have the following structure:")
                    logger.info(tree_view)

                    logger.error(
                        "Download cancelled: Repository structure does not match the expected format."
                    )
                    sys.exit(1)  # Exit without proceeding with download
                else:
                    logger.info("Remote structure validation passed!")

            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

            logger.info("\nStarting download...")
            start_time = time.time()
            try:
                self.process_folder(owner, repo, folder_path, output_dir, branch)

                elapsed_time = time.time() - start_time
                logger.info(f"\nDownload completed in {elapsed_time:.2f} seconds")
                logger.info(f"Files saved to: {os.path.abspath(output_dir)}")

                if validate_structure and expected_structure:
                    logger.info("\nValidating downloaded folder structure...")
                    validation_result = self.validate_folder_structure(
                        output_dir, expected_structure
                    )

                    if validation_result['valid']:
                        logger.info("Folder structure post validation passed!")
                    else:
                        logger.error("Folder structure validation failed!")

                        if validation_result['missing_files']:
                            logger.info(
                                f"Missing files: {', '.join(validation_result['missing_files'])}"
                            )

                        if validation_result['missing_dirs']:
                            logger.info(
                                f"Missing directories: {', '.join(validation_result['missing_dirs'])}"
                            )

                        if validation_result['warnings']:
                            for warning in validation_result['warnings']:
                                logger.info(f"Warng: {warning}")
            except requests.RequestException as e:
                # Critical error - the main folder cannot be processed
                logger.error(
                    f"\nDownload failed: {e}, \n No files were downloaded to: {os.path.abspath(output_dir)}"
                )
                sys.exit(1)

        except ValueError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)
