import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.utils.logging import logger


def get_secrets_path() -> Optional[Path]:
    path = os.environ.get("CLARIFAI_SECRETS_PATH", None)
    return Path(path) if path else None


def load_secrets(path: Path) -> Optional[dict[str, str]]:
    """load_secrets reads .env style secret files, sets them as environment variables, and
    returns the added variables.
    Args:
        path (Path): Path to the directory containing secrets files.
    Returns:
        dict[str, str] | None: Dict of loaded environment variables, or None if the file does not exist.
    """
    variables = get_env_variable(path)
    if variables is not None:
        set_env_variable(variables)
        return variables
    return None


def set_env_variable(variables: dict[str, str]) -> None:
    for key, value in variables.items():
        os.environ[key] = value


def get_env_variable(path: Path) -> Optional[dict[str, str]]:
    """get_env_variable reads .env style secret files and returns variables to be added to the environment.
    Args:
        path (Path): Path to the secrets directory.
    Returns:
        dict[str, str] | None: Dictionary of environment variable keys and values, or None if the files do not exist.
    """
    if not path.exists() or not path.is_dir():
        return None
    loaded_keys = {}
    for secret_dir in path.iterdir():
        if not secret_dir.is_dir():
            continue
        secrets_file_path = secret_dir / secret_dir.name
        if secrets_file_path.exists() and secrets_file_path.is_file():
            secrets = read_secrets_file(secrets_file_path)
            if secrets:
                loaded_keys.update(secrets)
    return loaded_keys


def read_secrets_file(path: Path) -> Optional[dict[str, str]]:
    """Read secrets from a single .env formatted file with robust error handling."""
    if not path.exists() or not path.is_file():
        logger.warning(f"Secret file does not exist or is not a file: {path}")
        return None
    loaded_keys = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    logger.warning(f"Invalid line format in {path}:{line_num}: {line}")
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:  # Only add non-empty keys
                    loaded_keys[key] = value
                    logger.debug(f"Loaded secret key: {key}")
    except (IOError, OSError, UnicodeDecodeError) as e:
        logger.error(f"Error reading secrets file {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading secrets file {path}: {e}")
        return None

    return loaded_keys if loaded_keys else None


def start_secrets_watcher(
    secrets_path: Path, reload_callback: Callable, interval: float = 10
) -> Thread:
    """start_secrets_watcher starts a background thread that watches the secret file directory for changes
    and calls the reload_callback when changes are detected.

    Args:
        secrets_path (Path): Path to the secrets file directory.
        reload_callback (Callable): Callback function to call when the file changes.
        interval (float, optional): Interval to wait before checking again. Defaults to 10.
    """

    def watch_loop():
        previous_state = None

        while True:
            current_state = {}

            # Build current state of all secret files
            if secrets_path.exists():
                for secret_dir in secrets_path.iterdir():
                    if not secret_dir.is_dir():
                        continue
                    try:
                        filepath = secret_dir / secret_dir.name
                        if filepath.exists() and filepath.is_file():
                            current_state[secret_dir.name] = filepath.stat().st_mtime
                    except Exception as e:
                        logger.error(f"Error checking secret file {secret_dir.name}: {e}")

            # Trigger callback if state changed (but not on first run)
            if previous_state is not None and current_state != previous_state:
                try:
                    logger.info("Secrets changed, calling reload callback...")
                    reload_callback()
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")

            previous_state = current_state
            time.sleep(interval)

    watcher_thread = Thread(target=watch_loop, daemon=True)
    watcher_thread.start()
    return watcher_thread


def inject_secrets(request: Optional[service_pb2.PostModelOutputsRequest]) -> None:
    """inject_secrets injects secrets into the request's model version output info params.
    The request is modified in place.

    Args:
        request (service_pb2.PostModelOutputsRequest): The request to inject secrets into.
    """
    if request is None:
        return

    if secrets_path := get_secrets_path():
        # Since only env type secrets are injected into the shared volume, we can read them directly.
        variables = get_env_variable(secrets_path)
    else:
        # If no secrets path is set, assume no secrets and return the request as is.
        return

    if not request.HasField("model"):
        request.model.CopyFrom(resources_pb2.Model())
    if not request.model.HasField("model_version"):
        request.model.model_version.CopyFrom(resources_pb2.ModelVersion())
    if not request.model.model_version.HasField("output_info"):
        request.model.model_version.output_info.CopyFrom(resources_pb2.OutputInfo())
    if not request.model.model_version.output_info.HasField("params"):
        request.model.model_version.output_info.params.CopyFrom(struct_pb2.Struct())

    if variables:
        request.model.model_version.output_info.params.update(variables)
    return


def get_secrets(
    request: Optional[service_pb2.PostModelOutputsRequest],
) -> dict[str, Any]:
    """get_secrets extracts and returns the secrets from the request's model version output info params and environment.

    Args:
        request (Optional[service_pb2.PostModelOutputsRequest]): The request from which to extract secrets.
    """
    params = {}
    env_params = {}
    req_params = {}

    if request is not None:
        req_params = get_request_secrets(request)

    if secrets_path := get_secrets_path():
        # Since only env type secrets are injected into the shared volume, we can read them directly.
        env_params = get_env_variable(secrets_path)
    if env_params:
        params.update(env_params)
    if req_params:
        params.update(req_params)
    return params


def get_request_secrets(request: service_pb2.PostModelOutputsRequest) -> Optional[dict[str, Any]]:
    if (
        request.HasField("model")
        and request.model.HasField("model_version")
        and request.model.model_version.HasField("output_info")
        and request.model.model_version.output_info.HasField("params")
    ):
        return MessageToDict(request.model.model_version.output_info.params)
    return None


def get_secret(param_name: str) -> Optional[str]:
    """get_secret retrieves a secret value from environment variables
    Args:
        param_name (str): Name of the secret to retrieve.
    Returns:
        Optional[str]: The value of the secret if found, otherwise None.
    """
    env_value = os.environ.get(param_name) or os.environ.get(param_name.upper())
    if env_value:
        return env_value
    return None
