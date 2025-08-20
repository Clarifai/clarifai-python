import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.utils.logging import logger

_secrets_cache = {}
_last_cache_time = 0


def get_secrets_path() -> Optional[Path]:
    secrets_path = os.environ.get("CLARIFAI_SECRETS_PATH", None)
    if secrets_path is not None:
        return Path(secrets_path)


def load_secrets_file(path: Path) -> Optional[dict[str, str]]:
    """load_secrets_file reads a .env style secrets file, sets them as environment variables, and
    returns the keys of the added variables.
    Args:
        path (Path): Path to the secrets file.
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
    """get_env_variable reads a .env style secrets file and returns variables.
    Args:
        path (Path): Path to the secrets file.
    Returns:
        dict[str, str] | None: Dictionary of environment variable keys and values, or None if the file does not exist.
    """
    global _secrets_cache, _last_cache_time
    if not path.exists() or not path.is_file():
        return None
    # Use cache if the file has not changed
    current_mtime = path.stat().st_mtime
    if current_mtime == _last_cache_time and _secrets_cache:
        return _secrets_cache
    loaded_keys = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                loaded_keys[key] = value
    except Exception as e:
        logger.error(f"Error reading secrets file {path}: {e}")
        return None
    _secrets_cache = loaded_keys
    _last_cache_time = current_mtime
    return loaded_keys


def start_secrets_watcher(
    secrets_path: Path, reload_callback: Callable, interval: float = 10
) -> Thread:
    """start_secrets_watcher starts a background thread that watches the secrets file for changes
    and calls the reload_callback when changes are detected.

    Args:
        secrets_path (Path): Path to the secrets file to watch.
        reload_callback (Callable): Callback function to call when the file changes.
        interval (int, optional): Interval to wait before checking again. Defaults to 10.
    """

    def watch_loop():
        last_modified = 0
        while True:
            try:
                if secrets_path.exists():
                    current_modified = secrets_path.stat().st_mtime
                    if current_modified != last_modified and last_modified != 0:
                        logger.info("Secrets file changed, reloading...")
                        reload_callback()
                    last_modified = current_modified
            except Exception as e:
                logger.error(f"Error watching secrets file: {e}")
            time.sleep(interval)

    watcher_thread = Thread(target=watch_loop, daemon=True)
    watcher_thread.start()
    return watcher_thread


def inject_secrets(request: service_pb2.PostModelOutputsRequest) -> None:
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
    if (
        request is not None
        and request.HasField("model")
        and request.model.HasField("model_version")
        and request.model.model_version.HasField("output_info")
        and request.model.model_version.output_info.HasField("params")
    ):
        req_params = MessageToDict(request.model.model_version.output_info.params)
    if secrets_path := get_secrets_path():
        # Since only env type secrets are injected into the shared volume, we can read them directly.
        env_params = get_env_variable(secrets_path)
    if env_params:
        params.update(env_params)
    if req_params:
        params.update(req_params)
    return params


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
