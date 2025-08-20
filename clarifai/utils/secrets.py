import os
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Thread

from clarifai.utils.logging import logger

# Context variable to store current request parameters
_current_request_params: ContextVar[dict] = ContextVar('current_request_params', default={})


def load_secrets_file(path: Path) -> list[str] | None:
    """load_secrets_file loads a .env style secrets file and sets the environment variables.

    Args:
        path (Path): Path to the secrets file.

    Returns:
        list[str] | None: List of loaded environment variable keys, or None if the file does not exist.
    """
    if not path.exists() or not path.is_file():
        return None
    loaded_keys = []
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
            os.environ[key] = value
            loaded_keys.append(key)
    return loaded_keys


def start_secrets_watcher(secrets_path: Path, reload_callback, interval=2.0):
    """Start watching secrets file for changes."""

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


def populate_params_from_secrets(request_params: dict, secret_params: list[str] = None) -> dict:
    """Populate missing request parameters from environment variables (secrets).

    Args:
        request_params: The current request parameters
        secret_params: List of parameter names to check for in secrets. If None,
                      uses common parameter names.

    Returns:
        Updated request parameters dictionary
    """
    if secret_params is None:
        # Default common secret parameter names
        secret_params = [
            'api_key',
            'token',
            'password',
            'secret',
            'key',
            'openai_api_key',
            'anthropic_api_key',
            'huggingface_token',
            'aws_access_key_id',
            'aws_secret_access_key',
        ]

    updated_params = request_params.copy()

    for param_name in secret_params:
        if param_name not in updated_params or not updated_params[param_name]:
            # Check environment variable (case insensitive)
            env_value = os.environ.get(param_name) or os.environ.get(param_name.upper())
            if env_value:
                updated_params[param_name] = env_value
                logger.debug(f"Populated parameter '{param_name}' from secrets")

    return updated_params


def set_request_context(request_params: dict):
    """Set the current request context for the secrets helper."""
    _current_request_params.set(request_params or {})


def secrets(param_name: str) -> str | None:
    """Helper method to get a secret parameter value.

    This function first checks the current request parameters, then falls back
    to environment variables.

    Args:
        param_name: Name of the parameter/secret to retrieve

    Returns:
        The parameter value, or None if not found
    """
    # First check current request parameters
    current_params = _current_request_params.get({})
    if param_name in current_params and current_params[param_name]:
        return current_params[param_name]

    # Fallback to environment variables (try both cases)
    env_value = os.environ.get(param_name) or os.environ.get(param_name.upper())
    return env_value
