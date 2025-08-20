import os
import time
from contextvars import ContextVar
from pathlib import Path
from threading import Thread
from typing import Callable, Optional

from clarifai.utils.logging import logger

# Context variable to store current request parameters
_current_request_params: ContextVar[dict] = ContextVar('current_request_params', default={})


def get_secrets_path() -> Optional[Path]:
    secrets_path = os.environ.get("CLARIFAI_SECRETS_PATH", None)
    if secrets_path is not None:
        return Path(secrets_path)


def load_secrets_file(path: Path) -> Optional[list[str]]:
    """load_secrets_file reads a .env style secrets file, sets the environment variables, and
    returns the keys of the added variables.
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


def start_secrets_watcher(
    secrets_path: Path, reload_callback: Callable, interval: int = 10
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
