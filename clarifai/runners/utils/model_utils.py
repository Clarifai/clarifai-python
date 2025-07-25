import inspect
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

import psutil
import requests
from clarifai_grpc.grpc.api import service_pb2

from clarifai.utils.logging import logger


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes.

    Args:
        parent_pid (int): The PID of the parent process to kill. If None, uses current process.
        include_parent (bool): Whether to kill the parent process as well.
        skip_pid (int, optional): PID to skip when killing child processes.

    Raises:
        psutil.AccessDenied: If process cannot be accessed due to permissions.
        psutil.NoSuchProcess: If process does not exist.
    """
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        logger.warning(f"Process {parent_pid} does not exist")
        return
    except psutil.AccessDenied:
        logger.error(f"Cannot access process {parent_pid} due to permissions")
        raise

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Failed to kill child process {child.pid}: {e}")

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            if hasattr(signal, 'SIGQUIT'):
                itself.send_signal(signal.SIGQUIT)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Failed to kill parent process {parent_pid}: {e}")


def execute_shell_command(command: str, stdout=None, stderr=subprocess.STDOUT) -> subprocess.Popen:
    """Execute a shell command and return its process handle.

    Args:
        command (str): The shell command to execute.
        stdout : Verbose logging control,
        stderr : Verbose error logging control

    Returns:
        subprocess.Popen: Process handle for the executed command.

    Raises:
        ValueError: If command is empty or invalid.
        subprocess.SubprocessError: If command execution fails.
    """
    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")

    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = shlex.split(command)

    try:
        process = subprocess.Popen(parts, text=True, stdout=stdout, stderr=stderr)

        return process
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to execute command: {e}")
        raise


def terminate_process(process):
    """
    Terminate the process
    """
    kill_process_tree(process.pid)


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                logger.info(
                    """\n
                    NOTE: Typically, the server runs in a separate terminal.
                    In this notebook, we run the server and notebook code together, so their outputs are combined.
                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
                    We are running those notebooks in a CI parallel environment, so the throughput is not representative of the actual performance.
                    """
                )
                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


def is_proto_style_method(method):
    """
    Determines if the given method is likely an old-style proto method:
    - Has a 'request' parameter after 'self'
    - Optionally, returns a known proto response type
    """
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())

        # Must have at least 'self' and one argument
        if len(params) < 2:
            return False

        # First parameter should be 'self'
        if params[0].name != 'self':
            return False
        # Second param typically should be named 'request'
        request_param = params[1]
        if request_param.name != 'request':
            return False
        # Optionally: check annotation is a proto type
        # (If signature is incomplete, this part will gracefully fall through)
        return_annotation = sig.return_annotation
        # If type annotation is available, check it's PostModelOutputsRequest
        if (
            request_param.annotation != inspect.Parameter.empty
            and request_param.annotation != service_pb2.PostModelOutputsRequest
        ):
            return False
        # If return annotation is available, check it's MultiOutputResponse
        if (
            return_annotation != inspect.Signature.empty
            and return_annotation != service_pb2.MultiOutputResponse
        ):
            return False
        if (
            request_param.annotation is inspect.Parameter.empty
            and return_annotation is inspect.Signature.empty
        ):
            return True  # signature OK, even if signature is empty
        return True

    except (ValueError, TypeError):
        return False
