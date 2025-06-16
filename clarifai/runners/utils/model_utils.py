import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import weakref

import psutil
import requests

from clarifai.utils.logging import logger

_process_socket_map = weakref.WeakKeyDictionary()


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


def reserve_port(host, start=30000, end=40000):
    """
    Reserve an available port by trying to bind a socket.
    Returns a tuple (port, lock_socket) where `lock_socket` is kept open to hold the lock.
    """
    if not isinstance(host, str):
        raise ValueError("host must be a string")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("start and end must be integers")
    if start < 0 or end < 0 or start > end:
        raise ValueError("invalid port range")
    candidates = list(range(start, end))
    random.shuffle(candidates)

    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Attempt to bind to the port on localhost
            sock.bind((host, port))
            return port, sock
        except socket.error as e:
            logger.debug(f"Failed to bind to port {port}: {e}")
            sock.close()  # Failed to bind, try next port
            continue
    raise RuntimeError("No free port available.")


def release_port(lock_socket):
    """Release the reserved port by closing the lock socket.

    Args:
        lock_socket (socket.socket): The socket object holding the port lock.

    Raises:
        ValueError: If lock_socket is None or not a socket object.
    """
    if lock_socket is None:
        raise ValueError("lock_socket cannot be None")
    if not isinstance(lock_socket, socket.socket):
        raise ValueError("lock_socket must be a socket object")

    try:
        lock_socket.close()
    except socket.error as e:
        logger.error(f"Error closing socket: {e}")
        raise


def execute_shell_command(
    command: str,
) -> subprocess.Popen:
    """Execute a shell command and return its process handle.

    Args:
        command (str): The shell command to execute.

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
        process = subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)

        return process
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to execute command: {e}")
        raise


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """Launch the server using the given command.

    Args:
        command (str): The command to launch the server.
        host (str): The host address to bind to.
        port (int, optional): The port to use. If None, a free port is reserved.

    Returns:
        tuple: (process, port) where process is the subprocess.Popen object and port is the used port.

    Raises:
        ValueError: If command is empty or host is invalid.
        RuntimeError: If port reservation fails.
    """
    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")
    if not isinstance(host, str):
        raise ValueError("host must be a string")

    if port is None and '--port' not in command:
        try:
            port, lock_socket = reserve_port(host)
            command = f"{command} --port {port}"
        except Exception as e:
            logger.error(f"Failed to reserve port: {e}")
            raise RuntimeError(f"Failed to reserve port: {e}")
    else:
        lock_socket = None

    try:
        process = execute_shell_command(command)
    except subprocess.SubprocessError as e:
        if lock_socket is not None:
            release_port(lock_socket)
        raise RuntimeError(f"Failed to launch server: {e}")

    if lock_socket is not None:
        _process_socket_map[process] = lock_socket

    return process, port


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    kill_process_tree(process.pid)

    lock_socket = _process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)


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
