import importlib.util
import os

from clarifai import __version__

CLIENT_VERSION = __version__
OS_VER = os.sys.platform
PYTHON_VERSION = '.'.join(
    map(str, [os.sys.version_info.major, os.sys.version_info.minor, os.sys.version_info.micro])
)


def get_latest_version_from_pypi():
    """
    Fetch the latest version of the clarifai package from PyPI.

    Returns:
        str: The latest version string, or None if the request fails.
    """
    # Check if requests is installed
    if importlib.util.find_spec("requests") is None:
        return None

    try:
        import requests

        response = requests.get("https://pypi.org/pypi/clarifai/json", timeout=5)
        if response.status_code == 200:
            return response.json().get("info", {}).get("version")
        return None
    except Exception:
        return None
