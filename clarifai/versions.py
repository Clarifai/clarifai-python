import os
import sys

from clarifai import __version__

CLIENT_VERSION = __version__
OS_VER = os.sys.platform
PYTHON_VERSION = '.'.join(
    map(str, [os.sys.version_info.major, os.sys.version_info.minor, os.sys.version_info.micro])
)

# Minimum supported Python version
MINIMUM_PYTHON_VERSION = (3, 8)


def validate_python_version():
    """
    Validate that the current Python version meets the minimum requirements.
    Exits with code 1 and an error message if version is below minimum.
    """
    current_version = (sys.version_info.major, sys.version_info.minor)
    
    if current_version < MINIMUM_PYTHON_VERSION:
        min_version_str = '.'.join(map(str, MINIMUM_PYTHON_VERSION))
        current_version_str = '.'.join(map(str, current_version))
        
        print(f"Error: Clarifai requires Python {min_version_str} or higher.", file=sys.stderr)
        print(f"You are currently using Python {current_version_str}.", file=sys.stderr)
        print(f"Please upgrade your Python version to continue.", file=sys.stderr)
        sys.exit(1)
