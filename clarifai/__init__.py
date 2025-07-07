import sys

__version__ = "11.6.0"

# Validate Python version when package is imported
MINIMUM_PYTHON_VERSION = (3, 8)
current_version = (sys.version_info.major, sys.version_info.minor)

if current_version < MINIMUM_PYTHON_VERSION:
    min_version_str = '.'.join(map(str, MINIMUM_PYTHON_VERSION))
    current_version_str = '.'.join(map(str, current_version))
    
    print(f"Error: Clarifai requires Python {min_version_str} or higher.", file=sys.stderr)
    print(f"You are currently using Python {current_version_str}.", file=sys.stderr)
    print(f"Please upgrade your Python version to continue.", file=sys.stderr)
    sys.exit(1)
