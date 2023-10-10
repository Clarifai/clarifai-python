import os

file_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "VERSION"))
with open(file_path, "r") as f:
  version = f.read().strip()

CLIENT_VERSION = f"{version}"
OS_VER = os.sys.platform
PYTHON_VERSION = '.'.join(
    map(str, [os.sys.version_info.major, os.sys.version_info.minor, os.sys.version_info.micro]))
