import os

CLIENT_VERSION = '9.7.6'
OS_VER = os.sys.platform
PYTHON_VERSION = '.'.join(
    map(str, [os.sys.version_info.major, os.sys.version_info.minor, os.sys.version_info.micro]))
