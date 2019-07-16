import sys

from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

has_enum = sys.version_info >= (3, 4)
has_typing = sys.version_info >= (3, 5)

setup(
    name="clarifai",
    description='Clarifai API Python Client',
    version='2.6.2',
    author='Clarifai',
    maintainer='Robert Wen',
    maintainer_email='robert@clarifai.com',
    url='https://github.com/clarifai/clarifai-python',
    author_email='support@clarifai.com',
    install_requires=['future>=0.15, <2',
                      'requests>=2.13, <3',
                      'configparser>=3.5, <4',
                      'jsonschema>=2.5, <3',
                      'grpcio>=1.13.0, <2',
                      'protobuf>=3.6, <4',
                      'googleapis-common-protos>=1.5.0, <2'] +
                     ([] if has_enum else ['enum34>=1.1, <2']) +
                     ([] if has_typing else ['typing>=3.6']),
    packages=find_packages(),
    license="Apache 2.0",
    scripts=['scripts/clarifai'],
)
