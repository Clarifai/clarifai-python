from setuptools import setup, find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="clarifai",
    description='Clarifai API Python Client',
    version='2.0.22',
    author='Clarifai',
    maintainer='Robert Wen',
    maintainer_email='robert@clarifai.com',
    url='https://github.com/clarifai/clarifai-python',
    author_email='support@clarifai.com',
    install_requires=required,
    packages=find_packages(),
    license="Apache 2.0",
    scripts=['scripts/clarifai'],
)
