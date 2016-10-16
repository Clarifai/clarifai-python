from setuptools import setup, find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="clarifai",
    description='Clarifai API Python Client',
    version='2.0.6',
    author='Clarifai',
    maintainer='Robert Wen',
    maintainer_email='robert@clarifai.com',
    url='https://github.com/clarifai/clarifai-python',
    author_email='support@clarifai.com',
    install_requires=['future==0.15.2', 'requests==2.11.1', 'configparser==3.5.0'],
    packages=find_packages(),
    license="Apache 2.0",
    scripts=['scripts/clarifai'],
)
