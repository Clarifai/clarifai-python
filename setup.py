from setuptools import setup, find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="clarifai",
    description='Clarifai API Python Client',
    version='2.0.27',
    author='Clarifai',
    maintainer='Robert Wen',
    maintainer_email='robert@clarifai.com',
    url='https://github.com/clarifai/clarifai-python',
    author_email='support@clarifai.com',
    install_requires=['future==0.15.2',
                       'requests==2.13.0',
                       'configparser==3.5.0',
                       'jsonschema==2.5.1',
                       'Pillow==2.9.0'],
    packages=find_packages(),
    license="Apache 2.0",
    scripts=['scripts/clarifai'],
)
