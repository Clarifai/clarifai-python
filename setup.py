from setuptools import setup, find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="clarifai",
    description='Clarifai API Python Client',
    version='2.0.4',
    author='Clarifai',
    url='https://github.com/clarifai/clarifai-python',
    author_email='support@clarifai.com',
    install_requires=['six>=1.10.0', 'requests>=2.11.0'],
    packages=find_packages(),
    license="Apache 2.0",
    scripts=[],
)
