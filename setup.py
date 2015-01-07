try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Clarifai API Python Client',
    'author': 'Clarifai',
    'url': 'https://github.com/clarifai/clarifai_py',
    'author_email': 'support@clarifai.com',
    'version': '0.1',
    'install_requires': [],
    'packages': ['clarifai'],
    'scripts': [],
    'name': 'clarifai-py'
}

setup(**config)