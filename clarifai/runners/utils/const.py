import os

registry = os.environ.get('CLARIFAI_BASE_IMAGE_REGISTRY', 'public.ecr.aws/clarifai-models')

PYTHON_BASE_IMAGE = registry + '/python-base:{python_version}'
TORCH_BASE_IMAGE = registry + '/torch:{torch_version}-py{python_version}-cuda{cuda_version}'

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.11', '3.12', '3.13']

DEFAULT_PYTHON_VERSION = 3.12

# List of available torch images
AVAILABLE_TORCH_IMAGES = [
    '2.2.2-py3.11-cuda121',
    '2.3.1-py3.11-cuda121',
    '2.4.0-py3.11-cuda121',
    '2.4.0-py3.11-cuda124',
    '2.4.1-py3.11-cuda121',
    '2.4.1-py3.11-cuda124',
    '2.5.1-py3.11-cuda121',
    '2.5.1-py3.11-cuda124',
    '2.2.2-py3.12-cuda121',
    '2.3.1-py3.12-cuda121',
    '2.4.0-py3.12-cuda121',
    '2.4.0-py3.12-cuda124',
    '2.4.1-py3.12-cuda121',
    '2.4.1-py3.12-cuda124',
    '2.5.1-py3.12-cuda121',
    '2.5.1-py3.12-cuda124',
    # '2.2.2-py3.13-cuda121',
    # '2.3.1-py3.13-cuda121',
    # '2.4.0-py3.13-cuda121',
    # '2.4.0-py3.13-cuda124',
    # '2.4.1-py3.13-cuda121',
    # '2.4.1-py3.13-cuda124',
    # '2.5.1-py3.13-cuda121',
    # '2.5.1-py3.13-cuda124',
]
CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
]
