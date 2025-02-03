import os

registry = os.environ.get('CLARIFAI_BASE_IMAGE_REGISTRY', 'public.ecr.aws/clarifai-models')

PYTHON_BUILDER_IMAGE = registry + '/python-base:builder-{python_version}'
PYTHON_RUNTIME_IMAGE = registry + '/python-base:runtime-{python_version}'
TORCH_BASE_IMAGE = registry + '/torch:builder-{torch_version}-py{python_version}-cuda{cuda_version}'

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.11', '3.12']

DEFAULT_PYTHON_VERSION = 3.12

# List of available torch images
# Keep sorted by most recent cuda version.
AVAILABLE_TORCH_IMAGES = [
    '2.4.0-py3.11-cuda124',
    '2.4.1-py3.11-cuda124',
    '2.5.1-py3.11-cuda124',
    '2.4.0-py3.12-cuda124',
    '2.4.1-py3.12-cuda124',
    '2.5.1-py3.12-cuda124',
    # '2.4.0-py3.13-cuda124',
    # '2.4.1-py3.13-cuda124',
    # '2.5.1-py3.13-cuda124',
]
CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
]
