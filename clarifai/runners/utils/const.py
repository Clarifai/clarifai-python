import os

registry = os.environ.get('CLARIFAI_BASE_IMAGE_REGISTRY', 'public.ecr.aws/clarifai-models')

GIT_SHA = "de9e9a77b952b30c735d8734dd4308734dbbc5b4"

PYTHON_BASE_IMAGE = registry + '/python-base:{python_version}-' + GIT_SHA
TORCH_BASE_IMAGE = registry + '/torch:{torch_version}-py{python_version}-cuda{cuda_version}-' + GIT_SHA

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.11', '3.12']

DEFAULT_PYTHON_VERSION = 3.12

# List of available torch images
# Keep sorted by most recent cuda version.
AVAILABLE_TORCH_IMAGES = [
    '2.4.1-py3.11-cuda124',
    '2.5.1-py3.11-cuda124',
    '2.4.1-py3.12-cuda124',
    '2.5.1-py3.12-cuda124',
    # '2.4.0-py3.13-cuda124',
    # '2.4.1-py3.13-cuda124',
    # '2.5.1-py3.13-cuda124',
]
CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
]
