import os

registry = os.environ.get('CLARIFAI_BASE_IMAGE_REGISTRY', 'public.ecr.aws/clarifai-models')

GIT_SHA = "df565436eea93efb3e8d1eb558a0a46df29523ec"

PYTHON_BASE_IMAGE = registry + '/python-base:{python_version}-' + GIT_SHA
TORCH_BASE_IMAGE = registry + '/torch:{torch_version}-py{python_version}-cuda{cuda_version}-' + GIT_SHA

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.11', '3.12']

DEFAULT_PYTHON_VERSION = 3.12

# By default we download at runtime.
DEFAULT_DOWNLOAD_CHECKPOINT_WHEN = "runtime"

# Folder for downloading checkpoints at runtime.
DEFAULT_RUNTIME_DOWNLOAD_PATH = os.path.join(os.sep, "tmp", ".cache")

# List of available torch images
# Keep sorted by most recent cuda version.
AVAILABLE_TORCH_IMAGES = [
    '2.4.1-py3.11-cuda124',
    '2.5.1-py3.11-cuda124',
    '2.4.1-py3.12-cuda124',
    '2.5.1-py3.12-cuda124',
    # '2.4.1-py3.13-cuda124',
    # '2.5.1-py3.13-cuda124',
]
CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
]
