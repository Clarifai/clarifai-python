import os

registry = os.environ.get('CLARIFAI_BASE_IMAGE_REGISTRY', 'public.ecr.aws/clarifai-models')

GIT_SHA = "b8ae56bf3b7c95e686ca002b07ca83d259c716eb"

AMD_GIT_SHA = "b8ae56bf3b7c95e686ca002b07ca83d259c716eb"

PYTHON_BASE_IMAGE = registry + '/python-base:{python_version}-' + GIT_SHA
TORCH_BASE_IMAGE = registry + '/torch:{torch_version}-py{python_version}-{gpu_version}-' + GIT_SHA

AMD_PYTHON_BASE_IMAGE = registry + '/amd-python-base:{python_version}-' + AMD_GIT_SHA
AMD_TORCH_BASE_IMAGE = (
    registry + '/torch:{torch_version}-py{python_version}-{gpu_version}-' + AMD_GIT_SHA
)
AMD_VLLM_BASE_IMAGE = registry + '/amd-vllm-base:{python_version}-' + AMD_GIT_SHA

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.11', '3.12']

DEFAULT_PYTHON_VERSION = 3.12

DEFAULT_AMD_TORCH_VERSION = '2.7.0'

DEFAULT_AMD_GPU_VERSION = 'rocm6.3'

# By default we download at runtime.
DEFAULT_DOWNLOAD_CHECKPOINT_WHEN = "runtime"

# Folder for downloading checkpoints at runtime.
DEFAULT_RUNTIME_DOWNLOAD_PATH = os.path.join(os.sep, "tmp", ".cache")

# List of available torch images
# Keep sorted by most recent cuda version.
AVAILABLE_TORCH_IMAGES = [
    '2.4.1-py3.11-cu124',
    '2.5.1-py3.11-cu124',
    '2.4.1-py3.12-cu124',
    '2.5.1-py3.12-cu124',
    '2.6.0-py3.12-cu126',
    '2.7.0-py3.12-cu128',
    '2.7.0-py3.12-rocm6.3',
]

CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier',
    'visual-detector',
    'visual-segmenter',
    'text-classifier',
]
