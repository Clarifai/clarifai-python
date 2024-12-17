PYTHON_BASE_IMAGE = 'public.ecr.aws/clarifai-models/python-base:{python_version}'
TORCH_BASE_IMAGE = 'public.ecr.aws/clarifai-models/torch:{torch_version}-py{python_version}-cuda{cuda_version}'

# List of available python base images
AVAILABLE_PYTHON_IMAGES = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']

DEFAULT_PYTHON_VERSION = 3.11

# List of available torch images
AVAILABLE_TORCH_IMAGES = [
    '1.13.1-py3.8-cuda117',
    '1.13.1-py3.9-cuda117',
    '1.13.1-py3.10-cuda117',
    '2.1.2-py3.8-cuda121',
    '2.1.2-py3.9-cuda121',
    '2.1.2-py3.10-cuda121',
    '2.1.2-py3.11-cuda121',
    '2.2.2-py3.8-cuda121',
    '2.2.2-py3.9-cuda121',
    '2.2.2-py3.10-cuda121',
    '2.2.2-py3.11-cuda121',
    '2.2.2-py3.12-cuda121',
    '2.3.1-py3.8-cuda121',
    '2.3.1-py3.9-cuda121',
    '2.3.1-py3.10-cuda121',
    '2.3.1-py3.11-cuda121',
    '2.3.1-py3.12-cuda121',
    '2.4.1-py3.8-cuda124',
    '2.4.1-py3.9-cuda124',
    '2.4.1-py3.10-cuda124',
    '2.4.1-py3.11-cuda124',
    '2.4.1-py3.12-cuda124',
    '2.5.1-py3.9-cuda124',
    '2.5.1-py3.10-cuda124',
    '2.5.1-py3.11-cuda124',
    '2.5.1-py3.12-cuda124',
]
CONCEPTS_REQUIRED_MODEL_TYPE = [
    'visual-classifier', 'visual-detector', 'visual-segmenter', 'text-classifier'
]
