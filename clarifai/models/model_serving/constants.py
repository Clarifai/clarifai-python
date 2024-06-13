import os

from clarifai.utils.constants import CLARIFAI_HOME

MAX_HW_DIM = 1024
IMAGE_TENSOR_NAME = "image"
TEXT_TENSOR_NAME = "text"

BUILT_MODEL_EXT = ".clarifai"

CLARIFAI_EXAMPLES_REPO = "https://github.com/Clarifai/examples.git"
repo_name = CLARIFAI_EXAMPLES_REPO.split("/")[-1].replace(".git", "")
CLARIFAI_EXAMPLES_REPO_PATH = os.path.join(CLARIFAI_HOME, repo_name)
MODEL_UPLOAD_EXAMPLE_FOLDER = "model_upload"
