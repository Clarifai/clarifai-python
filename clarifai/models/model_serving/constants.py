import os

MAX_HW_DIM = 1024
IMAGE_TENSOR_NAME = "image"
TEXT_TENSOR_NAME = "text"

BUILT_MODEL_EXT = ".clarifai"

default_home = os.path.join(os.path.expanduser("~"), ".cache")
CLARIFAI_HOME = os.path.expanduser(
    os.getenv(
        "CLARIFAI_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "clarifai"),
    ))
os.makedirs(CLARIFAI_HOME, exist_ok=True)
CLARIFAI_PAT_PATH = os.path.join(CLARIFAI_HOME, "pat")

CLARIFAI_EXAMPLES_REPO = "https://github.com/Clarifai/examples.git"
repo_name = CLARIFAI_EXAMPLES_REPO.split("/")[-1].replace(".git", "")
CLARIFAI_EXAMPLES_REPO_PATH = os.path.join(CLARIFAI_HOME, repo_name)
MODEL_UPLOAD_EXAMPLE_FOLDER = "model_upload"
