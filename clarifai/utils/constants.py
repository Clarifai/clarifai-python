import os

USER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache")
CLARIFAI_HOME = os.path.expanduser(
    os.getenv(
        "CLARIFAI_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", USER_CACHE_DIR), "clarifai"),
    ))
os.makedirs(CLARIFAI_HOME, exist_ok=True)
CLARIFAI_PAT_PATH = os.path.join(CLARIFAI_HOME, "pat")
CLARIFAI_PAT_ENV_VAR = "CLARIFAI_PAT"
CLARIFAI_SESSION_TOKEN_ENV_VAR = "CLARIFAI_SESSION_TOKEN"
