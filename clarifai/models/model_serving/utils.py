import os

from clarifai.models.model_serving.constants import CLARIFAI_PAT_PATH
from clarifai.utils.constants import CLARIFAI_PAT_ENV_VAR


def _persist_pat(pat: str):
  """ Write down pat to CLARIFAI_PAT_PATH """
  with open(CLARIFAI_PAT_PATH, "w") as f:
    f.write(pat)


def _read_pat():
  if not os.path.exists(CLARIFAI_PAT_PATH) and not os.environ.get(CLARIFAI_PAT_ENV_VAR, ""):
    return None
  if os.path.exists(CLARIFAI_PAT_PATH):
    with open(CLARIFAI_PAT_PATH, "r") as f:
      return f.read().replace("\n", "").replace("\r", "").strip()
  elif os.environ.get(CLARIFAI_PAT_ENV_VAR):
    return os.environ.get(CLARIFAI_PAT_ENV_VAR)
  else:
    raise ValueError(
        f"PAT not found, please run `clarifai login` to persist your PAT or set it as an environment variable under the name '{CLARIFAI_PAT_ENV_VAR}'"
    )


def login(pat=None):
  """ if pat provided, set pat to CLARIFAI_PAT otherwise read pat from file"""
  pat = pat or _read_pat()
  assert pat, Exception("PAT is not found, please run `clarifai login` to persist your PAT")
  os.environ["CLARIFAI_PAT"] = pat
