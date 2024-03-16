import os

from .constants import CLARIFAI_PAT_PATH


def _persist_pat(pat: str):
  """ Write down pat to CLARIFAI_PAT_PATH """
  with open(CLARIFAI_PAT_PATH, "w") as f:
    f.write(pat)


def _read_pat():
  if not os.path.exists(CLARIFAI_PAT_PATH):
    return None
  with open(CLARIFAI_PAT_PATH, "r") as f:
    return f.read().replace("\n", "").replace("\r", "").strip()


def login(pat=None):
  """ if pat provided, set pat to CLARIFAI_PAT otherwise read pat from file"""
  pat = pat or _read_pat()
  assert pat, Exception("PAT is not found, please run `clarifai login` to persist your PAT")
  os.environ["CLARIFAI_PAT"] = pat
