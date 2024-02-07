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
  os.environ["CLARIFAI_PAT"] = pat or _read_pat()
