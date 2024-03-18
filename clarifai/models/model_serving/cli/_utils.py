from __future__ import annotations  # isort: skip
import os
import shutil
import subprocess
from typing import Dict, Union

from ..constants import (CLARIFAI_EXAMPLES_REPO, CLARIFAI_EXAMPLES_REPO_PATH,
                         MODEL_UPLOAD_EXAMPLE_FOLDER)


def download_examples_repo(forced_download: bool = False):

  def _pull():
    subprocess.run(f"git clone {CLARIFAI_EXAMPLES_REPO} {CLARIFAI_EXAMPLES_REPO_PATH}", shell=True)

  if not os.path.isdir(CLARIFAI_EXAMPLES_REPO_PATH):
    print(f"Download examples to {CLARIFAI_EXAMPLES_REPO_PATH}")
    _pull()
  else:
    if forced_download:

      def _rm_dir_readonly(func, path, _):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

      print("Removing old examples...")
      shutil.rmtree(CLARIFAI_EXAMPLES_REPO_PATH, onerror=_rm_dir_readonly)
      _pull()


def list_model_upload_examples(
    forced_download: bool = False) -> Dict[str, tuple[str, Union[str, None]]]:
  download_examples_repo(forced_download)
  model_upload_folder = MODEL_UPLOAD_EXAMPLE_FOLDER
  model_upload_path = os.path.join(CLARIFAI_EXAMPLES_REPO_PATH, model_upload_folder)
  examples = {}
  for model_type_ex in os.listdir(model_upload_path):
    _folder = os.path.join(model_upload_path, model_type_ex)
    if os.path.isdir(_folder):
      _walk = list(os.walk(_folder))
      if len(_walk) > 0:
        _, model_names, _files = _walk[0]
        readme = [item for item in _files if "readme" in item.lower()]
        for name in model_names:
          examples.update({
              f"{model_type_ex}/{name}": [
                  os.path.join(_folder, name),
                  os.path.join(_folder, readme[0]) or None
              ]
          })

  return examples
