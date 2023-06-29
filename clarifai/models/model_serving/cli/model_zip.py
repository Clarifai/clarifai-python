# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton model zip commandline interface."""

import argparse
import zipfile
from pathlib import Path
from typing import Union


def zip_dir(triton_repository_dir: Union[Path, str], zip_filename: Union[Path, str]):
  """
  Generate triton model repository zip file for upload.
  Args:
  -----
  triton_repository_dir: Directory of triton model respository to be zipped
  zip_filename: Triton model repository zip filename

  Returns:
  --------
  None
  """
  # Convert to Path object
  dir = Path(triton_repository_dir)

  with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for entry in dir.rglob("*"):
      zip_file.write(entry, entry.relative_to(dir))


def main():
  """Triton model zip cli."""
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument(
      "--triton_model_repository",
      type=str,
      required=True,
      help="Path to the triton model repository to zip.")
  parser.add_argument(
      "--zipfile_name",
      type=str,
      required=True,
      help="Name of the zipfile to be created. \
      <model_name>_<model_type> is the recommended naming convention.e.g. yolov5_visual-detector.zip"
  )
  args = parser.parse_args()
  zip_dir(args.triton_model_repository, args.zipfile_name)


if __name__ == "__main__":
  main()
