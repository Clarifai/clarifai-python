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
"""Triton model repository generation commandline interface."""

import argparse

from ..constants import MODEL_TYPES
from ..model_config.triton_config import TritonModelConfig
from ..pb_model_repository import TritonModelRepository


def model_upload_init():
  """
  Clarifai triton model upload commandline tool.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  # TritonModelConfig args
  parser.add_argument("--model_name", type=str, required=True, help="Inference Model Name")
  parser.add_argument(
      "--model_version",
      type=str,
      default="1",
      required=False,
      help="Triton inference model version name. 1 stands for version 1. \
    Leave as default value (Recommended).")
  parser.add_argument(
      "--model_type",
      type=str,
      choices=MODEL_TYPES,
      required=True,
      help=f"Clarifai supported model types.\n Model-types-map: {MODEL_TYPES}",
  )
  parser.add_argument(
      "--repo_dir",
      type=str,
      default=".",  #curdir
      required=True,
      help="Directory to create triton repository.")

  args = parser.parse_args()

  model_config = TritonModelConfig(
      model_name=args.model_name, model_version="1", model_type=args.model_type)

  triton_repo = TritonModelRepository(model_config)
  triton_repo.build_repository(args.repo_dir)


if __name__ == "__main__":
  model_upload_init()
