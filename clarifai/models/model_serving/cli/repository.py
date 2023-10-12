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

from ..constants import MAX_HW_DIM
from ..model_config import MODEL_TYPES, get_model_config
from ..pb_model_repository import TritonModelRepository


def dims_type(shape_string: str):
  """Read list string from cli and convert values to a list of integers."""
  shape_string = shape_string.replace("[", "").replace("]", "")
  shapes = list(map(int, shape_string.split(",")))
  return shapes


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
      "--image_shape",
      type=dims_type,
      default="[-1, -1]",
      required=False,
      help="(H, W) dims for models with an image input type. H and W each have a max value of 1024",
  )
  parser.add_argument(
      "--repo_dir",
      type=str,
      default=".",
      required=True,
      help="Directory to create triton repository.")

  args = parser.parse_args()

  if len(args.image_shape) != 2:
    raise ValueError(
        f"image_shape takes 2 values, Height and Width. Got {len(args.image_shape)} values instead."
    )

  if args.image_shape[0] > MAX_HW_DIM or args.image_shape[1] > MAX_HW_DIM:
    raise ValueError(
        f"H and W each have a maximum value of 1024. Got H: {args.image_shape[0]}, W: {args.image_shape[1]}"
    )

  model_config = get_model_config(args.model_type).make_triton_model_config(
      model_name=args.model_name,
      model_version="1",
      image_shape=args.image_shape,
  )

  triton_repo = TritonModelRepository(model_config)
  triton_repo.build_repository(args.repo_dir)


if __name__ == "__main__":
  model_upload_init()
