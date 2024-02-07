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
"""
Triton python backend inference model controller.
"""

import inspect
import os
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal, Union

import yaml
from tqdm import tqdm

from ..constants import BUILT_MODEL_EXT
from ..model_config import MODEL_TYPES, ModelConfigClass, base, get_model_config, load_user_config
from ..model_config.base import *  # noqa
from ..model_config.config import parse_config
from ..model_config.triton.serializer import Serializer


def __parse_type_to_class():
  _t = {}
  _classes = inspect.getmembers(base, inspect.isclass)
  for cls_name, cls_obj in _classes:
    if cls_obj.__base__ is base._BaseClarifaiModel:
      _t.update({cls_obj._config.clarifai_model.type: cls_name})
  return _t


_TYPE_TO_CLASS = __parse_type_to_class()


def _get_static_file_path(relative_path: str):
  curr_dir = os.path.dirname(__file__)
  return os.path.join(curr_dir, "static_files", relative_path)


def _read_static_file(relative_path: str):
  path = _get_static_file_path(relative_path)
  with open(path, "r") as f:
    return f.read()


def copy_folder(src_folder, dest_folder, exclude_items=None):

  if exclude_items is None:
    exclude_items = set()

  # Ensure the destination folder exists
  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

  loader = tqdm(os.listdir(src_folder))
  if exclude_items:
    print(f"NOTE: skipping {exclude_items}")

  for item in loader:
    loader.set_description(f"copying {item}...")
    src_item = os.path.join(src_folder, item)
    dest_item = os.path.join(dest_folder, item)

    # Skip items in the exclude list
    if item in exclude_items or item.endswith(BUILT_MODEL_EXT):
      continue

    # Copy files directly
    if os.path.isfile(src_item):
      shutil.copy2(src_item, dest_item)

    # Copy directories using copytree
    elif os.path.isdir(src_item):
      shutil.copytree(src_item, dest_item, symlinks=False, ignore=None, dirs_exist_ok=True)


def zip_dir(input: Union[Path, str], zip_filename: Union[Path, str]):
  """
  Zip folder without compressing
  """
  # Convert to Path object
  dir = Path(input)

  with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_STORED) as zip_file:
    for entry in dir.rglob("*"):
      zip_file.write(entry, entry.relative_to(dir))


class RepositoryBuilder:

  @staticmethod
  def init_repository(model_type: str, working_dir: str, backend=Literal['triton'], **kwargs):
    assert model_type in MODEL_TYPES
    model_type = model_type
    default_model_type_config: ModelConfigClass = get_model_config(model_type)

    os.makedirs(working_dir, exist_ok=True)

    def __write_to(filename, data):
      with open(os.path.join(working_dir, filename), "w") as f:
        f.write(data)

    # create inference.py
    _filename = "inference.py"
    inference_py = _read_static_file(_filename)
    inference_py = inference_py.replace("InferenceModel()",
                                        f"InferenceModel({_TYPE_TO_CLASS[model_type]})")
    inference_py = inference_py.replace("predict_docstring",
                                        eval(_TYPE_TO_CLASS[model_type]).predict.__doc__)
    # create config
    config = asdict(default_model_type_config)
    if backend == "triton":
      max_batch_size = kwargs.get("max_batch_size", None)
      image_shape = kwargs.get("image_shape", None)
      if max_batch_size:
        config['serving_backend']['triton']['max_batch_size'] = max_batch_size
      if image_shape:
        config['serving_backend']['triton']['image_shape'] = image_shape
      config = parse_config(config).dump_to_user_config()
      config_data = yaml.dump(config)
      sample_yaml = _read_static_file("sample_clarifai_config.yaml")
      config_data = sample_yaml + "\n\n" + config_data
      __write_to("clarifai_config.yaml", config_data)
      #
    # create inference.py after checking all configs
    __write_to(_filename, inference_py)
    # create test.py
    __write_to("test.py", _read_static_file("test.py"))
    # create requirements.txt
    __write_to("requirements.txt", _read_static_file("_requirements.txt"))

  @staticmethod
  def build(working_dir: str, output_dir: str = None, name: str = None, backend=Literal['triton']):
    if not output_dir:
      output_dir = working_dir
    else:
      os.makedirs(output_dir, exist_ok=True)

    temp_folder = os.path.join(working_dir, ".cache")
    os.makedirs(temp_folder, exist_ok=True)

    user_config_file = os.path.join(working_dir, "clarifai_config.yaml")
    assert os.path.exists(
        user_config_file
    ), f"FileNotFound: please make sure `clarifai_config.yaml` exists in {working_dir}"
    user_config = load_user_config(user_config_file)

    if backend == "triton":
      triton_1_ver = os.path.join(temp_folder, "1")
      os.makedirs(triton_1_ver, exist_ok=True)
      # check if labels exists
      for output_config in user_config.serving_backend.triton.output:
        if output_config.label_filename:
          user_labels = user_config.clarifai_model.labels
          assert user_labels, f"Model type `{user_config.clarifai_model.type}` requires labels, "\
          f"but can not found value of `clarifai_model.labels` in {user_config_file}. Please update this attribute to build the model"
          with open(os.path.join(temp_folder, "labels.txt"), "w") as f:
            if not isinstance(user_labels, Iterable):
              user_labels = [user_labels]
            f.write("\n".join([str(lb) for lb in user_labels]) + "\n")

      # copy model.py
      shutil.copy(_get_static_file_path("triton/model.py"), triton_1_ver)
      # copy requirements.txt
      shutil.copy(os.path.join(working_dir, "requirements.txt"), temp_folder)
      # copy all other files
      copy_folder(
          working_dir, triton_1_ver, exclude_items=["requirements.txt", ".cache", "__pycache__"])
      # generate config.pbtxt
      _config_pbtxt_serializer = Serializer(user_config.serving_backend.triton)
      _config_pbtxt_serializer.to_file(temp_folder)

    else:
      raise ValueError(f"backend must be ['triton'], got {backend}")

    clarifai_model_name = name or user_config.clarifai_model.clarifai_model_id or "model"
    clarifai_model_name += BUILT_MODEL_EXT
    clarifai_model_name = os.path.join(output_dir, clarifai_model_name)

    print(
        "Model building in progress; the duration may vary depending on the size of checkpoints/assets..."
    )
    zip_dir(temp_folder, clarifai_model_name)
    print(f"Finished. Your model is located at {clarifai_model_name}")

    return clarifai_model_name
