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
import logging
import os
from pathlib import Path
from typing import Type

logging.getLogger("clarifai.models.model_serving.model_config.config").setLevel(logging.ERROR)

from .model_config import Serializer, TritonModelConfig  # noqa: E402
from .models import inference, pb_model, test  # noqa: E402


def confirm_action(filename):
  while True:
    user_input = input(f"Do you want to overwrite file `{filename}`? (y: yes/ n: no): ")
    if user_input in ["y", "n"]:
      return user_input == 'y'
    else:
      print(f"Expected input in [y, n], got {user_input}")


class TritonModelRepository:
  """
  Triton Python BE Model Repository Generator.
  """

  def __init__(self, model_config: Type[TritonModelConfig]):
    self.model_config = model_config
    self.config_proto = Serializer(model_config)

  def _module_to_file(self, module, file_path: str, func: callable = None):
    """
    Write Python Module to file.

    Args:
    -----
      module: Python module to write to file
      file_path: Path of file to write module code into.
      func: A function to process code of module. It contains only 1 argument, text of module. If it is None, then only save text to `file_path`
    Returns:
    --------
      None
    """
    source_code = inspect.getsource(module)
    with open(file_path, "w") as fp:
      # change model type
      if func:
        source_code = func(source_code)
      # write it to file
      fp.write(source_code)

  def build_repository(self, repository_dir: Path = os.curdir):
    """
    Generate Triton Model Repository.

    Args:
    -----
    repository_dir: Directory to create triton model repository

    Returns:
    --------
    None
    """
    model_version = self.model_config.model_version
    model_version_path = os.path.join(repository_dir, model_version)

    os.makedirs(repository_dir, exist_ok=True)

    os.path.join(repository_dir, "config.pbtxt")
    self.config_proto.to_file(repository_dir)

    for out_field in self.model_config.output:
      #predicted int labels must have corresponding names in file
      if hasattr(out_field, "label_filename"):
        labels_txt_path = os.path.join(repository_dir, "labels.txt")
        with open(labels_txt_path, "w"):
          pass
      else:
        continue
    # gen requirements
    requirements_txt_path = os.path.join(repository_dir, "requirements.txt")
    with open(requirements_txt_path, "w") as f:
      f.write("clarifai>9.10.4\ntritonclient[all]")  # for model upload utils

    os.makedirs(model_version_path, exist_ok=True)
    _init_py_path = os.path.join(model_version_path, "__init__.py")
    with open(_init_py_path, "w"):
      pass

    # generate model.py
    model_py_path = os.path.join(model_version_path, "model.py")
    self._module_to_file(pb_model, model_py_path, func=None)

    # generate inference.py
    def insert_model_type_func(x):
      return x.replace("MODEL_TYPE_PLACEHOLDER", self.model_config.model_type)

    inference_py_path = os.path.join(model_version_path, "inference.py")
    self._module_to_file(inference, inference_py_path, insert_model_type_func)

    # generate test.py
    custom_test_path = os.path.join(model_version_path, "test.py")
    self._module_to_file(test, custom_test_path, insert_model_type_func)
