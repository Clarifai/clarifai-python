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
from pathlib import Path
from typing import Callable, Type

from .model_config import Serializer, TritonModelConfig
from .models import inference, pb_model, test


class TritonModelRepository:
  """
  Triton Python BE Model Repository Generator.
  """

  def __init__(self, model_config: Type[TritonModelConfig]):
    self.model_config = model_config
    self.config_proto = Serializer(model_config)

  def _module_to_file(self, module_name: Callable, filename: str, destination_dir: str) -> None:
    """
    Write Python Module to file.

    Args:
    -----
    module_name: Python module name to write to file
    filename: Name of the file to write to destination_dir
    destination_dir: Directory to save the generated triton model file.

    Returns:
    --------
    None
    """
    module_path: Path = os.path.join(destination_dir, filename)
    source_code = inspect.getsource(module_name)
    with open(module_path, "w") as pb_model:
      pb_model.write(source_code)

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
    model_repository = self.model_config.model_name
    model_version = self.model_config.model_version
    repository_path = os.path.join(repository_dir, model_repository)
    model_version_path = os.path.join(repository_path, model_version)

    if not os.path.isdir(repository_path):
      os.mkdir(repository_path)
      self.config_proto.to_file(repository_path)
      for out_field in self.model_config.output:
        #predicted int labels must have corresponding names in file
        if hasattr(out_field, "label_filename"):
          with open(os.path.join(repository_path, "labels.txt"), "w"):
            pass
        else:
          continue
      # gen requirements
      with open(os.path.join(repository_path, "requirements.txt"), "w") as f:
        f.write("clarifai>9.5.3\ntritonclient[all]")  # for model upload utils

    if not os.path.isdir(model_version_path):
      os.mkdir(model_version_path)
    if not os.path.exists(os.path.join(model_version_path, "__init__.py")):
      with open(os.path.join(model_version_path, "__init__.py"), "w"):
        pass
    # generate model.py & inference.py modules
    self._module_to_file(pb_model, filename="model.py", destination_dir=model_version_path)
    self._module_to_file(inference, filename="inference.py", destination_dir=model_version_path)
    # generate test.py
    custom_test_path = os.path.join(model_version_path, "test.py")
    test_source_code = inspect.getsource(test)
    with open(custom_test_path, "w") as fp:
      # change model type
      test_source_code = test_source_code.replace("clarifai-model-type",
                                                  self.model_config.model_type)
      # write it to file
      fp.write(test_source_code)
