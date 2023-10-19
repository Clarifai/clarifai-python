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
Parse & Serialize TritonModelConfig objects into proto format.
"""

import os
from pathlib import Path
from typing import Type

from google.protobuf.text_format import MessageToString
from tritonclient.grpc import model_config_pb2

from .config import TritonModelConfig


class Serializer:
  """
  Serialize TritonModelConfig type object.
  """

  def __init__(self, model_config: Type[TritonModelConfig]) -> None:
    self.model_config = model_config  #python dataclass config
    self.config_proto = model_config_pb2.ModelConfig()  #holds parsed python config

    self._set_all_fields()

  def _set_input(self) -> None:
    """
    Parse InputConfig object to proto.
    """
    for in_field in self.model_config.input:
      input_config = self.config_proto.input.add()
      for key, value in in_field.__dict__.items():
        try:
          setattr(input_config, key, value)
        except AttributeError:
          field = getattr(input_config, key)
          if isinstance(value, list):
            field.extend(value)
          else:
            field.extend([value])
    return

  def _set_output(self) -> None:
    """
    Parse OutputConfig object to proto.
    """
    # loop over output dataclass list
    for out_field in self.model_config.output:
      output_config = self.config_proto.output.add()
      for key, value in out_field.__dict__.items():
        try:
          setattr(output_config, key, value)
        except AttributeError:  #Proto Repeated Field assignment not allowed
          field = getattr(output_config, key)
          if isinstance(value, list):
            field.extend(value)
          else:
            field.extend([value])
    return

  def _set_instance_group(self) -> None:
    """
    Parse triton model instance group settings to proto.
    """
    instance = self.config_proto.instance_group.add()
    for field_name, value in self.model_config.instance_group.__dict__.items():
      try:
        setattr(instance, field_name, value)
      except AttributeError:
        continue
    return

  def _set_batch_info(self) -> model_config_pb2.ModelDynamicBatching:
    """
    Parse triton model dynamic batching settings to proto.
    """
    dbatch_msg = model_config_pb2.ModelDynamicBatching()
    for key, value in self.model_config.dynamic_batching.__dict__.items():
      try:
        setattr(dbatch_msg, key, value)
      except AttributeError:  #Proto Repeated Field assignment not allowed
        field = getattr(dbatch_msg, key)
        if isinstance(value, list):
          field.extend(value)
        else:
          field.extend([value])

    return dbatch_msg

  def _set_all_fields(self) -> None:
    """
    Set all config fields.
    """
    self.config_proto.name = self.model_config.model_name
    self.config_proto.backend = self.model_config.backend
    self.config_proto.max_batch_size = self.model_config.max_batch_size
    self._set_input()
    self._set_output()
    self._set_instance_group()
    dynamic_batch_msg = self._set_batch_info()
    self.config_proto.dynamic_batching.CopyFrom(dynamic_batch_msg)

  @property
  def get_config(self) -> model_config_pb2.ModelConfig:
    """
    Return model config proto.
    """
    return self.config_proto

  def to_file(self, save_dir: Path) -> None:
    """
    Serialize all triton config parameters and save output
    to file.
    Args:
    -----
    save_dir: Directory where to save resultant config.pbtxt file.
          Defaults to the current working dir.
    """
    msg_string = MessageToString(self.config_proto)

    with open(os.path.join(save_dir, "config.pbtxt"), "w") as cfile:
      cfile.write(msg_string)
