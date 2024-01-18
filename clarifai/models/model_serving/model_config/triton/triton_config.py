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
""" Model Config classes."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Union


### Triton Model Config classes.###
@dataclass
class DType:
  """
  Triton Model Config data types.
  """
  # https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
  TYPE_UINT8: int = 2
  TYPE_INT8: int = 6
  TYPE_INT16: int = 7
  TYPE_INT32: int = 8
  TYPE_INT64: int = 9
  TYPE_FP16: int = 10
  TYPE_FP32: int = 11
  TYPE_STRING: int = 13
  KIND_GPU: int = 1
  KIND_CPU: int = 2


@dataclass
class InputConfig:
  """
  Triton Input definition.
  Params:
  -------
  name: input name
  data_type: input data type
  dims: Pre-defined input data shape(s).

  Returns:
  --------
  InputConfig
  """
  name: str
  data_type: int
  dims: List = field(default_factory=list)
  optional: bool = False


@dataclass
class OutputConfig:
  """
  Triton Output definition.
  Params:
  -------
  name: output name
  data_type: output data type
  dims: Pre-defined output data shape(s).
  labels (bool): If labels file is required for inference.

  Returns:
  --------
  OutputConfig
  """
  name: str
  data_type: int
  dims: List = field(default_factory=list)
  labels: bool = False

  def __post_init__(self):
    if self.labels:
      self.label_filename = "labels.txt"
    del self.labels


@dataclass
class Device:
  """
  Triton instance_group.
  Define the type of inference device and number of devices to use.
  Params:
  -------
  count: number of devices
  use_gpu: whether to use cpu or gpu.

  Returns:
  --------
  Device object
  """
  count: int = 1
  use_gpu: bool = True

  def __post_init__(self):
    if self.use_gpu:
      self.kind: str = DType.KIND_GPU
    else:
      self.kind: str = DType.KIND_CPU


@dataclass
class DynamicBatching:
  """
  Triton dynamic_batching config.
  Params:
  -------
  preferred_batch_size: batch size
  max_queue_delay_microseconds: max queue delay for a request batch

  Returns:
  --------
  DynamicBatching object
  """
  #preferred_batch_size: List[int] = [1] # recommended not to set
  max_queue_delay_microseconds: int = 500


@dataclass
class TritonModelConfig:
  """
  Triton Model Config base.
  Params:
  -------
  name: triton inference model name
  input: a list of an InputConfig field
  output: a list of OutputConfig fields/dicts
  instance_group: Device. see Device
  dynamic_batching: Triton dynamic batching settings.
  max_batch_size: max request batch size
  backend: Triton Python Backend. Constant

  Returns:
  --------
  TritonModelConfig
  """
  #model_type: str
  model_name: str = ""
  model_version: str = "1"
  image_shape: tuple[Union[int, float], Union[int, float]] = field(
      default_factory=lambda: [-1, -1])  #(H, W)
  input: List[InputConfig] = field(default_factory=list)
  output: List[OutputConfig] = field(default_factory=list)
  instance_group: Device = field(default_factory=Device)
  dynamic_batching: DynamicBatching = field(default_factory=DynamicBatching)
  max_batch_size: int = 1
  backend: str = "python"

  def __post_init__(self):
    if "image" in [each.name for each in self.input]:
      image_dims = deepcopy(self.image_shape)
      image_dims.append(3)  # add channel dim
      self.input[0].dims = image_dims
