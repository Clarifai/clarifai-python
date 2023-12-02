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

import logging
from dataclasses import asdict, dataclass, field
from typing import List

import yaml

from ..models.model_types import *  # noqa # pylint: disable=unused-import
from ..models.output import *  # noqa # pylint: disable=unused-import

logger = logging.getLogger(__name__)

__all__ = ["get_model_config", "MODEL_TYPES", "TritonModelConfig", "ModelTypes"]

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
  model_type: str
  model_name: str
  model_version: str
  image_shape: List  #(H, W)
  input: List[InputConfig] = field(default_factory=list)
  output: List[OutputConfig] = field(default_factory=list)
  instance_group: Device = field(default_factory=Device)
  dynamic_batching: DynamicBatching = field(default_factory=DynamicBatching)
  max_batch_size: int = 1
  backend: str = "python"

  def __post_init__(self):
    if "image" in [each.name for each in self.input]:
      image_dims = self.image_shape
      image_dims.append(3)  # add channel dim
      self.input[0].dims = image_dims


### General Model Config classes & functions ###


# Clarifai model types
@dataclass
class ModelTypes:
  visual_detector: str = "visual-detector"
  visual_classifier: str = "visual-classifier"
  text_classifier: str = "text-classifier"
  text_to_text: str = "text-to-text"
  text_embedder: str = "text-embedder"
  text_to_image: str = "text-to-image"
  visual_embedder: str = "visual-embedder"
  visual_segmenter: str = "visual-segmenter"
  multimodal_embedder: str = "multimodal-embedder"

  def __post_init__(self):
    self.all = list(asdict(self).values())


@dataclass
class InferenceConfig:
  wrap_func: callable
  return_type: dataclass


@dataclass
class FieldMapsConfig:
  input_fields_map: dict
  output_fields_map: dict


@dataclass
class DefaultTritonConfig:
  input: List[InputConfig] = field(default_factory=list)
  output: List[OutputConfig] = field(default_factory=list)


@dataclass
class ModelConfigClass:
  type: str = field(init=False)
  triton: DefaultTritonConfig
  inference: InferenceConfig
  field_maps: FieldMapsConfig

  def make_triton_model_config(
      self,
      model_name: str,
      model_version: str,
      image_shape: List = None,
      instance_group: Device = Device(),
      dynamic_batching: DynamicBatching = DynamicBatching(),
      max_batch_size: int = 1,
      backend: str = "python",
  ) -> TritonModelConfig:

    return TritonModelConfig(
        model_type=self.type,
        model_name=model_name,
        model_version=model_version,
        image_shape=image_shape,
        instance_group=instance_group,
        dynamic_batching=dynamic_batching,
        max_batch_size=max_batch_size,
        backend=backend,
        input=self.triton.input,
        output=self.triton.output)


def read_config(cfg: str):
  with open(cfg, encoding="utf-8") as f:
    config = yaml.safe_load(f)  # model dict

  # parse default triton
  input_triton_configs = config["triton"]["input"]
  output_triton_configs = config["triton"]["output"]
  triton = DefaultTritonConfig(
      input=[
          InputConfig(
              name=input["name"],
              data_type=eval(f"DType.{input['data_type']}"),
              dims=input["dims"],
              optional=input.get("optional", False),
          ) for input in input_triton_configs
      ],
      output=[
          OutputConfig(
              name=output["name"],
              data_type=eval(f"DType.{output['data_type']}"),
              dims=output["dims"],
              labels=output["labels"],
          ) for output in output_triton_configs
      ])

  # parse inference config
  inference = InferenceConfig(
      wrap_func=eval(config["inference"]["wrap_func"]),
      return_type=eval(config["inference"]["return_type"]),
  )

  # parse field maps for deployment
  field_maps = FieldMapsConfig(**config["field_maps"])

  return ModelConfigClass(triton=triton, inference=inference, field_maps=field_maps)


def get_model_config(model_type: str) -> ModelConfigClass:
  """
  Get model config by model type

  Args:

    model_type (str): One of field value of ModelTypes

  Return:
    ModelConfigClass

  ### Example:
  >>> from clarifai.models.model_serving.models.output import ClassifierOutput
  >>> from clarifai.models.model_serving.model_config import get_model_config, ModelTypes
  >>> cfg = get_model_config(ModelTypes.text_classifier)
  >>> custom_triton_config = cfg.make_triton_model_config(**kwargs)
  >>> cfg.inference.return_type is ClassifierOutput # True


  """
  if model_type == "MODEL_TYPE_PLACEHOLDER":
    logger.warning(
        "Warning: A placeholder value has been detected for obtaining the model configuration. This will result in empty `ModelConfigClass` object."
    )
    return ModelConfigClass(
        triton=None,
        inference=InferenceConfig(wrap_func=lambda x: x, return_type=None),
        field_maps=None)

  import os
  assert model_type in MODEL_TYPES, f"`model_type` must be in {MODEL_TYPES}"
  cfg = read_config(
      os.path.join(os.path.dirname(__file__), "model_types_config", f"{model_type}.yaml"))
  cfg.type = model_type
  return cfg


_model_types = ModelTypes()
MODEL_TYPES = _model_types.all
del _model_types
