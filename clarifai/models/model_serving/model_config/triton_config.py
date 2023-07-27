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
"""Triton Model Config classes."""

from dataclasses import dataclass, field
from typing import List


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
  data_type: str = None
  dims: List = field(default_factory=list)


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
  data_type: str = None
  dims: List = field(default_factory=list)
  labels: bool = False

  def __post_init__(self):
    if self.labels:
      self.label_filename = "labels.txt"
    else:
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
      self.kind: str = "KIND_GPU"
    else:
      self.kind: str = "KIND_CPU"


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
  model_name: str
  model_version: str
  model_type: str
  input: List[InputConfig] = field(default_factory=list)
  output: List[OutputConfig] = field(default_factory=list)
  instance_group: Device = Device()
  dynamic_batching: DynamicBatching = DynamicBatching()
  max_batch_size: int = 1
  backend: str = "python"

  def __post_init__(self):
    """
    Set supported input dims and data_types for
    a given model_type.
    """
    image_input = InputConfig(name="image", data_type="TYPE_UINT8", dims=[-1, -1, 3])
    text_input = InputConfig(name="text", data_type="TYPE_STRING", dims=[1])

    if self.model_type == "visual-detector":
      self.input.append(image_input)
      pred_bboxes = OutputConfig(name="predicted_bboxes", data_type="TYPE_FP32", dims=[-1, 4])
      pred_labels = OutputConfig(
          name="predicted_labels", data_type="TYPE_INT32", dims=[-1, 1], labels=True)
      del pred_labels.labels
      pred_scores = OutputConfig(name="predicted_scores", data_type="TYPE_FP32", dims=[-1, 1])
      self.output.extend([pred_bboxes, pred_labels, pred_scores])

    elif self.model_type == "visual-classifier":
      self.input.append(image_input)
      pred_labels = OutputConfig(
          name="softmax_predictions", data_type="TYPE_FP32", dims=[-1], labels=True)
      del pred_labels.labels
      self.output.append(pred_labels)

    elif self.model_type == "text-classifier":
      self.input.append(text_input)
      pred_labels = OutputConfig(
          name="softmax_predictions", data_type="TYPE_FP32", dims=[-1], labels=True)
      #'Len of out list expected to be the number of concepts returned by the model,
      # with each value being the confidence for the respective model output.
      del pred_labels.labels
      self.output.append(pred_labels)

    elif self.model_type == "text-to-text":
      self.input.append(text_input)
      pred_text = OutputConfig(name="text", data_type="TYPE_STRING", dims=[1], labels=False)
      self.output.append(pred_text)

    elif self.model_type == "text-embedder":
      self.input.append(text_input)
      embedding_vector = OutputConfig(
          name="embeddings", data_type="TYPE_FP32", dims=[-1], labels=False)
      self.output.append(embedding_vector)

    elif self.model_type == "text-to-image":
      self.input.append(text_input)
      gen_image = OutputConfig(
          name="image", data_type="TYPE_UINT8", dims=[-1, -1, 3], labels=False)
      self.output.append(gen_image)

    elif self.model_type == "visual-embedder":
      self.input.append(image_input)
      embedding_vector = OutputConfig(
          name="embeddings", data_type="TYPE_FP32", dims=[-1], labels=False)
      self.output.append(embedding_vector)

    elif self.model_type == "visual-segmenter":
      self.input.append(image_input)
      pred_masks = OutputConfig(
          name="predicted_mask", data_type="TYPE_INT64", dims=[-1, -1], labels=True)
      del pred_masks.labels
      self.output.append(pred_masks)
