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
class ClarifaiFieldsMap:
  """
  Triton Model Config base.
  Params:
  -------
  model_type

  Returns:
  --------
  ClarifaiFieldsMap
  """
  model_type: str
  input_fields_map: List = field(default_factory=list)
  output_fields_map: List = field(default_factory=list)

  def __post_init__(self):
    """
    Set mapping of clarifai in/output vs triton in/output
    """
    text_input_fields = {"text": "text"}
    image_input_fields = {"image": "image"}

    embedding_output_fields = {"embeddings": "embeddings"}

    if self.model_type == "visual-detector":
      self.input_fields_map = image_input_fields
      self.output_fields_map = {
          "regions[...].region_info.bounding_box": "predicted_bboxes",
          "regions[...].data.concepts[...].id": "predicted_labels",
          "regions[...].data.concepts[...].value": "predicted_scores"
      }
    elif self.model_type == "visual-classifier":
      self.input_fields_map = image_input_fields
      self.output_fields_map = {"concepts": "softmax_predictions"}
    elif self.model_type == "text-classifier":
      self.input_fields_map = text_input_fields
      self.output_fields_map = {"concepts": "softmax_predictions"}
    elif self.model_type == "text-embedder":
      self.input_fields_map = text_input_fields
      self.output_fields_map = embedding_output_fields
    elif self.model_type == "text-to-text":
      self.input_fields_map = text_input_fields
      # input and output fields are the same for text-to-text
      self.output_fields_map = text_input_fields
    elif self.model_type == "text-to-image":
      self.input_fields_map = text_input_fields
      # image output fields match image_input fields
      self.output_fields_map = image_input_fields
    elif self.model_type == "visual-embedder":
      self.input_fields_map = image_input_fields
      self.output_fields_map = embedding_output_fields
    elif self.model_type == "visual-segmenter":
      self.input_fields_map = image_input_fields
      self.output_fields_map = {
          "regions[...].region_info.mask,regions[...].data.concepts": "predicted_mask"
      }
