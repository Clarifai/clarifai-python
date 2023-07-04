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
Output Predictions format for different model types.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VisualDetectorOutput:
  predicted_bboxes: np.ndarray
  predicted_labels: np.ndarray
  predicted_scores: np.ndarray

  def __post_init__(self):
    """
    Validate input upon initialization.
    """
    assert self.predicted_bboxes.ndim == self.predicted_labels.ndim == \
      self.predicted_scores.ndim==2, f"All predictions must be 2-dimensional, \
        Got bbox-dims: {self.predicted_bboxes.ndim}, label-dims: {self.predicted_labels.ndim}, \
          scores-dims: {self.predicted_scores.ndim} instead."
    assert self.predicted_bboxes.shape[0] == self.predicted_labels.shape[0] == \
      self.predicted_scores.shape[0], f"The Number of predicted bounding boxes, \
        predicted labels and predicted scores MUST match. Got {len(self.predicted_bboxes)}, \
          {self.predicted_labels.shape[0]}, {self.predicted_scores.shape[0]} instead."

    if len(self.predicted_labels) > 0:
      assert self.predicted_bboxes.shape[1] == 4, f"Box coordinates must have a length of 4."
      assert np.all(np.logical_and(0 <= self.predicted_bboxes, self.predicted_bboxes <= 1)), \
       "Bounding box coordinates must be between 0 and 1"


@dataclass
class ClassifierOutput:
  """
  Takes model softmax predictions
  """
  predicted_scores: np.ndarray

  # the index of each predicted score as returned by the model must correspond
  # to the predicted label index in the labels.txt file

  def __post_init__(self):
    """
    Validate input upon initialization.
    """
    assert self.predicted_scores.ndim == 1, \
      f"All predictions must be 1-dimensional, Got scores-dims: {self.predicted_scores.ndim} instead."
