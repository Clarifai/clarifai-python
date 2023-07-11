# This file contains boilerplate code to allow users write their model
# inference code that will then interact with the Triton Inference Server
# Python backend to serve end user requests.
# The module name, module path, class name & get_predictions() method names MUST be maintained as is
# but other methods may be added within the class as deemed fit provided
# they are invoked within the main get_predictions() inference method
# if they play a role in any step of model inference
"""User model inference script."""

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from clarifai.models.model_serving.models.model_types import visual_detector
from clarifai.models.model_serving.models.output import VisualDetectorOutput


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.checkpoint_path = os.path.join(self.base_path, "model.pt")  #yolov5x
    self.model: Callable = torch.hub.load(
        os.path.join(self.base_path, 'yolov5'),
        'custom',
        autoshape=True,
        path=self.checkpoint_path,
        source='local')
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  @visual_detector
  def get_predictions(self, input_data) -> VisualDetectorOutput:
    """
    Main model inference method.

    Args:
    -----
      input_data: A single input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

    Returns:
    --------
      One of the clarifai.models.model_serving.models.output types. Refer to the README/docs
    """
    preds = self.model(input_data)
    max_bbox_count = 300  # max allowed detected bounding boxes per image
    preds = preds.xyxy[0].cpu().numpy()
    labels = [[pred[5]] for pred in preds]
    scores = [[pred[4]] for pred in preds]
    h, w, _ = input_data.shape  # input image shape
    bboxes = [[x[1] / h, x[0] / w, x[3] / h, x[2] / w]
              for x in preds]  # normalize the bboxes to [0,1]
    if len(bboxes) != 0:
      bboxes = np.concatenate((bboxes, np.zeros((max_bbox_count - len(bboxes), 4))))
      scores = np.concatenate((scores, np.zeros((max_bbox_count - len(scores), 1))))
      labels = np.concatenate((labels, np.zeros((max_bbox_count - len(labels), 1),
                                                dtype=np.int32)))
    else:
      bboxes = np.zeros((max_bbox_count, 4), dtype=np.float32)
      scores = np.zeros((max_bbox_count, 1), dtype=np.float32)
      labels = np.zeros((max_bbox_count, 1), dtype=np.int32)

    return VisualDetectorOutput(
        predicted_bboxes=bboxes, predicted_labels=labels, predicted_scores=scores)
