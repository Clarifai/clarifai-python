# This file contains boilerplate code to allow users write their model
# inference code that will then interact with the Triton Inference Server
# Python backend to serve end user requests.
# The module name, module path and the setup() & predict() function names MUST be maintained as is
# but other functions may be added within this module as deemed fit provided
# they are invoked within the main predict() function if they play a role any
# step of model inference
"""User model inference script."""

import os
from typing import Callable

import numpy as np
import torch
from clarifai.models.model_serving.models.model_types import visual_detector
from clarifai.models.model_serving.models.output import VisualDetectorOutput

BASE_PATH = os.path.dirname(__file__)


def setup():
  """
  Load inference model.
  The model checkpoint(s) should be saved in the same directory and directory
  level as this inference.py module.

  Returns:
  --------
    Inference Model Callable
  """
  checkpoint_path = os.path.join(BASE_PATH, "model.pt")  #yolov5x
  model = torch.hub.load(
      os.path.join(BASE_PATH, 'yolov5'),
      'custom',
      autoshape=True,
      path=checkpoint_path,
      source='local')
  return model


@visual_detector
def predict(input_data, model: Callable):
  """
  Main model inference function.

  Args:
  -----
    input_data: A single input data item to predict on.
      Input data can be an image or text, etc depending on the model type.
    model: Inference model callable as returned by setup() above.

  Returns:
  --------
    One of the clarifai.model_serving.models.output types. Refer to the README/docs
  """
  preds = model(input_data)
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
    labels = np.concatenate((labels, np.zeros((max_bbox_count - len(labels), 1), dtype=np.int32)))
  else:
    bboxes = np.zeros((max_bbox_count, 4), dtype=np.float32)
    scores = np.zeros((max_bbox_count, 1), dtype=np.float32)
    labels = np.zeros((max_bbox_count, 1), dtype=np.int32)

  return VisualDetectorOutput(
      predicted_bboxes=bboxes, predicted_labels=labels, predicted_scores=scores)
