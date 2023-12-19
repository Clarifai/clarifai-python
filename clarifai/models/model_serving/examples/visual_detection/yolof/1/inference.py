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

import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

from clarifai.models.model_serving.model_config import ModelTypes, get_model_config
from clarifai.models.model_serving.models.output import VisualDetectorOutput

# Initialize the DetInferencer
register_all_modules()

config = get_model_config(ModelTypes.visual_detector)


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.checkpoint = os.path.join(self.base_path,
                                   "config/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth")
    self.config_path = os.path.join(self.base_path, "config/yolof_r50_c5_8x8_1x_coco.py")
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = init_detector(self.config_path, self.checkpoint, device=self.device)

  @config.inference.wrap_func
  def get_predictions(self, input_data: list, **kwargs) -> list:
    """
    Main model inference method.

    Args:
    -----
      input_data: A list of input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

      **kwargs: your inference parameters.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.models.output types` or `config.inference.return_type(your_output)`. Refer to the README/docs
    """
    max_bbox_count = 500  # max allowed detected bounding boxes per image
    outputs = []

    if isinstance(input_data, np.ndarray) and len(input_data.shape) == 4:
      input_data = list(input_data)

    predictions = inference_detector(self.model, input_data)
    for inp_data, preds in zip(input_data, predictions):

      labels = preds.pred_instances.labels.cpu().numpy()
      bboxes = preds.pred_instances.bboxes.cpu().numpy()
      scores = preds.pred_instances.scores.cpu().numpy()
      labels = [[each] for each in labels]
      scores = [[each] for each in scores]
      h, w, _ = inp_data.shape  # input image shape
      bboxes = [[x[1] / h, x[0] / w, x[3] / h, x[2] / w]
                for x in bboxes]  # normalize the bboxes to [0,1]
      bboxes = np.clip(bboxes, 0, 1.)
      if len(bboxes) != 0:
        bboxes = np.concatenate((bboxes, np.zeros((max_bbox_count - len(bboxes), 4))))
        scores = np.concatenate((scores, np.zeros((max_bbox_count - len(scores), 1))))
        labels = np.concatenate((labels, np.zeros(
            (max_bbox_count - len(labels), 1), dtype=np.int32)))
      else:
        bboxes = np.zeros((max_bbox_count, 4), dtype=np.float32)
        scores = np.zeros((max_bbox_count, 1), dtype=np.float32)
        labels = np.zeros((max_bbox_count, 1), dtype=np.int32)

      outputs.append(
          VisualDetectorOutput(
              predicted_bboxes=bboxes, predicted_labels=labels, predicted_scores=scores))

    return outputs
