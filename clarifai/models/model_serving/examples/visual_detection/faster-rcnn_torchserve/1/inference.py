# This file contains boilerplate code to allow users write their model
# inference code that will then interact with the Triton Inference Server
# Python backend to serve end user requests.
# The module name, module path, class name & get_predictions() method names MUST be maintained as is
# but other methods may be added within the class as deemed fit provided
# they are invoked within the main get_predictions() inference method
# if they play a role in any step of model inference
"""User model inference script."""

import os

ROOT = os.path.dirname(__file__)
os.environ['TORCH_HOME'] = os.path.join(ROOT, "model_store")

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models, transforms  # noqa: E402

from clarifai.models.model_serving.model_config import ModelTypes, get_model_config  # noqa: E402
from clarifai.models.model_serving.models.output import VisualDetectorOutput  # noqa: E402

config = get_model_config(ModelTypes.visual_detector)


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    #self.checkpoint = os.path.join(ROOT, "model_store/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    self.model = self.model.to(self.device)
    self.model.eval()

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
    max_bbox_count = 300  # max allowed detected bounding boxes per image
    outputs = []

    if isinstance(input_data, np.ndarray) and len(input_data.shape) == 4:
      input_data = list(input_data)

    input_tensor = [self.transform(Image.fromarray(each)) for each in input_data]
    input_tensor = torch.stack(input_tensor).to(self.device)

    with torch.no_grad():
      predictions = self.model(input_tensor)

    for inp_data, preds in zip(input_data, predictions):
      boxes = preds["boxes"].cpu().numpy()
      labels = [[pred] for pred in preds["labels"].detach().cpu().numpy()]
      scores = [[pred] for pred in preds["scores"].detach().cpu().numpy()]
      h, w, _ = inp_data.shape  # input image shape
      bboxes = [[x[1] / h, x[0] / w, x[3] / h, x[2] / w]
                for x in boxes]  # normalize the bboxes to [0,1]
      bboxes = np.clip(bboxes, 0, 1)
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
