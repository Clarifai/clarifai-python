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

import torch
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from clarifai.models.model_serving.models.model_types import visual_segmenter
from clarifai.models.model_serving.models.output import MasksOutput


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "segformer_b2_clothes")
    #self.labels_path = os.path.join(Path(self.base_path).parents[0], "labels.txt")
    self.processor = SegformerImageProcessor.from_pretrained(self.huggingface_model_path)
    self.model = AutoModelForSemanticSegmentation.from_pretrained(self.huggingface_model_path)

  @visual_segmenter
  def get_predictions(self, input_data):
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
    inputs = self.processor(images=input_data, return_tensors="pt")
    with torch.no_grad():
      output = self.model(**inputs)

    logits = output.logits.cpu()
    mask = logits.argmax(dim=1)[0].numpy()

    return MasksOutput(predicted_mask=mask)
