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
from transformers import AutoModel, ViTImageProcessor

from clarifai.models.model_serving.models.model_types import visual_embedder
from clarifai.models.model_serving.models.output import EmbeddingOutput


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "vit-base-patch16-224")
    self.processor = ViTImageProcessor.from_pretrained(self.huggingface_model_path)
    self.model = AutoModel.from_pretrained(self.huggingface_model_path)

  @visual_embedder
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
      embedding_vector = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()

    return EmbeddingOutput(embedding_vector=embedding_vector[0])
