# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from clarifai.models.model_serving.model_config import EmbeddingOutput, TextEmbedder


class InferenceModel(TextEmbedder):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)

  def predict(self,
              input_data: list,
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}) -> list:
    """ Custom prediction function for `text-embedder` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput

    """

    outputs = []

    for inp in input_data:
      assert isinstance(inp, str), "Incorrect type of text, expected str"
      output = np.random.randn(768)
      output = EmbeddingOutput(output)
      outputs.append(output)

    return outputs
