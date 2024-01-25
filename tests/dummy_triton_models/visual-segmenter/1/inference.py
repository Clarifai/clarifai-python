# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from clarifai.models.model_serving.model_config import MasksOutput, VisualSegmenter


class InferenceModel(VisualSegmenter):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `visual-segmenter` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of MasksOutput

    """

    outputs = []

    for inp in input_data:
      assert isinstance(inp, np.ndarray), "Incorrect type of image, expected np.ndarray"
      output = np.random.randint(0, 1, size=(200, 200))
      output = MasksOutput(output)
      outputs.append(output)

    return outputs
