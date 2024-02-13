# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union
from clarifai.models.model_serving.model_config import *  # noqa


class InferenceModel():
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
    """predict_docstring
    """

    raise NotImplementedError()
