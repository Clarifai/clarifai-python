# User model inference script.

import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from clarifai.models.model_serving.model_config import EmbeddingOutput, MultiModalEmbedder


class InferenceModel(MultiModalEmbedder):
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
    """ Custom prediction function for `multimodal-embedder` model.

    Args:
      input_data (_MultiModalEmbdderInputTypeDict): dict of key-value: `image`(List[np.ndarray]) and `text` (List[str])
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput

    """
    outputs = []
    for inp_data in input_data:
      image, text = inp_data.get("image", None), inp_data.get("text", None)
      if text is not None:
        assert isinstance(text, str), "Incorrect type of text, expected str"
        embeddings = np.zeros(768)
      else:
        assert isinstance(image, np.ndarray), "Incorrect type of image, expected np.ndarray"
        embeddings = np.ones(768)
      outputs.append(EmbeddingOutput(embedding_vector=embeddings))

    return outputs
