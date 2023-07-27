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
from transformers import pipeline

from clarifai.models.model_serving.models.model_types import text_to_text
from clarifai.models.model_serving.models.output import TextOutput


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.huggingface_model_path = os.path.join(self.base_path, "bart-large-summarizer")
    self.pipeline = pipeline("summarization", model=self.huggingface_model_path)

  @text_to_text
  def get_predictions(self, input_data):
    """
    Generates summaries of input text.

    Args:
    -----
      input_data: A single input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

    Returns:
    --------
      One of the clarifai.models.model_serving.models.output types. Refer to the README/docs
    """
    summary = self.pipeline(input_data, max_length=50, min_length=30, do_sample=False)
    generated_text = np.array([summary[0]['summary_text']], dtype=object)
    return TextOutput(predicted_text=generated_text)
