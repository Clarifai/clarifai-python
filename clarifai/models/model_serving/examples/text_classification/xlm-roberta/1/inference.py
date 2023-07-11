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
from typing import Callable

import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clarifai.models.model_serving.models.model_types import text_classifier
from clarifai.models.model_serving.models.output import ClassifierOutput


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    self.checkpoint_path: Path = os.path.join(self.base_path, "twitter-xlm-roberta-base-sentiment")
    self.model: Callable = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_path)
    self.tokenizer: Callable = AutoTokenizer.from_pretrained(self.checkpoint_path)
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  @text_classifier
  def get_predictions(self, input_data) -> ClassifierOutput:
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
    encoded_input = self.tokenizer(input_data, return_tensors='pt')
    output = self.model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return ClassifierOutput(predicted_scores=scores)
