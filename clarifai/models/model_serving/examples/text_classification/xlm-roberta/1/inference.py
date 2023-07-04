# This file contains boilerplate code to allow users write their model
# inference code that will then interact with the Triton Inference Server
# Python backend to serve end user requests.
# The module name, module path and the setup() & predict() function names MUST be maintained as is
# but other functions may be added within this module as deemed fit provided
# they are invoked within the main predict() function if they play a role any
# step of model inference
"""User model inference script."""

import os
from typing import Callable

from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clarifai.models.model_serving.models.model_types import text_classifier
from clarifai.models.model_serving.models.output import ClassifierOutput

BASE_PATH = os.path.dirname(__file__)
HUGGINGFACE_MODEL_PATH = os.path.join(BASE_PATH, "twitter-xlm-roberta-base-sentiment")
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)


def setup():
  """
  Load inference model.
  The model checkpoint(s) should be saved in the same directory and directory
  level as this inference.py module.

  Returns:
  --------
    Inference Model Callable
  """
  model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)

  return model


@text_classifier
def predict(input_data, model: Callable):
  """
  Main model inference function.

  Args:
  -----
    input_data: A single input data item to predict on.
      Input data can be an image or text, etc depending on the model type.
    model: Inference model callable as returned by setup() above.

  Returns:
  --------
    One of the clarifai.model_serving.models.output types. Refer to the README/docs
  """
  encoded_input = tokenizer(input_data, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)

  return ClassifierOutput(predicted_scores=scores)
