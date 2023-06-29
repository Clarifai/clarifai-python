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

BASE_PATH = os.path.dirname(__file__)


def setup():
  """
  Load inference model.
  The model checkpoint(s) should be saved in the same directory and directory
  level as this inference.py module.

  Returns:
  --------
    Inference Model Callable
  """
  # sample model loading code:
  # checkpoint_path = os.path.join(BASE_PATH, your_checkpoint_name)
  # model = <load model from checkpoint_path>
  # return model

  # Delete/Comment out line below and add your code
  raise NotImplementedError()


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
  # Delete/Comment out line below and add your inference code
  raise NotImplementedError()
