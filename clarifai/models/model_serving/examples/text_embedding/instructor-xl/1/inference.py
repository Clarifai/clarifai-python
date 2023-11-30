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

# Set up env for huggingface
ROOT_PATH = os.path.join(os.path.dirname(__file__))
PIPELINE_PATH = os.path.join(ROOT_PATH, 'checkpoint')

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # noqa
os.environ['TRANSFORMERS_CACHE'] = PIPELINE_PATH  # noqa
os.environ['TORCH_HOME'] = PIPELINE_PATH

import torch  # noqa
from InstructorEmbedding import INSTRUCTOR  # noqa

from clarifai.models.model_serving.model_config import (  # noqa # pylint: disable=unused-import
    ModelTypes, get_model_config)

config = get_model_config("text-embedder")


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    ## sample model loading code:
    #self.checkpoint_path: Path = os.path.join(self.base_path, "your checkpoint filename/path")
    #self.model: Callable = <load_your_model_here from checkpoint or folder>
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = INSTRUCTOR('hkunlp/instructor-xl')

  @config.inference.wrap_func
  def get_predictions(self, input_data: list, **kwargs) -> list:
    """
    Main model inference method.

    Args:
    -----
      input_data: A list of input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

      **kwargs: your inference parameters.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.models.output types` or `config.inference.return_type(your_output)`. Refer to the README/docs
    """
    batch_preds = self.model.encode(input_data, device=self.device)

    return [config.inference.return_type(each) for each in batch_preds]
