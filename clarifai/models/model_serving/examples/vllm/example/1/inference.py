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

from vllm import LLM, SamplingParams

from clarifai.models.model_serving.model_config import ModelTypes, get_model_config

config = get_model_config(ModelTypes.text_to_text)


class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    path = os.path.join(self.base_path, "weights")
    self.model = LLM(
        model=path,
        dtype="float16",
        gpu_memory_utilization=0.7,
        swap_space=1,
        #quantization="awq"
    )

  @config.inference.wrap_func
  def get_predictions(self, input_data: list, **kwargs):
    """
    Main model inference method.

    Args:
    -----
      input_data: A single input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.models.output types` or `config.inference.return_type(your_output)`. Refer to the README/docs
    """
    sampling_params = SamplingParams(**kwargs)
    preds = self.model.generate(input_data, sampling_params)
    outputs = [config.inference.return_type(each.outputs[0].text) for each in preds]

    return outputs
