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

import torch
from transformers import CLIPModel, CLIPProcessor
from clarifai.models.model_serving.model_config import ModelTypes, get_model_config

config = get_model_config(ModelTypes.multimodal_embedder)


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
    self.model = CLIPModel.from_pretrained(os.path.join(self.base_path, "checkpoint"))
    self.model.eval()
    #self.text_model = CLIPTextModel.from_pretrained(os.path.join(self.base_path, "openai/clip-vit-base-patch32"))
    self.processor = CLIPProcessor.from_pretrained(os.path.join(self.base_path, "checkpoint"))

  #Add relevant model type decorator to the method below (see docs/model_types for ref.)
  @config.inference.wrap_func
  def get_predictions(self, input_data, **kwargs):
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
    outputs = []
    for inp in input_data:
      image, text = inp["image"], inp["text"]
      with torch.no_grad():
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        if text is not None:
          inputs = self.processor(text=text, return_tensors="pt", padding=True)
          embeddings = self.model.get_text_features(**inputs)
        else:
          inputs = self.processor(images=image, return_tensors="pt", padding=True)
          embeddings = self.model.get_image_features(**inputs)
      embeddings = embeddings.squeeze().cpu().numpy()
      outputs.append(config.inference.return_type(embedding_vector=embeddings))

    return outputs
