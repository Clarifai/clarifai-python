import os
from typing import Iterator

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_handler import Output


class MyModel(ModelClass):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoints, torch_dtype="auto", device_map=self.device)

  def predict(self, prompt: str = "") -> Output:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
    outputs = self.model.generate(inputs)
    output_text = self.tokenizer.decode(outputs[0])
    return Output(text=output_text)

  def generate(self, prompt: str = "") -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""
    raise NotImplementedError("This method is not implemented yet.")

  def stream(self, input_iterator) -> Iterator[Output]:
    """Example yielding a whole batch of streamed stuff back."""
    pass
