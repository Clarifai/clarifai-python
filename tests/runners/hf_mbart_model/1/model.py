import os
from typing import Iterator

import torch
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger

NUM_GPUS = 1


def set_output(texts: list):
  assert isinstance(texts, list)
  output_protos = []
  for text in texts:
    output_protos.append(
        resources_pb2.Output(
            data=resources_pb2.Data(text=resources_pb2.Text(raw=text)),
            status=status_pb2.Status(code=status_code_pb2.SUCCESS)))
  return output_protos


class MyRunner(ModelRunner):
  """A custom runner that loads the model and generates text using lmdeploy inference.
  """

  def load_model(self):
    """Load the model here"""
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Running on device: {self.device}")
    checkpoints = os.path.join(os.path.dirname(__file__), "checkpoints")

    for root, dirs, files in os.walk(checkpoints):
      for f in files:
        logger.info(os.path.join(root, f))

    # if checkpoints section is in config.yaml file then checkpoints will be downloaded at this path during model upload time.
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoints)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoints, torch_dtype="auto", device_map=self.device)

  def predict(self, request: service_pb2.PostModelOutputsRequest
             ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """
    texts = [inp.data.text.raw for inp in request.inputs]

    raw_texts = []
    for t in texts:
      inputs = self.tokenizer.encode(t, return_tensors="pt").to(self.device)
      outputs = self.model.generate(inputs)
      raw_texts.append(self.tokenizer.decode(outputs[0]))
    output_protos = set_output(raw_texts)

    return service_pb2.MultiOutputResponse(outputs=output_protos)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""
    raise NotImplementedError("This method is not implemented yet.")

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    pass
