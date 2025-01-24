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
    # print all files in the checkpoints folder recursively
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
      print(self.tokenizer.decode(outputs[0]))
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


if __name__ == "__main__":
  runner = MyRunner(runner_id="a", nodepool_id="b", compute_cluster_id="c", user_id="d")
  runner.load_model()

  import concurrent.futures
  import random
  import string

  def generate_random_word(length=5):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

  def generate_random_sentence(word_count=1000, word_length=5):
    return ' '.join(generate_random_word(word_length) for _ in range(word_count))

  def generate_random_sentences(num_sentences=100, word_count=1000, word_length=5):
    return [generate_random_sentence(word_count, word_length) for _ in range(num_sentences)]

  input_texts = generate_random_sentences(num_sentences=10, word_count=5, word_length=5)

  def run_prediction(runner, input_text):
    print(input_text)
    response_generator = runner.predict(
        service_pb2.PostModelOutputsRequest(
            model=resources_pb2.Model(model_version=resources_pb2.ModelVersion(id="")),
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(
                    text=resources_pb2.Text(raw="hello how are you doing?")  # input_text),
                    # image=resources_pb2.Image(url="https://goodshepherdcentres.ca/wp-content/uploads/2020/06/home-page-banner.jpg")
                ))
            ],
        ))
    return response_generator

  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(run_prediction, runner, text) for text in input_texts]
    for future in concurrent.futures.as_completed(futures):
      try:
        response = future.result()
        print(response)
      except Exception as e:
        print(f"Prediction failed: {e}")
  """
  # send an inference.
  response_generator = runner.generate(
          service_pb2.PostModelOutputsRequest(
              model=resources_pb2.Model(model_version=resources_pb2.ModelVersion(id="")),
              inputs=[
                  resources_pb2.Input(
                    data=resources_pb2.Data(
                      text=resources_pb2.Text(raw="How many people are in this image?"),
                      # image=resources_pb2.Image(url="https://goodshepherdcentres.ca/wp-content/uploads/2020/06/home-page-banner.jpg")
                    )
                  )
              ],
          ))
  for response in response_generator:
    print(response)
  """
