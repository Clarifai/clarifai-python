import time

import requests
from clarifai_grpc.grpc.api import resources_pb2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from clarifai.client.runner import Runner

# This example requires to run the following before running this example:
# pip install transformers

# https://huggingface.co/TheBloke/Llama-2-70B-chat-GPTQ
model_name_or_path = "TheBloke/Llama-2-7B-chat-GPTQ"
model_basename = "model"

use_triton = False


class Llama2Runner(Runner):
  """A custom runner that runs the LLama2 LLM.
  """

  def __init__(self, *args, **kwargs):
    super(Llama2Runner, self).__init__(*args, **kwargs)
    self.logger.info("Starting to load the model...")
    st = time.time()
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto')

    self.logger.info("Loading model complete in (%f seconds), ready to loop for requests." %
                     (time.time() - st))

  def run_input(self, input: resources_pb2.Input,
                output_info: resources_pb2.OutputInfo) -> resources_pb2.Output:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    output = resources_pb2.Output()
    data = input.data
    if data.text.raw != "":
      input_text = data.text.raw
    elif data.text.url != "":
      input_text = str(requests.get(data.text.url).text)
    else:
      raise Exception("Need to include data.text.raw or data.text.url in your inputs.")

    if "params" in output_info:
      params_dict = output_info["params"]
      self.logger.info("params_dict: %s", params_dict)

    time.time()
    max_tokens = 1024
    # # Method 1
    # input_ids = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
    # out = self.model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=max_tokens)
    # out_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
    # output.data.text.raw = out_text.replace(input_text, '')

    # # Method 2
    pipe = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False)
    a = pipe(input_text)
    output.data.text.raw = a[0]['generated_text']
    return output


if __name__ == '__main__':
  # Make sure you set these env vars before running the example.
  # CLARIFAI_PAT
  # CLARIFAI_USER_ID

  # You need to first create a runner in the Clarifai API and then use the ID here.
  Llama2Runner(runner_id="sdk-llama2-runner").start()
