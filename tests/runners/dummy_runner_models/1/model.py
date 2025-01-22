from typing import Iterator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.runners.models.model_runner import ModelRunner


class MyRunner(ModelRunner):
  """A custom runner that adds "Hello World" to the end of the text."""

  def load_model(self):
    """Load the model here."""

  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This is the method that will be called when the runner is run. It takes in an input and
    returns an output.
    """

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    outputs = []
    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      output = resources_pb2.Output()

      data = inp.data

      # Optional use of output_info
      params_dict = {}
      if "params" in output_info:
        params_dict = output_info["params"]

      if data.text.raw != "":
        output.data.text.raw = data.text.raw + "Hello World" + params_dict.get("hello", "")
      if data.image.url != "":
        output.data.text.raw = data.image.url.replace(
            "samples.clarifai.com",
            "newdomain.com" + params_dict.get("domain", ""),
        )

      output.status.code = status_code_pb2.SUCCESS
      outputs.append(output)
    return service_pb2.MultiOutputResponse(outputs=outputs,)

  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)
    # Optional use of output_info
    params_dict = {}
    if "params" in output_info:
      params_dict = output_info["params"]

    for i in range(10):  # fake something iterating generating 10 times.
      outputs = []
      for inp in request.inputs:
        output = resources_pb2.Output()
        if inp.data.text.raw != "":
          output.data.text.raw = f"{inp.data.text.raw}Generate Hello World {i}" + params_dict.get(
              "hello", "")

        output.status.code = status_code_pb2.SUCCESS
        outputs.append(output)
      yield service_pb2.MultiOutputResponse(outputs=outputs,)

  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    """Example yielding a whole batch of streamed stuff back."""

    for ri, request in enumerate(request_iterator):
      if ri == 0:  # only first request has model information.
        model = request.model
        output_info = None
        if request.model.model_version.id != "":
          output_info = json_format.MessageToDict(
              model.model_version.output_info, preserving_proto_field_name=True)
        # Optional use of output_info
        params_dict = {}
        if "params" in output_info:
          params_dict = output_info["params"]
      for i in range(10):  # fake something iterating generating 10 times.
        outputs = []
        for inp in request.inputs:
          output = resources_pb2.Output()
          if inp.data.text.raw != "":
            out_text = inp.data.text.raw + f"Stream Hello World {i}" + params_dict.get("hello", "")
          output.status.code = status_code_pb2.SUCCESS
          print(out_text)
          output.data.text.raw = out_text
          outputs.append(output)
        yield service_pb2.MultiOutputResponse(outputs=outputs,)
