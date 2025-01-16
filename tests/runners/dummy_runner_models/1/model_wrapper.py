from typing import Any, Dict, List

from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.runners.models.base_typed_model import TextInputModel
from clarifai.runners.utils.data_handler import OutputDataHandler


class MyRunner(TextInputModel):
  """A custom runner that adds "Hello World" to the end of the text"""

  def load_model(self):
    pass

  def predict(self, input_data: List[str],
              inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:
    outputs = []
    for input_text in input_data:
      output_text = input_text + "Hello World" + inference_parameters.get("hello", "")
      output = OutputDataHandler.from_data(
          status_code=status_code_pb2.SUCCESS,
          text=output_text,
      )
      outputs.append(output)
    return outputs

  def stream(self, inputs, inference_parameters) -> List[OutputDataHandler]:  # type: ignore
    outputs = []
    for i, each_input in enumerate(inputs):
      list_text = each_input
      output = OutputDataHandler.from_data(
          status_code=status_code_pb2.SUCCESS,
          text=f"{list_text[0]}Stream Hello World {i}" + inference_parameters.get("hello", ""),
      )
      outputs.append(output)
    yield outputs

  def generate(self, input_data: List[str],
               inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:  # type: ignore
    outputs = []
    for i, input_text in enumerate(input_data):
      output = OutputDataHandler.from_data(
          status_code=status_code_pb2.SUCCESS,
          text=f"{input_text}Generate Hello World {i}" + inference_parameters.get("hello", ""),
      )
      outputs.append(output)
    yield outputs
