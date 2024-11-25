import unittest
from typing import Any, Dict, Iterator, List

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import struct_pb2
from PIL import Image

from clarifai.runners.models.base_typed_model import TextInputModel
from clarifai.runners.utils.data_handler import OutputDataHandler
from clarifai.runners.utils.data_utils import image_to_bytes

IMAGE = np.ones([50, 50, 3], dtype="uint8")
AUDIO = b"000"
TEXT = "ABC"

INPUT_DATA_PROTO = resources_pb2.Input(data=resources_pb2.Data(
    image=resources_pb2.Image(base64=image_to_bytes(Image.fromarray(IMAGE))),
    text=resources_pb2.Text(raw=TEXT),
    audio=resources_pb2.Audio(base64=AUDIO),
))

EXPECTED_IMAGE = np.zeros([200, 200, 3], dtype="uint8")
EXPECTED_AUDIO = b"111"


class _TextGenerationModel(TextInputModel):

  def load_model(self):
    pass

  def predict(self, input_data: List[str],
              inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:
    outputs = []
    for text in input_data:
      text = self.runner_id
      output = OutputDataHandler.from_data(
          status_code=status_code_pb2.SUCCESS,
          text=text,
      )
      outputs.append(output)
    return outputs

  def stream(self, inputs: Iterator[List[str]],
             inference_params: Dict[str, Any]) -> List[OutputDataHandler]:  # type: ignore
    for input_data in inputs:
      yield self.predict(input_data, inference_params)

  def generate(self, input_data: List[str],
               inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:  # type: ignore
    yield self.predict(input_data, inference_parameters)


class TestTextInputModel(unittest.TestCase):

  def setUp(self):
    self.model = _TextGenerationModel(
        runner_id="any-anymodel",
        nodepool_id="fake-nodepool",
        compute_cluster_id="fake-compute_cluster_id",
    )

  def _test_predict(self):
    input_data = [TEXT]
    outputs = self.model.predict([input_data] * 2,)
    self.assertEqual(len(outputs), 2)
    for output in outputs:
      self.assertEqual(output.image, None)
      self.assertEqual(output.audio, None)
      self.assertEqual(output.text, self.model.runner_id)

  def test_parse_input_request(self):
    python_params = dict(a=1, b=True, c="abc")
    params = struct_pb2.Struct()
    params.update(python_params)
    input_req = service_pb2.PostModelOutputsRequest(
        inputs=[INPUT_DATA_PROTO],
        model=resources_pb2.Model(model_version=resources_pb2.ModelVersion(
            output_info=resources_pb2.OutputInfo(params=params))),
    )
    input_data, infer_params = self.model.parse_input_request(input_req)

    for inp in input_data:
      self.assertEqual(inp, TEXT)

    self.assertEqual(python_params.get("a"), infer_params.get("a"))
    self.assertEqual(python_params.get("b"), infer_params.get("b"))
    self.assertEqual(python_params.get("c"), infer_params.get("c"))

  def test_convert_output_to_proto(self):
    outputs = [OutputDataHandler.from_data(text=TEXT)]
    multi_output_response = self.model.convert_output_to_proto(outputs)

    for output in multi_output_response.outputs:
      self.assertFalse(output.data.image.base64)
      self.assertFalse(output.data.audio.base64)
      self.assertEqual(output.data.text.raw, TEXT)

  def test_predict_wrapper(self):
    input_req = service_pb2.PostModelOutputsRequest(inputs=[INPUT_DATA_PROTO] * 2)
    output_protos = self.model.predict_wrapper(input_req)
    count = 0
    for output in output_protos.outputs:
      count += 1
      self.assertFalse(output.data.image.base64)
      self.assertFalse(output.data.audio.base64)
      self.assertEqual(output.data.text.raw, self.model.runner_id)

    self.assertEqual(count, 2)

  def _preprocess_stream_iter(self, input_req):
    for req in input_req:
      yield req

  def test_stream_wrapper(self):
    input_req = [service_pb2.PostModelOutputsRequest(inputs=[INPUT_DATA_PROTO] * 2)] * 2
    output_protos = self.model.stream_wrapper(self._preprocess_stream_iter(input_req))
    count = 0
    for each in output_protos:
      for output in each.outputs:
        count += 1
        self.assertFalse(output.data.image.base64)
        self.assertFalse(output.data.audio.base64)
        self.assertEqual(output.data.text.raw, self.model.runner_id)

    self.assertEqual(count, 4)
