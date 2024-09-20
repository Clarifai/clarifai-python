import unittest
from typing import Any, Dict, Iterator, List

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import struct_pb2
from PIL import Image

from clarifai.runners.models import AnyAnyModel
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


class _AnyModel(AnyAnyModel):

  def load_model(self):
    pass

  def predict(self, input_data: List[Dict],
              inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:
    outputs = []
    for each_input in input_data:
      image = each_input.get("image")
      text = each_input.get("text")
      audio = each_input.get("audio")
      if isinstance(image, np.ndarray):
        image = EXPECTED_IMAGE
      if text:
        text = self.runner_id
      if audio:
        audio = EXPECTED_AUDIO

      output = OutputDataHandler.from_data(
          status_code=status_code_pb2.SUCCESS, text=text, image=image, audio=audio)
      outputs.append(output)
    return outputs

  def stream(self, inputs: Iterator[List[Dict[str, Any]]],
             inference_params: Dict[str, Any]) -> List[OutputDataHandler]:  # type: ignore
    for input_data in inputs:
      yield self.predict(input_data, inference_params)

  def generate(self, input_data: List[Dict],
               inference_parameters: Dict[str, Any]) -> List[OutputDataHandler]:  # type: ignore
    yield self.predict(input_data, inference_parameters)


class TestAnyAnyModel(unittest.TestCase):

  def setUp(self):
    self.model = _AnyModel(
        runner_id="any-anymodel",
        nodepool_id="fake-nodepool",
        compute_cluster_id="fake-compute_cluster_id",
    )

  def _test_predict(self):
    input_data = dict(image=IMAGE, text=TEXT, audio=AUDIO)
    outputs = self.model.predict([input_data] * 2,)
    self.assertEqual(len(outputs), 2)
    for output in outputs:
      self.assertEqual(output.image.all(), EXPECTED_IMAGE.all())
      self.assertEqual(output.audio, EXPECTED_AUDIO)
      self.assertEqual(output.text, self.model.runner_id)

  def test_parse_input_request(self):
    python_params = dict(a=1, b=True, c="abc")
    params = struct_pb2.Struct()
    params.update(python_params)
    input_req = service_pb2.PostModelOutputsRequest(
        inputs=[INPUT_DATA_PROTO],
        model=resources_pb2.Model(output_info=resources_pb2.OutputInfo(params=params)),
    )
    input_data, infer_params = self.model.parse_input_request(input_req)

    for inp in input_data:
      self.assertEqual(inp.get("image").all(), IMAGE.all())
      self.assertEqual(inp.get("audio"), AUDIO)
      self.assertEqual(inp.get("text"), TEXT)

    self.assertEqual(python_params.get("a"), infer_params.get("a"))
    self.assertEqual(python_params.get("b"), infer_params.get("b"))
    self.assertEqual(python_params.get("c"), infer_params.get("c"))

  def test_convert_output_to_proto(self):
    outputs = [OutputDataHandler.from_data(image=IMAGE, text=TEXT, audio=AUDIO)]
    multi_output_response = self.model.convert_output_to_proto(outputs)

    for output in multi_output_response.outputs:
      self.assertEqual(output.data.image.base64, image_to_bytes(Image.fromarray(IMAGE)))
      self.assertEqual(output.data.audio.base64, AUDIO)
      self.assertEqual(output.data.text.raw, TEXT)

  def test_predict_wrapper(self):
    input_req = service_pb2.PostModelOutputsRequest(inputs=[INPUT_DATA_PROTO] * 2)
    output_protos = self.model.predict_wrapper(input_req)
    count = 0
    for output in output_protos.outputs:
      count += 1
      self.assertEqual(output.data.image.base64, image_to_bytes(Image.fromarray(EXPECTED_IMAGE)))
      self.assertEqual(output.data.audio.base64, EXPECTED_AUDIO)
      self.assertEqual(output.data.text.raw, self.model.runner_id)

    self.assertEqual(count, 2)

  def _preprocess_stream_iter(self, input_req):
    for req in input_req:
      yield req

  def test_stream_wrapper(self):
    input_req = [service_pb2.PostModelOutputsRequest(inputs=[INPUT_DATA_PROTO] * 2)]
    output_protos = self.model.stream_wrapper(self._preprocess_stream_iter(input_req))
    count = 0
    for each in output_protos:
      for output in each.outputs:
        count += 1
        self.assertEqual(output.data.image.base64, image_to_bytes(Image.fromarray(EXPECTED_IMAGE)))
        self.assertEqual(output.data.audio.base64, EXPECTED_AUDIO)
        self.assertEqual(output.data.text.raw, self.model.runner_id)

    self.assertEqual(count, 2)
