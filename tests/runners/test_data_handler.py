import unittest

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from PIL import Image

from clarifai.models.runners.clarifai_runners.utils.data_handler import (InputDataHandler,
                                                                         OutputDataHandler)
from clarifai.models.runners.clarifai_runners.utils.data_utils import image_to_bytes

IMAGE = np.ones([50, 50, 3], dtype="uint8")
AUDIO = b"000"
TEXT = "ABC"
CONCEPTS = dict(a=0.0, b=0.2, c=1.0)
EMBEDDINGS = [0.1, 1.1, 2.0]

INPUT_DATA_PROTO = resources_pb2.Input(data=resources_pb2.Data(
    image=resources_pb2.Image(base64=image_to_bytes(Image.fromarray(IMAGE))),
    text=resources_pb2.Text(raw=TEXT),
    audio=resources_pb2.Audio(base64=AUDIO),
))


class TestDataHandler(unittest.TestCase):

  def test_input_proto_to_python(self):
    input_data = InputDataHandler.from_proto(proto=INPUT_DATA_PROTO)

    all_data = input_data.to_python()

    assert all_data.get("image").all() == IMAGE.all()
    assert all_data.get("audio") == AUDIO
    assert all_data.get("text") == TEXT

  def test_output_python_to_proto(self):
    output = OutputDataHandler.from_data(
        image=IMAGE, text=TEXT, audio=AUDIO, concepts=CONCEPTS, embeddings=[EMBEDDINGS])

    output_proto = output.proto

    assert output_proto.data.image.base64 == image_to_bytes(Image.fromarray(IMAGE))
    assert output_proto.data.audio.base64 == AUDIO
    assert output_proto.data.text.raw == TEXT

    for con_score in output_proto.data.concepts:
      self.assertAlmostEqual(CONCEPTS[con_score.id], con_score.value)

    for el_inp, el_out in zip(EMBEDDINGS, output_proto.data.embeddings[0].vector):
      self.assertAlmostEqual(el_inp, el_out)
    self.assertEqual(len(EMBEDDINGS), output_proto.data.embeddings[0].num_dimensions)
