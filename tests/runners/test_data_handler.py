import unittest

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from PIL import Image

from clarifai.runners.utils.data_utils import image_to_bytes

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
    pass

  def test_output_python_to_proto(self):
    pass
