import unittest

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from PIL import Image

from clarifai.runners.utils.data_utils import DataConverter, image_to_bytes

IMAGE = np.ones([50, 50, 3], dtype="uint8")
AUDIO = b"000"
TEXT = "ABC"
CONCEPTS = dict(a=0.0, b=0.2, c=1.0)
EMBEDDINGS = [0.1, 1.1, 2.0]

INPUT_DATA_PROTO = resources_pb2.Input(
    data=resources_pb2.Data(
        image=resources_pb2.Image(base64=image_to_bytes(Image.fromarray(IMAGE))),
        text=resources_pb2.Text(raw=TEXT),
        audio=resources_pb2.Audio(base64=AUDIO),
    )
)


class TestDataHandler(unittest.TestCase):
    def test_input_proto_to_python(self):
        pass

    def test_output_python_to_proto(self):
        pass


class TestIsOldFormat(unittest.TestCase):
    """Tests for DataConverter.is_old_format()."""

    def test_empty_data_returns_false(self):
        """An empty Data proto has no fields set — not old format."""
        data = resources_pb2.Data()
        self.assertFalse(DataConverter.is_old_format(data))

    def test_parts_present_returns_false(self):
        """Data with parts is new format."""
        data = resources_pb2.Data(parts=[resources_pb2.Part()])
        self.assertFalse(DataConverter.is_old_format(data))

    # ---- singular message fields ----

    def test_text_set_returns_true(self):
        data = resources_pb2.Data(text=resources_pb2.Text(raw="hello"))
        self.assertTrue(DataConverter.is_old_format(data))

    def test_image_set_returns_true(self):
        data = resources_pb2.Data(image=resources_pb2.Image(url="http://example.com/img.jpg"))
        self.assertTrue(DataConverter.is_old_format(data))

    # ---- scalar primitive fields (non-default values → True) ----

    def test_int_value_nonzero_returns_true(self):
        data = resources_pb2.Data(int_value=42)
        self.assertTrue(DataConverter.is_old_format(data))

    def test_int_value_zero_returns_false(self):
        data = resources_pb2.Data(int_value=0)
        self.assertFalse(DataConverter.is_old_format(data))

    def test_float_value_nonzero_returns_true(self):
        data = resources_pb2.Data(float_value=3.14)
        self.assertTrue(DataConverter.is_old_format(data))

    def test_float_value_zero_returns_false(self):
        data = resources_pb2.Data(float_value=0.0)
        self.assertFalse(DataConverter.is_old_format(data))

    def test_bytes_value_nonempty_returns_true(self):
        data = resources_pb2.Data(bytes_value=b"\x01\x02")
        self.assertTrue(DataConverter.is_old_format(data))

    def test_bytes_value_empty_returns_false(self):
        data = resources_pb2.Data(bytes_value=b"")
        self.assertFalse(DataConverter.is_old_format(data))

    def test_bool_value_true_returns_true(self):
        data = resources_pb2.Data(bool_value=True)
        self.assertTrue(DataConverter.is_old_format(data))

    def test_bool_value_false_returns_false(self):
        data = resources_pb2.Data(bool_value=False)
        self.assertFalse(DataConverter.is_old_format(data))

    def test_string_value_nonempty_returns_true(self):
        data = resources_pb2.Data(string_value="hello")
        self.assertTrue(DataConverter.is_old_format(data))

    def test_string_value_empty_returns_false(self):
        data = resources_pb2.Data(string_value="")
        self.assertFalse(DataConverter.is_old_format(data))

    # ---- repeated fields ----

    def test_concepts_set_returns_true(self):
        data = resources_pb2.Data(concepts=[resources_pb2.Concept(name="cat")])
        self.assertTrue(DataConverter.is_old_format(data))

    def test_embeddings_set_returns_true(self):
        data = resources_pb2.Data(embeddings=[resources_pb2.Embedding(vector=[0.1, 0.2, 0.3])])
        self.assertTrue(DataConverter.is_old_format(data))
