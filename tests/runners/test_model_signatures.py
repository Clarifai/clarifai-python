import unittest
from pprint import pprint
from typing import List

import numpy as np
from PIL import Image

from clarifai.client.model_client import ModelClient
from clarifai.runners.models.model_class import ModelClass, methods
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.runners.utils.data_handler import Concept


class TestModelSignatures(unittest.TestCase):

  def test_int__int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int) -> int:
        return 2 * x

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'return',
                'streaming': False
            }]
        })
    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(5)
    self.assertEqual(result, 10)

  def test_str_str__str(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str, y: str) -> str:
        return x + y

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'parts[x].string_value',
                'data_type': 'str',
                'name': 'x',
                'required': True,
                'streaming': False
            }, {
                'data_field': 'parts[y].string_value',
                'data_type': 'str',
                'name': 'y',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': False
            }]
        })
    # test calls
    client = _get_servicer_client(MyModel())

    result = client.f('a', 'b')
    self.assertEqual(result, 'ab')

    result = client.f(x='a', y='b')
    self.assertEqual(result, 'ab')

    result = client.f('a', y='b')
    self.assertEqual(result, 'ab')

    self.assertRaises(TypeError, lambda: client.f('a'))
    self.assertRaises(TypeError, lambda: client.f('a', x='a'))
    self.assertRaises(TypeError, lambda: client.f('a', 'b', 'c'))
    self.assertRaises(TypeError, lambda: client.f(x='a', z='c'))

  def test_str_int__str(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str, y: int) -> str:
        return x

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'x',
                'required': True,
                'streaming': False
            }, {
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'y',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': False
            }]
        })

  def test_ndarray__int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: np.ndarray) -> int:
        return 2 * x

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'ndarray',
                'data_type': 'ndarray',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'return',
                'streaming': False
            }]
        })

  def test_Image__str(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: Image) -> str:
        return x

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'image',
                'data_type': 'Image',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': False
            }]
        })

  def test_str__Image(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str) -> Image:
        return x

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'image',
                'data_type': 'Image',
                'name': 'return',
                'streaming': False
            }]
        })

  def test_Image__ListConcept(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: Image) -> List[Concept]:
        return [Concept('a', 0.9), Concept('b', 0.1)]

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'image',
                'data_type': 'Image',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'concepts',
                'data_type': 'List[Concept]',
                'name': 'return',
                'streaming': False
            }]
        })

  def test_str_ListImage__str_ListImage(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, prompt: str, images: List[Image]) -> (str, List[Image]):
        return (prompt, images)

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'prompt',
                'required': True,
                'streaming': False
            }, {
                'data_field': 'parts[].image',
                'data_type': 'List[Image]',
                'name': 'images',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return.0',
                'streaming': False
            }, {
                'data_field': 'parts[].image',
                'data_type': 'List[Image]',
                'name': 'return.1',
                'streaming': False
            }]
        })

  def test_ndarrayint__ndarrayfloat(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: np.ndarray[int]) -> np.ndarray[float]:
        return x / 2.0

    #pprint( MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'ndarray',
                'data_type': 'ndarray',
                'name': 'x',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'predict',
            'name':
                'f',
            'outputs': [{
                'data_field': 'ndarray',
                'data_type': 'ndarray',
                'name': 'return',
                'streaming': False
            }]
        })


def _get_servicer_client(model):
  servicer = ModelServicer(model)
  client = ModelClient(servicer)
  return client


if __name__ == '__main__':
  unittest.main()
