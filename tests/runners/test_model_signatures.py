import unittest
from typing import List

import numpy as np
from PIL import Image, ImageOps

from clarifai.client.model_client import ModelClient
from clarifai.runners.models.model_class import ModelClass, methods
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.runners.utils.data_handler import Concept, Output, Stream

_ENABLE_PPRINT = False

if _ENABLE_PPRINT:
  from pprint import pprint
else:

  def pprint(*args, **kwargs):
    pass


class TestModelCalls(unittest.TestCase):

  def test_int__int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int) -> int:
        return 2 * x

    pprint(MyModel._get_method_info('f').signature)
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

    pprint(MyModel._get_method_info('f').signature)
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
        return x + str(y)

    pprint(MyModel._get_method_info('f').signature)
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a', 5)
    self.assertEqual(result, 'a5')

  def test_ndarray__int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: np.ndarray) -> int:
        return int(np.sum(x))

    pprint(MyModel._get_method_info('f').signature)
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(np.array([1, 2, 3]))
    self.assertEqual(result, 6)

  def test_Image__str(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: Image) -> str:
        return str(x.size)

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
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': False
            }]
        })

    # test call
    client = _get_servicer_client(MyModel())
    testimg = Image.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    result = client.f(testimg)
    self.assertEqual(result, '(50, 50)')

  def test_str__Image(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str) -> Image:
        return Image.fromarray(np.ones([10, 10, 3], dtype="uint8"))

    pprint(MyModel._get_method_info('f').signature)
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a').to_pil()
    self.assertEqual(result.size, (10, 10))

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

    # test call
    client = _get_servicer_client(MyModel())
    testimg = Image.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    result = client.f(testimg)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].name, 'a')
    self.assertTrue(np.allclose(result[0].value, 0.9))
    self.assertEqual(result[1].name, 'b')
    self.assertTrue(np.allclose(result[1].value, 0.1))

  def test_str_ListImage__str_ListImage(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, prompt: str, images: List[Image]) -> (str, List[Image]):
        return (prompt + ' result', [ImageOps.invert(img) for img in images])

    pprint(MyModel._get_method_info('f').signature)
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

    # test call
    client = _get_servicer_client(MyModel())
    testimg1 = Image.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    testimg2 = Image.fromarray(200 + np.zeros([50, 50, 3], dtype="uint8"))
    result = client.f('prompt', [testimg1, testimg2])
    assert len(result) == 2
    (result_prompt, result_images) = result
    self.assertEqual(result_prompt, 'prompt result')
    self.assertEqual(len(result_images), 2)
    self.assertTrue(np.all(result_images[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result_images[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

  def test_ndarrayint__ndarrayfloat(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: np.ndarray[int]) -> np.ndarray[float]:
        return x / 2.0

    pprint(MyModel._get_method_info('f').signature)
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(np.array([1, 2, 3]))
    self.assertTrue(np.allclose(result, np.array([0.5, 1.0, 1.5])))

  def test_exception_in_predict(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int) -> int:
        raise ValueError('test exception')

    client = _get_servicer_client(MyModel())
    # TODO this raises Exception, not ValueError, because of server-client
    # should this raise common exception types as raised by the server?
    with self.assertRaisesRegex(Exception, 'test exception'):
      client.f(5)

  def test_generate(self):

    class MyModel(ModelClass):

      @methods.generate
      def f(self, x: int) -> Stream[int]:
        for i in range(x):
          yield i

    pprint(MyModel._get_method_info('f').signature)
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
                'generate',
            'name':
                'f',
            'outputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'return',
                'streaming': True
            }]
        })

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(5))
    self.assertEqual(result, [0, 1, 2, 3, 4])

  def test_generate_wrong_return_type(self):

    class MyModel(ModelClass):

      @methods.generate
      def f(self, x: int) -> Stream[str]:
        for i in range(x):
          yield int(i)

    client = _get_servicer_client(MyModel())
    with self.assertRaisesRegex(Exception, 'Incompatible type'), self.assertLogs(level='ERROR'):
      list(client.f(5))

  def test_generate_exception(self):

    class MyModel(ModelClass):

      @methods.generate
      def f(self, x: int) -> Stream[int]:
        for i in range(x):
          if i == 3:
            raise ValueError('test exception')
          yield i

    client = _get_servicer_client(MyModel())
    with self.assertRaisesRegex(Exception, 'test exception'):
      list(client.f(5))

  def test_generate_not_streaming(self):

    with self.assertRaisesRegex(TypeError, 'Generate methods must return a stream'):

      class MyModel(ModelClass):

        @methods.generate
        def f(self, x: int) -> int:
          return x

  def test_two_predict_functions(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int) -> int:
        return x + 1

      @methods.predict
      def g(self, x: str) -> int:
        return len(x)

    assert len(MyModel._get_method_info()) == 2
    assert MyModel._get_method_info().keys() == {'f', 'g'}

    pprint(MyModel._get_method_info('f').signature)
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

    pprint(MyModel._get_method_info('g').signature)
    self.assertEqual(
        MyModel._get_method_info('g').signature, {
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
                'g',
            'outputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'return',
                'streaming': False
            }]
        })

    # test calls
    client = _get_servicer_client(MyModel())
    result = client.f(5)
    self.assertEqual(result, 6)
    result = client.g('abc')
    self.assertEqual(result, 3)

  def test_named_outputs(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, input: str) -> Output(x=int, y=str):
        return Output(x=len(input), y=input + ' result')

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'input',
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
                'name': 'x',
                'streaming': False
            }, {
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'y',
                'streaming': False
            }]
        })

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('abc')
    self.assertEqual(result.x, 3)
    self.assertEqual(result.y, 'abc result')

  def test_kwarg_defaults_int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: int = 5) -> int:
        return x + 1

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'x',
                'default': 5,
                'required': False,
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
    result = client.f()
    self.assertEqual(result, 6)
    result = client.f(0)
    self.assertEqual(result, 1)
    result = client.f(-1)
    self.assertEqual(result, 0)
    result = client.f(10)
    self.assertEqual(result, 11)
    with self.assertRaises(TypeError):
      client.f('abc')
    with self.assertRaises(TypeError):
      client.f(4, 5)

  def test_kwarg_defaults_str(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str = 'abc') -> str:
        return x[::-1]

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'x',
                'default': 'abc',
                'required': False,
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f()
    self.assertEqual(result, 'cba')
    result = client.f('xyz')
    self.assertEqual(result, 'zyx')
    result = client.f('')
    self.assertEqual(result, '')
    with self.assertRaises(TypeError):
      client.f(5)
    with self.assertRaises(TypeError):
      client.f('abc', 'def')
    result = client.f(x='abc')
    self.assertEqual(result, 'cba')
    result = client.f(x='xyz')
    self.assertEqual(result, 'zyx')
    result = client.f(x='')
    self.assertEqual(result, '')
    with self.assertRaises(TypeError):
      client.f(x=5)
    with self.assertRaises(TypeError):
      client.f(x='abc', y='def')
    with self.assertRaises(TypeError):
      client.f(y='abc')
    with self.assertRaises(TypeError):
      client.f('abc', x='def')

  def test_kwarg_defaults_str_int(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: str = 'abc', y: int = 5) -> str:
        return x + str(y)

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'x',
                'default': 'abc',
                'required': False,
                'streaming': False
            }, {
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'y',
                'default': 5,
                'required': False,
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

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f()
    self.assertEqual(result, 'abc5')
    result = client.f('xyz')
    self.assertEqual(result, 'xyz5')
    result = client.f(y=10)
    self.assertEqual(result, 'abc10')
    result = client.f('xyz', 10)
    self.assertEqual(result, 'xyz10')
    with self.assertRaises(TypeError):
      client.f(5)
    with self.assertRaises(TypeError):
      client.f('abc', 5, 'def')
    result = client.f(x='abc')
    self.assertEqual(result, 'abc5')
    result = client.f(x='xyz')
    self.assertEqual(result, 'xyz5')
    result = client.f(y=10)
    self.assertEqual(result, 'abc10')
    result = client.f(x='xyz', y=10)
    self.assertEqual(result, 'xyz10')
    with self.assertRaises(TypeError):
      client.f(x=5)
    with self.assertRaises(TypeError):
      client.f('abc', x=5)
    with self.assertRaises(TypeError):
      client.f('abc', y='def')
    with self.assertRaises(TypeError):
      client.f(y='abc')
    with self.assertRaises(TypeError):
      client.f('abc', x='def')
    with self.assertRaises(TypeError):
      client.f('abc', y=5, x='def')

  def test_kwarg_defaults_ndarray(self):

    class MyModel(ModelClass):

      @methods.predict
      def f(self, x: np.ndarray = np.array([1, 2, 3])) -> np.ndarray:
        return x * 2

    pprint(MyModel._get_method_info('f').signature)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f()
    self.assertTrue(np.all(result == np.array([2, 4, 6])))
    result = client.f(np.array([10, 20, 30]))
    self.assertTrue(np.all(result == np.array([20, 40, 60])))
    result = client.f(5)
    self.assertTrue(np.all(result == np.array(10)))
    with self.assertRaises(TypeError):
      client.f(np.array([1, 2, 3]), np.array([4, 5, 6]))
    result = client.f(x=np.array([1, 2, 3]))
    self.assertTrue(np.all(result == np.array([2, 4, 6])))
    result = client.f(x=np.array([10, 20, 30]))
    self.assertTrue(np.all(result == np.array([20, 40, 60])))
    with self.assertRaises(TypeError):
      client.f(np.array([1, 2, 3]), x=np.array([4, 5, 6]))
    with self.assertRaises(TypeError):
      client.f(y=np.array([1, 2, 3]))
    with self.assertRaises(TypeError):
      client.f(np.array([1, 2, 3]), y=np.array([4, 5, 6]))

  def test_stream_str__str(self):

    class MyModel(ModelClass):

      @methods.stream
      def f(self, input: Stream[str]) -> Stream[str]:
        for i, x in enumerate(input):
          yield str(i) + x

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'input',
                'required': True,
                'streaming': True
            }],
            'method_type':
                'stream',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': True
            }]
        })

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz'])))
    self.assertEqual(result, ['0abc', '1xyz'])

  def test_stream_str_nonstream_int__str(self):

    class MyModel(ModelClass):

      @methods.stream
      def f(self, input_stream: Stream[str], y: int) -> Stream[str]:
        for i, x in enumerate(input_stream):
          yield str(i) + x + str(y)

    pprint(MyModel._get_method_info('f').signature)
    self.assertEqual(
        MyModel._get_method_info('f').signature, {
            'inputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'input_stream',
                'required': True,
                'streaming': True
            }, {
                'data_field': 'int_value',
                'data_type': 'int',
                'name': 'y',
                'required': True,
                'streaming': False
            }],
            'method_type':
                'stream',
            'name':
                'f',
            'outputs': [{
                'data_field': 'string_value',
                'data_type': 'str',
                'name': 'return',
                'streaming': True
            }]
        })

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz']), 5))
    self.assertEqual(result, ['0abc5', '1xyz5'])


#  @unittest.skip("Not implemented yet")
#  def test_stream_named_inputs_str_str__str(self):
#
#    class MyModel(ModelClass):
#
#      @methods.stream
#      def f(self, stream: Stream[Input(x=str, y=str)]) -> Stream[str]:
#        for i, input in enumerate(stream):
#          yield str(i) + input.x + input.y
#
#    pprint(MyModel._get_method_info('f').signature)
#    self.assertEqual(
#        MyModel._get_method_info('f').signature, {
#            'inputs': [{
#                'data_field': 'parts[x].string_value',
#                'data_type': 'str',
#                'name': 'x',
#                'required': True,
#                'streaming': True
#            }, {
#                'data_field': 'parts[y].string_value',
#                'data_type': 'str',
#                'name': 'y',
#                'required': True,
#                'streaming': True
#            }],
#            'method_type':
#                'stream',
#            'name':
#                'f',
#            'outputs': [{
#                'data_field': 'string_value',
#                'data_type': 'str',
#                'name': 'return',
#                'streaming': True
#            }]
#        })


def _get_servicer_client(model):
  servicer = ModelServicer(model)
  client = ModelClient(servicer)
  return client


if __name__ == '__main__':
  unittest.main()
