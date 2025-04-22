import functools
import os
import sys
import unittest
from typing import Dict, Iterator, List, Tuple

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Frame as FrameProto
from clarifai_grpc.grpc.api.resources_pb2 import Region as RegionProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage
from PIL import ImageOps

from clarifai.client.model_client import ModelClient
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.runners.utils.data_types import (Audio, Concept, Frame, Image, NamedFields, Region,
                                               Text, Video)
from clarifai.runners.utils.method_signatures import deserialize, serialize

_ENABLE_PPRINT = os.getenv("PRINT", "false").lower() in ("true", "1")
_ENABLE_PDB = os.getenv("PDB", "false").lower() in ("true", "1")
_USE_SERVER = os.getenv("USE_SERVER", "false").lower() in ("true", "1")

if _ENABLE_PPRINT:
  from pprint import pprint
else:

  def pprint(*args, **kwargs):
    pass


def pdb_on_exception(f, enabled=_ENABLE_PDB):
  import pdb
  import traceback
  if not enabled:
    return f

  if isinstance(f, type):
    for name, method in f.__dict__.items():
      if callable(method):
        setattr(f, name, pdb_on_exception(method))
    return f

  @functools.wraps(f)
  def decorated(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except:
      traceback.print_exc()
      pdb.post_mortem()
      raise

  return decorated


@pdb_on_exception
class TestModelCalls(unittest.TestCase):

  def setUpClass():
    if sys.version_info < (3, 9):
      raise unittest.SkipTest(
          "python <= 3.8 typing support is too limited for model function signatures")

  def test_int__int(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return 2 * x

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(5)
    self.assertEqual(result, 10)

  def test_str_str__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str, y: str) -> str:
        return x + y

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

      @ModelClass.method
      def f(self, x: str, y: int) -> str:
        return x + str(y)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a', 5)
    self.assertEqual(result, 'a5')

  def test_ndarray__int(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: np.ndarray) -> int:
        return int(np.sum(x))

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(np.array([1, 2, 3]))
    self.assertEqual(result, 6)

  def test_Image__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: PILImage.Image) -> str:
        return str(x.size)

    # test call
    client = _get_servicer_client(MyModel())
    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    result = client.f(testimg)
    self.assertEqual(result, '(50, 50)')

  def test_str__Image(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str) -> PILImage.Image:
        return PILImage.fromarray(np.ones([10, 10, 3], dtype="uint8"))

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a').to_pil()
    self.assertEqual(result.size, (10, 10))

  def test_Image__ListConcept(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: PILImage.Image) -> List[Concept]:
        return [Concept('a', 0.9), Concept('b', 0.1)]

    # test call
    client = _get_servicer_client(MyModel())
    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    result = client.f(testimg)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].name, 'a')
    self.assertTrue(np.allclose(result[0].value, 0.9))
    self.assertEqual(result[1].name, 'b')
    self.assertTrue(np.allclose(result[1].value, 0.1))

  def test_str_ListImage__str_ListImage(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, prompt: str, images: List[PILImage.Image]) -> (str, List[PILImage.Image]):
        return (prompt + ' result', [ImageOps.invert(img) for img in images])

      @ModelClass.method
      def g(self, prompt: str, images: List[Image]) -> (str, List[Image]):
        return (prompt + ' result', [ImageOps.invert(img.to_pil()) for img in images])

    # test call
    client = _get_servicer_client(MyModel())
    testimg1 = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    testimg2 = PILImage.fromarray(200 + np.zeros([50, 50, 3], dtype="uint8"))
    result = client.f('prompt', [testimg1, testimg2])
    assert len(result) == 2
    (result_prompt, result_images) = result
    self.assertEqual(result_prompt, 'prompt result')
    self.assertEqual(len(result_images), 2)
    self.assertTrue(np.all(result_images[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result_images[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

    result = client.g('prompt', [Image.from_pil(testimg1), Image.from_pil(testimg2)])
    assert len(result) == 2
    (result_prompt, result_images) = result
    self.assertEqual(result_prompt, 'prompt result')
    self.assertEqual(len(result_images), 2)
    self.assertTrue(np.all(result_images[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result_images[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

  def test_ndarrayint__ndarrayfloat(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: np.ndarray[int]) -> np.ndarray[float]:
        return x / 2.0

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(np.array([1, 2, 3]))
    self.assertTrue(np.allclose(result, np.array([0.5, 1.0, 1.5])))

  def test_exception_in_predict(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        raise ValueError('test exception')

    client = _get_servicer_client(MyModel())
    # TODO this raises Exception, not ValueError, because of server-client
    # should this raise common exception types as raised by the server?
    with self.assertRaisesRegex(Exception, 'test exception'), self.assertLogs(level='ERROR'):
      client.f(5)

  def test_generate(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Iterator[int]:
        for i in range(x):
          yield i

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(5))
    self.assertEqual(result, [0, 1, 2, 3, 4])

  def test_generate_wrong_return_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Iterator[str]:
        for i in range(x):
          yield int(i)

    client = _get_servicer_client(MyModel())
    with self.assertRaisesRegex(Exception, 'Incompatible type'), self.assertLogs(level='ERROR'):
      list(client.f(5))

  def test_generate_exception(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Iterator[int]:
        for i in range(x):
          if i == 3:
            raise ValueError('test exception')
          yield i

    client = _get_servicer_client(MyModel())
    with self.assertRaisesRegex(Exception, 'test exception'), self.assertLogs(level='ERROR'):
      list(client.f(5))

  def test_call_predict_with_generator(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return x

    client = _get_servicer_client(MyModel())
    with self.assertRaises(TypeError):
      client.f(range(5))

  def test_two_predict_functions(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return x + 1

      @ModelClass.method
      def g(self, x: str) -> int:
        return len(x)

    assert len(MyModel._get_method_info()) == 2
    assert MyModel._get_method_info().keys() == {'f', 'g'}

    # test calls
    client = _get_servicer_client(MyModel())
    result = client.f(5)
    self.assertEqual(result, 6)
    result = client.g('abc')
    self.assertEqual(result, 3)

  def test_named_outputs(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, input: str) -> NamedFields(x=int, y=str):
        return NamedFields(x=len(input), y=input + ' result')

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('abc')
    self.assertEqual(result.x, 3)
    self.assertEqual(result.y, 'abc result')

  def test_generate_named_outputs(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Iterator[NamedFields(x=int, y=str)]:
        for i in range(x):
          yield NamedFields(x=i, y=str(i))

    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(5))
    self.assertEqual(len(result), 5)
    for i, output in enumerate(result):
      self.assertEqual(output.x, i)
      self.assertEqual(output.y, str(i))

  def test_kwarg_defaults_int(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int = 5) -> int:
        return x + 1

    MyModel._get_method_info('f').signature

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

      @ModelClass.method
      def f(self, x: str = 'abc') -> str:
        return x[::-1]

    MyModel._get_method_info('f').signature

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

      @ModelClass.method
      def f(self, x: str = 'abc', y: int = 5) -> str:
        return x + str(y)

    MyModel._get_method_info('f').signature
    #self.assertEqual(sig)

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

      @ModelClass.method
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

      @ModelClass.method
      def f(self, input: Iterator[str]) -> Iterator[str]:
        for i, x in enumerate(input):
          yield str(i) + x

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz'])))
    self.assertEqual(result, ['0abc', '1xyz'])

  def test_stream_str_nonstream_int__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, input_stream: Iterator[str], y: int) -> Iterator[str]:
        for i, x in enumerate(input_stream):
          yield str(i) + x + str(y)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz']), 5))
    self.assertEqual(result, ['0abc5', '1xyz5'])

  def test_stream_named_inputs_str_str__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, stream: Iterator[NamedFields(x=str, y=str)]) -> Iterator[str]:
        for i, input in enumerate(stream):
          yield str(i) + input.x + input.y

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(stream=iter([NamedFields(x='a', y='b'), NamedFields(x='x', y='y')])))
    self.assertEqual(result, ['0ab', '1xy'])

  def test_stream_names_nonunique_nested(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, streamvar: Iterator[NamedFields(x=str, y=int)], x: str) -> Iterator[str]:
        for i, val in enumerate(streamvar):
          yield str(i) + val.x + str(val.y) + x

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter([NamedFields(x='a', y=1), NamedFields(x='x', y=2)]), 'z'))
    self.assertEqual(result, ['0a1z', '1x2z'])

  def test_docstrings(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        """This is a test function."""
        return x + 1

      @ModelClass.method
      def g(self, x: str) -> str:
        """This is another test function."""
        return x + 'a'

      @ModelClass.method
      def generate(self, x: str) -> Iterator[int]:
        """This is a generate test function."""
        return range(len(x))

      @ModelClass.method
      def stream(self, stream: Iterator[NamedFields(x=str, y=str)],
                 n: int) -> Iterator[NamedFields(xout=str, yout=str)]:
        """This is a stream test function."""
        for i, input in enumerate(stream):
          yield NamedFields(xout=input.x + str(i), yout=input.y + str(n))

    pprint(MyModel._get_method_info())

    self.assertEqual(
        MyModel._get_method_info('f').signature.description, 'This is a test function.')
    self.assertEqual(
        MyModel._get_method_info('g').signature.description, 'This is another test function.')
    self.assertEqual(
        MyModel._get_method_info('generate').signature.description,
        'This is a generate test function.')
    self.assertEqual(
        MyModel._get_method_info('stream').signature.description,
        'This is a stream test function.')

    client = _get_servicer_client(MyModel())
    self.assertEqual(client.f.__doc__, 'This is a test function.')
    self.assertEqual(client.g.__doc__, 'This is another test function.')
    self.assertEqual(client.generate.__doc__, 'This is a generate test function.')
    self.assertEqual(client.stream.__doc__, 'This is a stream test function.')

    # TODO: Uncomment this once we have a way to get the signature from the client
    # import inspect
    # sig = inspect.signature(client.f)
    # # strip out quotes, since the transfered annos are strings
    # sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    # self.assertEqual(str(sig), '(x: int) -> int'.replace(' ', ''))

    # sig = inspect.signature(client.g)
    # sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    # self.assertEqual(str(sig), '(x: str) -> str'.replace(' ', ''))

    # sig = inspect.signature(client.generate)
    # sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    # self.assertEqual(str(sig), '(x: str) -> Iterator[int]'.replace(' ', ''))

    # sig = inspect.signature(client.stream)
    # sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    # self.assertEqual(
    #     str(sig),
    #     '(stream:Iterator[NamedFields(x=str,y=str)],n:int)->Iterator[NamedFields(xout=str,yout=str)]'.
    #     replace(' ', ''))

  def test_nonexistent_function(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return x + 1

    client = _get_servicer_client(MyModel())

    with self.assertRaises(AttributeError):
      client.g

    result = client.f(5)
    self.assertEqual(result, 6)

    with self.assertRaises(AttributeError):
      client.g

    result = client.f(10)
    self.assertEqual(result, 11)

  def test_nonexistent_function_with_docstring(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        """This is a test function."""
        return x + 1

    client = _get_servicer_client(MyModel())

    with self.assertRaises(AttributeError):
      client.g

    self.assertEqual(client.f.__doc__, MyModel.f.__doc__)

    result = client.f(5)
    self.assertEqual(result, 6)

  def test_int_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return x + 1

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(5), 6)

    self.assertEqual(client.f(0), 1)

    with self.assertRaises(TypeError):
      client.f(float(5.0))

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3, 4)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=5)

  def test_float_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: float) -> float:
        return x + 1.0

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(5), 6.0)
    self.assertEqual(type(client.f(5)), float)

    self.assertEqual(client.f(0.0), 1.0)
    self.assertEqual(client.f(0), 1.0)

    self.assertEqual(client.f(5.5), 6.5)

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3.0, 4.0)

    with self.assertRaises(TypeError):
      client.f(3.0, x=4.0)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=5.0)

  def test_str_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str) -> str:
        return x + '1'

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f('5'), '51')
    self.assertEqual(client.f(''), '1')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f(3.0)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y='5')

  def test_return_empty_string(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f_str(self, x: str) -> str:
        return ''

      @ModelClass.method
      def f_bytes(self, x: bytes) -> bytes:
        return b''

      @ModelClass.method
      def f_string_return_bytes(self, x: str) -> str:
        return b''

      @ModelClass.method
      def f_bytes_return_string(self, x: bytes) -> bytes:
        return ''

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f_str(''), '')
    self.assertEqual(client.f_bytes(b''), b'')

    self.assertEqual(client.f_str('5'), '')
    self.assertEqual(client.f_bytes(b'5'), b'')

    # apparently protobuf allows this and returns the empty string
    #with self.assertRaises(TypeError):
    #  client.f_string_return_bytes('5')

    with self.assertRaises(Exception), self.assertLogs(level='ERROR'):
      client.f_bytes_return_string(b'5')

  def test_bytes_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: bytes) -> bytes:
        return x + b'1'

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(b'5'), b'51')
    self.assertEqual(client.f(b''), b'1')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f(b'3', b'4')

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=b'5')

  def test_bool_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: bool) -> bool:
        return not x

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(True), False)
    self.assertEqual(client.f(False), True)
    self.assertEqual(client.f(1), False)
    self.assertEqual(client.f(0), True)
    self.assertEqual(type(client.f(1)), bool)
    self.assertEqual(type(client.f(0)), bool)

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(True, False)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=True)

  def test_ndarray_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: np.ndarray) -> np.ndarray:
        return x + 1

    client = _get_servicer_client(MyModel())

    # 0d arrays
    self.assertTrue(np.all(client.f(np.array(1)) == np.array(2)))
    self.assertTrue(np.all(client.f(np.array(0)) == np.array(1)))
    self.assertTrue(np.all(client.f(np.array(1.5)) == np.array(2.5)))
    self.assertTrue(np.all(client.f(2.5) == np.array(3.5)))

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(np.array(1), np.array(2))

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=np.array(1))

    # 1d arrays
    self.assertTrue(np.all(client.f(np.array([1, 2, 3])) == np.array([2, 3, 4])))

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(np.array([1, 2, 3]), np.array([4, 5, 6]))

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=np.array([1, 2, 3]))

    # 2d arrays
    self.assertTrue(np.all(client.f(np.array([[1, 2], [3, 4]])) == np.array([[2, 3], [4, 5]])))

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=np.array([[1, 2], [3, 4]]))

    # 1d array, preserve dtype as float32
    res = client.f(np.array([1, 2, 3], dtype=np.float32))
    self.assertTrue(np.all(res == np.array([2, 3, 4], dtype=np.float32)))
    self.assertTrue(res.dtype == np.float32)

  def test_Text_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Text) -> Text:
        return Text(x.text + '1')

    client = _get_servicer_client(MyModel())

    # check value conversions for text
    assert Text('abc') == 'abc'
    assert Text('abc') == Text('abc')
    assert 'abc' == Text('abc')
    assert Text('abc') != 'xyz'
    assert Text('abc') != Text('xyz')
    assert 'abc' != Text('xyz')
    assert Text('') == ''
    assert '' == Text('')

    self.assertEqual(client.f(Text('5')), Text('51'))
    self.assertEqual(client.f(Text('')), Text('1'))
    self.assertEqual(client.f('5'), Text('51'))
    self.assertEqual(client.f(''), Text('1'))

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f(3.0)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y='5')

  def test_Image_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f_pil(self, x: PILImage.Image) -> PILImage.Image:
        return ImageOps.invert(x)

      @ModelClass.method
      def f_datatype(self, x: Image) -> Image:
        return ImageOps.invert(x.to_pil())

    client = _get_servicer_client(MyModel())

    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))

    result = client.f_pil(testimg)
    self.assertEqual(type(result), Image)  # always returns datatype Image to the client
    self.assertTrue(np.all(result.to_numpy() == np.asarray(ImageOps.invert(testimg))))

    with self.assertRaises(TypeError):
      client.f_pil('abc')

    with self.assertRaises(TypeError):
      client.f_pil(3)

    with self.assertRaises(TypeError):
      client.f_pil()

    with self.assertRaises(TypeError):
      client.f_pil(y=testimg)

    result = client.f_datatype(testimg)
    self.assertEqual(type(result), Image)  # always returns datatype Image to the client
    self.assertTrue(np.all(result.to_numpy() == np.asarray(ImageOps.invert(testimg))))

    with self.assertRaises(TypeError):
      client.f_datatype('abc')

    with self.assertRaises(TypeError):
      client.f_datatype(3)

    with self.assertRaises(TypeError):
      client.f_datatype()

    with self.assertRaises(TypeError):
      client.f_datatype(y=testimg)

    def test_Image_type_stream(self):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: Iterator[PILImage.Image]) -> Iterator[PILImage.Image]:
          for i, img in enumerate(x):
            yield ImageOps.invert(img)

      client = _get_servicer_client(MyModel())

      testimg1 = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
      testimg2 = PILImage.fromarray(200 + np.zeros([50, 50, 3], dtype="uint8"))

      result = list(client.f(iter([testimg1, testimg2])))
      self.assertEqual(len(result), 2)
      self.assertTrue(np.all(result[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
      self.assertTrue(np.all(result[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

      with self.assertRaises(TypeError):
        client.f('abc')

      with self.assertRaises(TypeError):
        client.f(3)

      with self.assertRaises(TypeError):
        client.f()

      with self.assertRaises(TypeError):
        client.f(y=testimg1)

    def test_Concept_type(self):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: Concept) -> Concept:
          return x

      client = _get_servicer_client(MyModel())

      testconcept = Concept('testconcept', 0.9)

      result = client.f(testconcept)
      self.assertEqual(result, testconcept)

      with self.assertRaises(TypeError):
        client.f('abc')

      with self.assertRaises(TypeError):
        client.f(3)

      with self.assertRaises(TypeError):
        client.f()

      with self.assertRaises(TypeError):
        client.f(y=testconcept)

  def test_List_int_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[int]) -> List[int]:
        return [i + 1 for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f([1, 2, 3]), [2, 3, 4])
    self.assertEqual(client.f([]), [])
    self.assertEqual(client.f([0]), [1])

  def test_List_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List) -> List:
        return [i + 1 for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f([1, 2, 3]), [2, 3, 4])
    self.assertEqual(client.f([]), [])
    self.assertEqual(client.f([0]), [1])

  def test_List_str_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[str]) -> List[str]:
        return [i + '1' for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(['1', '2', '3']), ['11', '21', '31'])
    self.assertEqual(client.f([]), [])
    self.assertEqual(client.f(['']), ['1'])

  def test_str_List_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str) -> List:
        return [i + '1' for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f('123'), ['11', '21', '31'])
    self.assertEqual(client.f(''), [])

  def test_List_str_type_with_str_param(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[str], y: str) -> List[str]:
        return [xi + y for xi in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(['1', '2', '3'], 'a'), ['1a', '2a', '3a'])
    self.assertEqual(client.f([], 'a'), [])
    self.assertEqual(client.f([''], 'a'), ['a'])

  def test_List_Image_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f_pil(self, x: List[PILImage.Image]) -> List[PILImage.Image]:
        return [ImageOps.invert(i) for i in x]

      @ModelClass.method
      def f_datatype(self, x: List[Image]) -> List[Image]:
        return [ImageOps.invert(i.to_pil()) for i in x]

      @ModelClass.method
      def f_pil_image(self, x: List[PILImage.Image]) -> List[PILImage.Image]:
        return [ImageOps.invert(i) for i in x]

    client = _get_servicer_client(MyModel())

    testimg1 = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    testimg2 = PILImage.fromarray(200 + np.zeros([50, 50, 3], dtype="uint8"))

    result = client.f_pil([testimg1, testimg2])
    self.assertEqual(len(result), 2)
    self.assertTrue(np.all(result[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

    result = client.f_pil_image([testimg1, testimg2])
    self.assertEqual(len(result), 2)
    self.assertTrue(np.all(result[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

    with self.assertRaises(TypeError):
      client.f_pil('abc')

    with self.assertRaises(TypeError):
      client.f_pil(3)

    with self.assertRaises(TypeError):
      client.f_pil()

    with self.assertRaises(TypeError):
      client.f_pil(y=[testimg1, testimg2])

    result = client.f_datatype([testimg1, testimg2])
    self.assertEqual(len(result), 2)
    self.assertTrue(np.all(result[0].to_numpy() == np.asarray(ImageOps.invert(testimg1))))
    self.assertTrue(np.all(result[1].to_numpy() == np.asarray(ImageOps.invert(testimg2))))

    with self.assertRaises(TypeError):
      client.f_datatype('abc')

    with self.assertRaises(TypeError):
      client.f_datatype(3)

    with self.assertRaises(TypeError):
      client.f_datatype()

    with self.assertRaises(TypeError):
      client.f_datatype(y=[testimg1, testimg2])

  def test_ListListNamedFieldsImagestrint_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[List[NamedFields(x=str,
                                           y=int)]]) -> List[List[NamedFields(x=str, y=int)]]:
        return [[NamedFields(x=xi.x + '1', y=xi.y + 1) for xi in xj] for xj in x]

      @ModelClass.method
      def g(self, input: List[List[NamedFields(fld1=str, fld2=int)]]) -> str:
        return ''.join([str(i.fld2) + i.fld1 for j in input for i in j])

    client = _get_servicer_client(MyModel())

    result = client.f([[
        NamedFields(x='a', y=1),
        NamedFields(x='b', y=2),
    ], [
        NamedFields(x='x', y=3),
        NamedFields(x='y', y=4),
    ]])
    self.assertEqual(len(result), 2)
    self.assertEqual(len(result[0]), 2)
    self.assertEqual(len(result[1]), 2)
    self.assertEqual(result[0][0].x, 'a1')
    self.assertEqual(result[0][0].y, 2)
    self.assertEqual(result[0][1].x, 'b1')
    self.assertEqual(result[0][1].y, 3)
    self.assertEqual(result[1][0].x, 'x1')
    self.assertEqual(result[1][0].y, 4)
    self.assertEqual(result[1][1].x, 'y1')
    self.assertEqual(result[1][1].y, 5)

    result = client.g([[NamedFields(fld1='a', fld2=1),
                        NamedFields(fld1='b', fld2=2)],
                       [NamedFields(fld1='x', fld2=3),
                        NamedFields(fld1='y', fld2=4)]])
    self.assertEqual(result, '1a2b3x4y')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=[[
          NamedFields(x='a', y=1),
      ]])

    with self.assertRaises(TypeError):
      client.f([[
          NamedFields(x='a', y='stringvalue'),
      ]])

  def test_NamedFields_List_str_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: NamedFields(x=List[str], y=str)) -> NamedFields(x=List[str], y=str):
        return NamedFields(x=[xi + '1' for xi in x.x], y=x.y + '1')

    client = _get_servicer_client(MyModel())

    result = client.f(NamedFields(x=['1', '2', '3'], y='a'))
    self.assertEqual(result.x, ['11', '21', '31'])
    self.assertEqual(result.y, 'a1')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=NamedFields(x=['1', '2', '3'], y='a'))

    with self.assertRaises(TypeError):
      client.f(NamedFields(x=['1', '2', '3'], y=3))

  def test_untyped_list_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List) -> List:
        return [i + 1 for i in x]

      @ModelClass.method
      def g(self, x: list) -> list:
        return [i + 1 for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f([1, 2, 3]), [2, 3, 4])
    self.assertEqual(client.f([]), [])
    self.assertEqual(client.f([0]), [1])

    self.assertEqual(client.g([1, 2, 3]), [2, 3, 4])
    self.assertEqual(client.g([]), [])
    self.assertEqual(client.g([0]), [1])

  def test_Dict_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Dict[str, int]) -> Dict[str, int]:
        return {k: v + 1 for k, v in x.items()}

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f({'a': 1, 'b': 2, 'c': 3}), {'a': 2, 'b': 3, 'c': 4})
    self.assertEqual(client.f({}), {})
    self.assertEqual(client.f({'a': 0}), {'a': 1})

  def test_untyped_dict_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Dict) -> Dict:
        return {k: v + 1 for k, v in x.items()}

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f({'a': 1, 'b': 2, 'c': 3}), {'a': 2, 'b': 3, 'c': 4})
    self.assertEqual(client.f({}), {})
    self.assertEqual(client.f({'a': 0}), {'a': 1})

  def test_untyped_NamedFields_type_error(self):

    with self.assertRaisesRegex(TypeError, "NamedFields must have types specified"):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: NamedFields) -> NamedFields:
          return NamedFields(x=x.x + 1, y=x.y + 1)

  def test_pilimage_pilimageclass_type_error(self):

    with self.assertRaisesRegex(TypeError, "Use PIL.Image.Image instead of PIL.Image module"):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: PILImage) -> PILImage.Image:
          return ImageOps.invert(x)

  def test_pilimageclass_pilimage_type_error(self):

    with self.assertRaisesRegex(TypeError, "Use PIL.Image.Image instead of PIL.Image module"):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: PILImage.Image) -> PILImage:
          return ImageOps.invert(x)

  def test_pilimage_pilimage_type__error(self):

    with self.assertRaisesRegex(TypeError, "Use PIL.Image.Image instead of PIL.Image module"):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: PILImage) -> PILImage:
          return ImageOps.invert(x)

  def test_untyped_Tuple_type_error(self):

    with self.assertRaisesRegex(TypeError, "Tuple must have types specified"):

      class MyModel(ModelClass):

        @ModelClass.method
        def f(self, x: Tuple) -> Tuple:
          return x[0] + 1, x[1] + 1

  def test_complex_dict_values(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Dict[str, List[int]]) -> Dict[str, List[int]]:
        return {k: [v[0] + 1] for k, v in x.items()}

      @ModelClass.method
      def g(self, x: dict) -> dict:
        return {k: [v[0] + 1] for k, v in x.items()}

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f({'a': [1, 2, 3], 'b': [4, 5, 6]}), {'a': [2], 'b': [5]})
    self.assertEqual(client.f({}), {})
    self.assertEqual(client.f({'a': [0]}), {'a': [1]})

    self.assertEqual(client.g({'a': [1, 2, 3], 'b': [4, 5, 6]}), {'a': [2], 'b': [5]})
    self.assertEqual(client.g({}), {})
    self.assertEqual(client.g({'a': [0]}), {'a': [1]})

  def test_tuple_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Tuple[int, str]) -> Tuple[int, str]:
        return x[0] + 1, x[1] + '1'

      @ModelClass.method
      def g(self, x: (int, str)) -> (int, str):
        return x[0] + 1, x[1] + '1'

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f((1, 'a')), (2, 'a1'))
    self.assertEqual(client.f((0, '')), (1, '1'))

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=(1, 'a'))

    with self.assertRaises(TypeError):
      client.f((1, 2))

    self.assertEqual(client.g((1, 'a')), (2, 'a1'))
    self.assertEqual(client.g((0, '')), (1, '1'))

    with self.assertRaises(TypeError):
      client.g('abc')

  def test_Region_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Region) -> Region:
        return x

    client = _get_servicer_client(MyModel())

    region_proto = RegionProto()
    region_proto.region_info.bounding_box.top_row = 0.1
    region_proto.region_info.bounding_box.left_col = 0.2
    region_proto.region_info.bounding_box.bottom_row = 0.3
    region_proto.region_info.bounding_box.right_col = 0.4

    test_region = Region(region_proto)

    result = client.f(test_region)
    expected_boxes = [0.2, 0.1, 0.4, 0.3]
    for i in range(4):
      self.assertAlmostEqual(result.box[i], expected_boxes[i], places=5)

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=test_region)

  def test_Audio_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Audio) -> Audio:
        return x

    client = _get_servicer_client(MyModel())

    test_audio = Audio(AudioProto(url='https://samples.clarifai.com/GoodMorning.wav'))

    result = client.f(test_audio)
    self.assertEqual(result.url, 'https://samples.clarifai.com/GoodMorning.wav')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=test_audio)

  def test_Frame_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Frame) -> Frame:
        return x

    client = _get_servicer_client(MyModel())

    frame_proto = FrameProto()
    frame_proto.frame_info.time = 1000
    frame_proto.data.image.url = 'https://samples.clarifai.com/metro-north.jpg'

    test_frame = Frame(frame_proto)

    result = client.f(test_frame)
    self.assertEqual(result.time, 1.0)
    self.assertEqual(result.image.url, 'https://samples.clarifai.com/metro-north.jpg')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=test_frame)

  def test_Video_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Video) -> Video:
        return x

    client = _get_servicer_client(MyModel())

    test_video = Video(VideoProto(url='https://samples.clarifai.com/beer.mp4'))

    result = client.f(test_video)
    self.assertEqual(result.url, 'https://samples.clarifai.com/beer.mp4')

    with self.assertRaises(TypeError):
      client.f('abc')

    with self.assertRaises(TypeError):
      client.f(3)

    with self.assertRaises(TypeError):
      client.f()

    with self.assertRaises(TypeError):
      client.f(y=test_video)

  def test_Region_output(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Region:
        region = Region(RegionProto())
        region.box = [0.2, 0.1, 0.4, 0.3]
        return region

    client = _get_servicer_client(MyModel())

    result = client.f(5)
    expected_boxes = [0.2, 0.1, 0.4, 0.3]
    for i in range(4):
      self.assertAlmostEqual(result.box[i], expected_boxes[i], places=5)

  def test_Audio_output(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Audio:
        return Audio(AudioProto(url='https://samples.clarifai.com/GoodMorning.wav'))

    client = _get_servicer_client(MyModel())

    result = client.f(5)
    self.assertEqual(result.url, 'https://samples.clarifai.com/GoodMorning.wav')

  def test_Frame_output(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Frame:
        frame_proto = FrameProto()
        frame_proto.frame_info.time = 1000
        frame_proto.data.image.url = 'https://samples.clarifai.com/metro-north.jpg'
        frame = Frame(frame_proto)
        return frame

    client = _get_servicer_client(MyModel())

    result = client.f(5)
    self.assertEqual(result.time, 1.0)
    self.assertEqual(result.image.url, 'https://samples.clarifai.com/metro-north.jpg')

  def test_Video_output(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Video:
        return Video(VideoProto(url='https://samples.clarifai.com/beer.mp4'))

    client = _get_servicer_client(MyModel())

    result = client.f(5)
    self.assertEqual(result.url, 'https://samples.clarifai.com/beer.mp4')

  def test_List_Region_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Region]) -> List[Region]:
        return x

    client = _get_servicer_client(MyModel())

    region_proto1 = RegionProto()
    region_proto1.region_info.bounding_box.top_row = 0.1
    region_proto1.region_info.bounding_box.left_col = 0.2
    region_proto1.region_info.bounding_box.bottom_row = 0.3
    region_proto1.region_info.bounding_box.right_col = 0.4

    test_region1 = Region(region_proto1)

    region_proto2 = RegionProto()
    region_proto2.region_info.bounding_box.top_row = 0.5
    region_proto2.region_info.bounding_box.left_col = 0.6
    region_proto2.region_info.bounding_box.bottom_row = 0.7
    region_proto2.region_info.bounding_box.right_col = 0.8

    test_region2 = Region(region_proto2)

    result = client.f([test_region1, test_region2])
    self.assertEqual(len(result), 2)
    expected_boxes = [[0.2, 0.1, 0.4, 0.3], [0.6, 0.5, 0.8, 0.7]]
    for i in range(2):
      for j in range(4):
        self.assertAlmostEqual(result[i].box[j], expected_boxes[i][j], places=5)

  def test_List_Audio_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Audio]) -> List[Audio]:
        return x

    client = _get_servicer_client(MyModel())

    test_audio1 = Audio(AudioProto(url='https://samples.clarifai.com/GoodMorning.wav'))
    test_audio2 = Audio(AudioProto(url='https://samples.clarifai.com/GoodMorning.wav'))

    result = client.f([test_audio1, test_audio2])
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].url, 'https://samples.clarifai.com/GoodMorning.wav')
    self.assertEqual(result[1].url, 'https://samples.clarifai.com/GoodMorning.wav')

  def test_List_Frame_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Frame]) -> List[Frame]:
        return x

    client = _get_servicer_client(MyModel())

    frame_proto1 = FrameProto()
    frame_proto1.frame_info.time = 1000
    frame_proto1.data.image.url = 'https://samples.clarifai.com/metro-north.jpg'

    test_frame1 = Frame(frame_proto1)

    frame_proto2 = FrameProto()
    frame_proto2.frame_info.time = 2000
    frame_proto2.data.image.url = 'https://samples.clarifai.com/metro-north.jpg'

    test_frame2 = Frame(frame_proto2)

    result = client.f([test_frame1, test_frame2])
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].time, 1.0)
    self.assertEqual(result[0].image.url, 'https://samples.clarifai.com/metro-north.jpg')
    self.assertEqual(result[1].time, 2.0)
    self.assertEqual(result[1].image.url, 'https://samples.clarifai.com/metro-north.jpg')

  def test_List_Video_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Video]) -> List[Video]:
        return x

    client = _get_servicer_client(MyModel())

    test_video1 = Video(VideoProto(url='https://samples.clarifai.com/beer.mp4'))
    test_video2 = Video(VideoProto(url='https://samples.clarifai.com/beer.mp4'))

    result = client.f([test_video1, test_video2])
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].url, 'https://samples.clarifai.com/beer.mp4')
    self.assertEqual(result[1].url, 'https://samples.clarifai.com/beer.mp4')


class TestSerialization(unittest.TestCase):

  def test_basic_serialize_deserialize(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> int:
        return x + 1

    _get_servicer_client(MyModel())

    signature = MyModel._get_method_info('f').signature

    kwargs = {'x': 5}
    proto = resources_pb2.Data()
    serialize(kwargs, signature.input_fields, proto)
    deserialized_kwargs = deserialize(proto, signature.input_fields)
    self.assertEqual(kwargs, deserialized_kwargs)

  def test_ListConcept_output_is_repeated_field(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Concept]) -> List[Concept]:
        return x

    _get_servicer_client(MyModel())

    signature = MyModel._get_method_info('f').signature

    kwargs = {'x': [Concept('testconcept', 0.9), Concept('testconcept2', 0.8)]}

    proto = resources_pb2.Data()
    serialize(kwargs, signature.input_fields, proto)

    # input list uses parts[x], then repeated concepts
    # also would be ok to put in data.concepts
    self.assertTrue(len(proto.parts) == 1)
    self.assertTrue(proto.parts[0].id == 'x')
    self.assertTrue(len(proto.parts[0].data.concepts) == 2)

    return_value = {'return': kwargs['x']}
    proto = resources_pb2.Data()
    serialize(return_value, signature.output_fields, proto, is_output=True)

    self.assertTrue(len(proto.parts) != 0)

  def test_default_image_first_arg_not_set(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Image = None) -> str:
        return 'a' if x is None else 'b'

    client = _get_servicer_client(MyModel())

    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))

    result = client.f()
    self.assertEqual(result, 'a')

    result = client.f(x=testimg)
    self.assertEqual(result, 'b')


def _get_servicer_client(model):
  servicer = ModelServicer(model)

  if _USE_SERVER:
    from concurrent import futures

    import grpc
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import service_pb2_grpc
    port = 50051
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_V2Servicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    channel = ClarifaiChannel.get_insecure_grpc_channel(base='localhost', port=port)
    stub = service_pb2_grpc.V2Stub(channel)
    client = ModelClient(stub)

    def _stop():
      server.stop(0)
      channel.close()
      server.wait_for_termination()

    client.__del__ = _stop
  else:
    # call the servicer directly
    client = ModelClient(servicer)

  return client


if __name__ == '__main__':
  unittest.main()
