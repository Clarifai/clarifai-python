import functools
import os
import sys
import unittest
from typing import Dict, List, Tuple

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from PIL import Image as PILImage
from PIL import ImageOps

from clarifai.client.model_client import ModelClient
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.runners.utils.data_types import Concept, Image, NamedFields, Stream, Text
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

    #sig = dict(MyModel._get_method_info('f').signature)
    #del sig['docstring']
    ##self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(5)
    self.assertEqual(result, 10)

  def test_str_str__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str, y: str) -> str:
        return x + y

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a', 5)
    self.assertEqual(result, 'a5')

  def test_ndarray__int(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: np.ndarray) -> int:
        return int(np.sum(x))

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f(np.array([1, 2, 3]))
    self.assertEqual(result, 6)

  def test_Image__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: PILImage) -> str:
        return str(x.size)

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    result = client.f(testimg)
    self.assertEqual(result, '(50, 50)')

  def test_str__Image(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: str) -> PILImage:
        return PILImage.fromarray(np.ones([10, 10, 3], dtype="uint8"))

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('a').to_pil()
    self.assertEqual(result.size, (10, 10))

  def test_Image__ListConcept(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: PILImage) -> List[Concept]:
        return [Concept('a', 0.9), Concept('b', 0.1)]

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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
    # skip if python version is below 3.11: can't use List[Image] in signature for <=3.10
    if sys.version_info < (3, 11):
      self.skipTest("python < 3.11 does not support using regular classes in generics")

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, prompt: str, images: List[PILImage]) -> (str, List[PILImage]):
        return (prompt + ' result', [ImageOps.invert(img) for img in images])

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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

  def test_ndarrayint__ndarrayfloat(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: np.ndarray[int]) -> np.ndarray[float]:
        return x / 2.0

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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
      def f(self, x: int) -> Stream[int]:
        for i in range(x):
          yield i

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(5))
    self.assertEqual(result, [0, 1, 2, 3, 4])

  def test_generate_wrong_return_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Stream[str]:
        for i in range(x):
          yield int(i)

    client = _get_servicer_client(MyModel())
    with self.assertRaisesRegex(Exception, 'Incompatible type'), self.assertLogs(level='ERROR'):
      list(client.f(5))

  def test_generate_exception(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Stream[int]:
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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    sig = dict(MyModel._get_method_info('g').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = client.f('abc')
    self.assertEqual(result.x, 3)
    self.assertEqual(result.y, 'abc result')

  def test_generate_named_outputs(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: int) -> Stream[NamedFields(x=int, y=str)]:
        for i in range(x):
          yield NamedFields(x=i, y=str(i))

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
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
      def f(self, input: Stream[str]) -> Stream[str]:
        for i, x in enumerate(input):
          yield str(i) + x

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz'])))
    self.assertEqual(result, ['0abc', '1xyz'])

  def test_stream_str_nonstream_int__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, input_stream: Stream[str], y: int) -> Stream[str]:
        for i, x in enumerate(input_stream):
          yield str(i) + x + str(y)

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(iter(['abc', 'xyz']), 5))
    self.assertEqual(result, ['0abc5', '1xyz5'])

  def test_stream_named_inputs_str_str__str(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, stream: Stream[NamedFields(x=str, y=str)]) -> Stream[str]:
        for i, input in enumerate(stream):
          yield str(i) + input.x + input.y

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

    # test call
    client = _get_servicer_client(MyModel())
    result = list(client.f(stream=iter([NamedFields(x='a', y='b'), NamedFields(x='x', y='y')])))
    self.assertEqual(result, ['0ab', '1xy'])

  def test_stream_names_nonunique_nested(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, streamvar: Stream[NamedFields(x=str, y=int)], x: str) -> Stream[str]:
        for i, val in enumerate(streamvar):
          yield str(i) + val.x + str(val.y) + x

    sig = dict(MyModel._get_method_info('f').signature)
    del sig['docstring']
    #self.assertEqual(sig)

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
      def generate(self, x: str) -> Stream[int]:
        """This is a generate test function."""
        return range(len(x))

      @ModelClass.method
      def stream(self, stream: Stream[NamedFields(x=str, y=str)],
                 n: int) -> Stream[NamedFields(xout=str, yout=str)]:
        """This is a stream test function."""
        for i, input in enumerate(stream):
          yield NamedFields(xout=input.x + str(i), yout=input.y + str(n))

    pprint(MyModel._get_method_info())

    self.assertEqual(MyModel._get_method_info('f').signature.docstring, 'This is a test function.')
    self.assertEqual(
        MyModel._get_method_info('g').signature.docstring, 'This is another test function.')
    self.assertEqual(
        MyModel._get_method_info('generate').signature.docstring,
        'This is a generate test function.')
    self.assertEqual(
        MyModel._get_method_info('stream').signature.docstring, 'This is a stream test function.')

    client = _get_servicer_client(MyModel())
    self.assertEqual(client.f.__doc__, 'This is a test function.')
    self.assertEqual(client.g.__doc__, 'This is another test function.')
    self.assertEqual(client.generate.__doc__, 'This is a generate test function.')
    self.assertEqual(client.stream.__doc__, 'This is a stream test function.')

    import inspect
    sig = inspect.signature(client.f)
    # strip out quotes, since the transfered annos are strings
    sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    self.assertEqual(str(sig), '(x: int) -> int'.replace(' ', ''))

    sig = inspect.signature(client.g)
    sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    self.assertEqual(str(sig), '(x: str) -> str'.replace(' ', ''))

    sig = inspect.signature(client.generate)
    sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    self.assertEqual(str(sig), '(x: str) -> Stream[int]'.replace(' ', ''))

    sig = inspect.signature(client.stream)
    sig = str(sig).replace("'", "").replace('"', '').replace(' ', '')
    self.assertEqual(
        str(sig),
        '(stream:Stream[NamedFields(x=str,y=str)],n:int)->Stream[NamedFields(xout=str,yout=str)]'.
        replace(' ', ''))

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
      def f_pil(self, x: PILImage) -> PILImage:
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
        def f(self, x: Stream[PILImage]) -> Stream[PILImage]:
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

  def test_List_str_type(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[str]) -> List[str]:
        return [i + '1' for i in x]

    client = _get_servicer_client(MyModel())

    self.assertEqual(client.f(['1', '2', '3']), ['11', '21', '31'])
    self.assertEqual(client.f([]), [])
    self.assertEqual(client.f(['']), ['1'])

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
      def f_pil(self, x: List[PILImage]) -> List[PILImage]:
        return [ImageOps.invert(i) for i in x]

      @ModelClass.method
      def f_datatype(self, x: List[Image]) -> List[Image]:
        return [ImageOps.invert(i.to_pil()) for i in x]

    client = _get_servicer_client(MyModel())

    testimg1 = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))
    testimg2 = PILImage.fromarray(200 + np.zeros([50, 50, 3], dtype="uint8"))

    result = client.f_pil([testimg1, testimg2])
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
    serialize(kwargs, signature.inputs, proto)
    deserialized_kwargs = deserialize(proto, signature.inputs)
    self.assertEqual(kwargs, deserialized_kwargs)

  def test_Image_one_input_not_in_parts(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: Image) -> Image:
        return x

    _get_servicer_client(MyModel())

    signature = MyModel._get_method_info('f').signature

    testimg = PILImage.fromarray(np.ones([50, 50, 3], dtype="uint8"))

    kwargs = {'x': testimg}
    proto = resources_pb2.Data()
    serialize(kwargs, signature.inputs, proto)

    self.assertTrue(len(proto.parts) == 0)
    self.assertTrue(proto.HasField('image'))

  def test_ListConcept_output_is_repeated_field(self):

    class MyModel(ModelClass):

      @ModelClass.method
      def f(self, x: List[Concept]) -> List[Concept]:
        return x

    _get_servicer_client(MyModel())

    signature = MyModel._get_method_info('f').signature

    kwargs = {'x': [Concept('testconcept', 0.9), Concept('testconcept2', 0.8)]}

    proto = resources_pb2.Data()
    serialize(kwargs, signature.inputs, proto)

    # input list uses parts[x], then repeated concepts
    # also would be ok to put in data.concepts
    self.assertTrue(len(proto.parts) == 1)
    self.assertTrue(proto.parts[0].id == 'x')
    self.assertTrue(len(proto.parts[0].data.concepts) == 2)

    return_value = {'return': kwargs['x']}
    proto = resources_pb2.Data()
    serialize(return_value, signature.outputs, proto, is_output=True)

    self.assertTrue(len(proto.parts) == 0)
    self.assertTrue(len(proto.concepts) == 2)


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
