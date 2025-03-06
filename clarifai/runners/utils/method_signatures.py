import ast
import inspect
import json
import textwrap
from collections import namedtuple
from typing import List, Tuple, get_args, get_origin

import numpy as np
import PIL.Image
import yaml
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.message import Message as MessageProto

from clarifai.runners.utils import data_types
from clarifai.runners.utils.serializers import (AtomicFieldSerializer, ListSerializer,
                                                MessageSerializer, NamedFieldsSerializer,
                                                NDArraySerializer, Serializer, TupleSerializer)


def build_function_signature(func):
  '''
  Build a signature for the given function.
  '''
  sig = inspect.signature(func)

  # check if func is bound, and if not, remove self/cls
  if getattr(func, '__self__', None) is None and sig.parameters and list(
      sig.parameters.values())[0].name in ('self', 'cls'):
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])

  return_annotation = sig.return_annotation
  if return_annotation == inspect.Parameter.empty:
    raise TypeError('Function must have a return annotation')

  input_sigs = [
      build_variable_signature(p.name, p.annotation, p.default) for p in sig.parameters.values()
  ]
  input_sigs, input_types, input_streaming = zip(*input_sigs)
  output_sig, output_type, output_streaming = build_variable_signature(
      'return', return_annotation, is_output=True)
  # TODO: flatten out "return" layer if not needed

  # check for streams and determine method type
  if sum(input_streaming) > 1:
    raise TypeError('streaming methods must have at most one streaming input')
  input_streaming = any(input_streaming)
  if not (input_streaming or output_streaming):
    method_type = 'predict'
  elif not input_streaming and output_streaming:
    method_type = 'generate'
  elif input_streaming and output_streaming:
    method_type = 'stream'
  else:
    raise TypeError('stream methods with streaming inputs must have streaming outputs')

  #method_signature = resources_pb2.MethodSignature()   # TODO
  method_signature = _SignatureDict()  #for now

  method_signature.name = func.__name__
  #method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)
  assert method_type in ('predict', 'generate', 'stream')
  method_signature.method_type = method_type
  method_signature.docstring = func.__doc__
  method_signature.annotations_json = json.dumps(_get_annotations_source(func))

  #method_signature.inputs.extend(input_vars)
  #method_signature.outputs.extend(output_vars)
  method_signature.inputs = input_sigs
  method_signature.outputs = output_sig
  return method_signature


def _get_annotations_source(func):
  """Extracts raw annotation strings from the function source."""
  source = inspect.getsource(func)  # Get function source code
  source = textwrap.dedent(source)  # Dedent source code
  tree = ast.parse(source)  # Parse into AST
  func_node = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef))  # Get function node

  annotations = {}
  for arg in func_node.args.args:  # Process arguments
    if arg.annotation:
      annotations[arg.arg] = ast.unparse(arg.annotation)  # Get raw annotation string

  if func_node.returns:  # Process return type
    annotations["return"] = ast.unparse(func_node.returns)

  return annotations


def build_variable_signature(name, annotation, default=inspect.Parameter.empty, is_output=False):
  '''
  Build a data proto signature and get the normalized python type for the given annotation.
  '''

  # check valid names (should already be constrained by python naming, but check anyway)
  if not name.isidentifier():
    raise ValueError(f'Invalid variable name: {name}')

  # get fields for each variable based on type
  tp, streaming = _normalize_type(annotation)

  #var = resources_pb2.VariableSignature()   # TODO
  sig = _VariableSignature()  #for now
  sig.name = name

  _fill_signature_type(sig, tp)

  sig.streaming = streaming

  if not is_output:
    sig.required = (default is inspect.Parameter.empty)
    if not sig.required:
      sig.default = default

  return sig, type, streaming


def _fill_signature_type(sig, tp):
  try:
    if tp in _DATA_TYPES:
      sig.data_type = _DATA_TYPES[tp].data_type
      return
  except TypeError:
    pass  # not hashable type

  if isinstance(tp, data_types.NamedFields):
    sig.data_type = DataType.NAMED_FIELDS
    for name, inner_type in tp.items():
      # inner_sig = sig.type_args.add()
      sig.type_args.append(inner_sig := _VariableSignature())
      inner_sig.name = name
      _fill_signature_type(inner_sig, inner_type)
    return

  if get_origin(tp) == tuple:
    sig.data_type = DataType.TUPLE
    for inner_type in get_args(tp):
      #inner_sig = sig.type_args.add()
      sig.type_args.append(inner_sig := _VariableSignature())
      _fill_signature_type(inner_sig, inner_type)
    return

  if get_origin(tp) == list:
    sig.data_type = DataType.LIST
    inner_type = get_args(tp)[0]
    #inner_sig = sig.type_args.add()
    sig.type_args.append(inner_sig := _VariableSignature())
    _fill_signature_type(inner_sig, inner_type)
    return

  raise TypeError(f'Unsupported type: {tp}')


def serializer_from_signature(signature):
  '''
    Get the serializer for the given signature.
    '''
  if signature.data_type in _SERIALIZERS_BY_TYPE_ENUM:
    return _SERIALIZERS_BY_TYPE_ENUM[signature.data_type]
  if signature.data_type == DataType.LIST:
    return ListSerializer(serializer_from_signature(signature.type_args[0]))
  if signature.data_type == DataType.TUPLE:
    return TupleSerializer([serializer_from_signature(sig) for sig in signature.type_args])
  if signature.data_type == DataType.NAMED_FIELDS:
    return NamedFieldsSerializer(
        {sig.name: serializer_from_signature(sig)
         for sig in signature.type_args})
  raise ValueError(f'Unsupported type: {signature.data_type}')


def signatures_to_json(signatures):
  assert isinstance(
      signatures, dict), 'Expected dict of signatures {name: signature}, got %s' % type(signatures)
  return json.dumps(signatures, default=repr)


def signatures_from_json(json_str):
  d = json.loads(json_str, object_pairs_hook=_SignatureDict)
  return d


def signatures_to_yaml(signatures):
  # XXX go in/out of json to get the correct format and python dict types
  d = json.loads(signatures_to_json(signatures))
  return yaml.dump(d, default_flow_style=False)


def signatures_from_yaml(yaml_str):
  d = yaml.safe_load(yaml_str)
  return signatures_from_json(json.dumps(d))


def serialize(kwargs, signatures, proto=None, is_output=False):
  '''
  Serialize the given kwargs into the proto using the given signatures.
  '''
  if proto is None:
    proto = resources_pb2.Data()
  unknown = set(kwargs.keys()) - set(sig.name for sig in signatures)
  if unknown:
    if unknown == {'return'} and len(signatures) > 1:
      raise TypeError('Got a single return value, but expected multiple outputs {%s}' %
                      ', '.join(sig.name for sig in signatures))
    raise TypeError('Got unexpected key: %s' % ', '.join(unknown))
  for sig in signatures:
    if sig.name not in kwargs:
      if sig.required:
        raise TypeError(f'Missing required argument: {sig.name}')
      continue  # skip missing fields, they can be set to default on the server
    data = kwargs[sig.name]
    serializer = serializer_from_signature(sig)
    # TODO determine if any (esp the first) var can go in the proto without parts
    # and whether to put this in the signature or dynamically determine it
    part = proto.parts.add()
    part.id = sig.name
    serializer.serialize(part.data, data)
  return proto


def deserialize(proto, signatures, is_output=False):
  '''
  Deserialize the given proto into kwargs using the given signatures.
  '''
  if isinstance(signatures, dict):
    signatures = [signatures]  # TODO update return key level and make consistnet
  kwargs = {}
  parts_by_name = {part.id: part for part in proto.parts}
  for sig in signatures:
    serializer = serializer_from_signature(sig)
    part = parts_by_name.get(sig.name)
    if part is None:
      if sig.required or is_output:  # TODO allow optional outputs?
        raise ValueError(f'Missing required field: {sig.name}')
      continue
    kwargs[sig.name] = serializer.deserialize(part.data)
  if len(kwargs) == 1 and 'return' in kwargs:
    return kwargs['return']
  return kwargs


def get_stream_from_signature(signatures):
  '''
  Get the stream signature from the given signatures.
  '''
  for sig in signatures:
    if sig.streaming:
      return sig
  return None


def _is_empty_proto_data(data):
  if isinstance(data, np.ndarray):
    return False
  if isinstance(data, MessageProto):
    return not data.ByteSize()
  return not data


def _normalize_type(tp):
  '''
  Normalize the types for the given parameter.
  Returns the normalized type and whether the parameter is streaming.
  '''
  # stream type indicates streaming, not part of the data itself
  # it can only be used at the top-level of the var type
  streaming = (get_origin(tp) == data_types.Stream)
  if streaming:
    tp = get_args(tp)[0]

  return _normalize_data_type(tp), streaming


def _normalize_data_type(tp):
  # check if list, and if so, get inner type
  if get_origin(tp) == list:
    tp = get_args(tp)[0]
    return List[_normalize_data_type(tp)]

  if isinstance(tp, (tuple, list)):
    return Tuple[tuple(_normalize_data_type(val) for val in tp)]

  if isinstance(tp, (dict, data_types.NamedFields)):
    return data_types.NamedFields(**{name: _normalize_data_type(val) for name, val in tp.items()})

  # check if numpy array type, and if so, use ndarray
  if get_origin(tp) == np.ndarray:
    return np.ndarray

  # check for PIL images (sometimes types use the module, sometimes the class)
  # set these to use the Image data handler
  if tp in (data_types.Image, PIL.Image, PIL.Image.Image):
    return data_types.Image

  # check for jsonable types
  # TODO should we include dict vs list in the data type somehow?
  if tp == dict or (get_origin(tp) == dict and tp not in _DATA_TYPES and _is_jsonable(tp)):
    return data_types.JSON
  if tp == list or (get_origin(tp) == list and tp not in _DATA_TYPES and _is_jsonable(tp)):
    return data_types.JSON

  # check for known data types
  try:
    if tp in _DATA_TYPES:
      return tp
  except TypeError:
    pass  # not hashable type

  raise TypeError(f'Unsupported type: {tp}')


def _is_jsonable(tp):
  if tp in (dict, list, tuple, str, int, float, bool, type(None)):
    return True
  if get_origin(tp) == list:
    return _is_jsonable(get_args(tp)[0])
  if get_origin(tp) == dict:
    return all(_is_jsonable(val) for val in get_args(tp))
  return False


# TODO --- tmp classes to stand-in for protos until they are defined and built into this package
class _SignatureDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


class _VariableSignature(_SignatureDict):

  def __init__(self):
    super().__init__()
    self.name = ''
    self.type = ''
    self.type_args = []
    self.streaming = False
    self.required = False
    self.default = ''
    self.description = ''


# data_type: name of the data type
# data_field: name of the field in the data proto
# serializer: serializer for the data type
_DataType = namedtuple('_DataType', ('data_type', 'serializer'))


# this will come from the proto module, but for now, define it here
class DataType:
  NOT_SET = 'NOT_SET'

  STR = 'STR'
  BYTES = 'BYTES'
  INT = 'INT'
  FLOAT = 'FLOAT'
  BOOL = 'BOOL'
  NDARRAY = 'NDARRAY'
  JSON = 'JSON'

  TEXT = 'TEXT'
  IMAGE = 'IMAGE'
  CONCEPT = 'CONCEPT'
  REGION = 'REGION'
  FRAME = 'FRAME'
  AUDIO = 'AUDIO'
  VIDEO = 'VIDEO'

  NAMED_FIELDS = 'NAMED_FIELDS'
  TUPLE = 'TUPLE'
  LIST = 'LIST'


# simple, non-container types that correspond directly to a data field
_DATA_TYPES = {
    str:
        _DataType(DataType.STR, AtomicFieldSerializer('string_value')),
    bytes:
        _DataType(DataType.BYTES, AtomicFieldSerializer('bytes_value')),
    int:
        _DataType(DataType.INT, AtomicFieldSerializer('int_value')),
    float:
        _DataType(DataType.FLOAT, AtomicFieldSerializer('float_value')),
    bool:
        _DataType(DataType.BOOL, AtomicFieldSerializer('bool_value')),
    np.ndarray:
        _DataType(DataType.NDARRAY, NDArraySerializer('ndarray')),
    data_types.Text:
        _DataType(DataType.TEXT, MessageSerializer('text', data_types.Text)),
    data_types.Image:
        _DataType(DataType.IMAGE, MessageSerializer('image', data_types.Image)),
    data_types.Concept:
        _DataType(DataType.CONCEPT, MessageSerializer('concepts', data_types.Concept)),
    data_types.Region:
        _DataType(DataType.REGION, MessageSerializer('regions', data_types.Region)),
    data_types.Frame:
        _DataType(DataType.FRAME, MessageSerializer('frames', data_types.Frame)),
    data_types.Audio:
        _DataType(DataType.AUDIO, MessageSerializer('audio', data_types.Audio)),
    data_types.Video:
        _DataType(DataType.VIDEO, MessageSerializer('video', data_types.Video)),
}

_SERIALIZERS_BY_TYPE_ENUM = {dt.data_type: dt.serializer for dt in _DATA_TYPES.values()}


class CompatibilitySerializer(Serializer):
  '''
  Serialization of basic value types, used for backwards compatibility
  with older models that don't have type signatures.
  '''

  def serialize(self, data_proto, value):
    tp = _normalize_data_type(type(value))

    try:
      serializer = _DATA_TYPES[tp].serializer
    except (KeyError, TypeError):
      raise TypeError(f'serializer currently only supports basic types, got {tp}')

    serializer.serialize(data_proto, value)

  def deserialize(self, data_proto):
    fields = [k.name for k, _ in data_proto.ListFields()]
    if 'parts' in fields:
      raise ValueError('serializer does not support parts')
    serializers = [
        serializer for serializer in _SERIALIZERS_BY_TYPE_ENUM.values()
        if serializer.field_name in fields
    ]
    if not serializers:
      raise ValueError('Returned data not recognized')
    if len(serializers) != 1:
      raise ValueError('Only single output supported for serializer')
    serializer = serializers[0]
    return serializer.deserialize(data_proto)
