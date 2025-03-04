import inspect
import json
import re
import types
from collections import OrderedDict, namedtuple
from typing import List, get_args, get_origin

import numpy as np
import PIL.Image
import yaml
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.message import Message as MessageProto

from clarifai.runners.utils import data_types
from clarifai.runners.utils.serializers import (AtomicFieldSerializer, ImageSerializer,
                                                ListSerializer, MessageSerializer,
                                                NDArraySerializer, NullValueSerializer, Serializer)


def build_function_signature(func, method_type: str):
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
    raise ValueError('Function must have a return annotation')
  # check for multiple return values and convert to dict for named values
  return_streaming = False
  if get_origin(return_annotation) == data_types.Stream:
    return_annotation = get_args(return_annotation)[0]
    return_streaming = True
  if get_origin(return_annotation) == tuple:
    return_annotation = tuple(get_args(return_annotation))
  if isinstance(return_annotation, tuple):
    return_annotation = {'return.%s' % i: tp for i, tp in enumerate(return_annotation)}
  if not isinstance(return_annotation, dict):
    return_annotation = {'return': return_annotation}

  input_vars = build_variables_signature(sig.parameters.values())
  output_vars = build_variables_signature(
      [
          # XXX inspect.Parameter errors for the special return names, so use SimpleNamespace
          types.SimpleNamespace(name=name, annotation=tp, default=inspect.Parameter.empty)
          for name, tp in return_annotation.items()
      ],
      is_output=True)
  if return_streaming:
    for var in output_vars:
      var.streaming = True

  # check for streams
  if method_type == 'predict':
    for var in input_vars:
      if var.streaming:
        raise TypeError('Stream inputs are not supported for predict methods')
    for var in output_vars:
      if var.streaming:
        raise TypeError('Stream outputs are not supported for predict methods')
  elif method_type == 'generate':
    for var in input_vars:
      if var.streaming:
        raise TypeError('Stream inputs are not supported for generate methods')
    if not all(var.streaming for var in output_vars):
      raise TypeError('Generate methods must return a stream')
  elif method_type == 'stream':
    input_stream_vars = [var for var in input_vars if var.streaming]
    if len(input_stream_vars) == 0:
      raise TypeError('Stream methods must include a Stream input')
    if not all(var.streaming for var in output_vars):
      raise TypeError('Stream methods must return a single Stream')
  else:
    raise TypeError('Invalid method type: %s' % method_type)

  #method_signature = resources_pb2.MethodSignature()   # TODO
  method_signature = _NamedFields()  #for now

  method_signature.name = func.__name__
  #method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)
  assert method_type in ('predict', 'generate', 'stream')
  method_signature.method_type = method_type
  method_signature.docstring = func.__doc__

  #method_signature.inputs.extend(input_vars)
  #method_signature.outputs.extend(output_vars)
  method_signature.inputs = input_vars
  method_signature.outputs = output_vars
  return method_signature


def build_variables_signature(parameters: List[inspect.Parameter], is_output=False):
  '''
  Build a data proto signature for the given variable or return type annotation.
  '''

  vars = []

  # check valid names (should already be constrained by python naming, but check anyway)
  for param in parameters:
    if not param.name.isidentifier() and not (is_output and
                                              re.match(r'return(\.\d+)?', param.name)):
      raise ValueError(f'Invalid variable name: {param.name}')

  # get fields for each variable based on type
  for param in parameters:
    param_types, streaming = _normalize_types(param, is_output=is_output)

    for name, tp in param_types.items():
      #var = resources_pb2.MethodVariable()   # TODO
      var = _NamedFields()
      var.name = name
      var.data_type = _DATA_TYPES[tp].data_type
      var.data_field = _DATA_TYPES[tp].data_field
      var.streaming = streaming
      if not is_output:
        var.required = (param.default is inspect.Parameter.empty)
        if not var.required:
          var.default = param.default
      vars.append(var)

  # check if any fields are used more than once, and if so, use parts
  # also if more than one field uses parts lists, also use parts, since the lists can be different lengths
  # NOTE this is a little fancy, another way would just be to check if there is more than one arg
  fields_unique = (len(set(var.data_field for var in vars)) == len(vars))
  num_parts_lists = sum(int(var.data_field.startswith('parts[]')) for var in vars)
  if not fields_unique or num_parts_lists > 1:
    for var in vars:
      var.data_field = 'parts[%s].%s' % (var.name, var.data_field)

  return vars


def signatures_to_json(signatures):
  assert isinstance(
      signatures, dict), 'Expected dict of signatures {name: signature}, got %s' % type(signatures)
  return json.dumps(signatures, default=repr)


def signatures_from_json(json_str):
  return json.loads(json_str, object_pairs_hook=_NamedFields)


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
  if not is_output:  # TODO: use this consistently for return keys also
    kwargs = flatten_nested_keys(kwargs, signatures, is_output)
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
    force_named_part = (_is_empty_proto_data(data) and not is_output and not sig.required)
    data_proto, field = _get_data_part(
        proto, sig, is_output=is_output, serializing=True, force_named_part=force_named_part)
    serializer = get_serializer(sig.data_type)
    serializer.serialize(data_proto, field, data)
  return proto


def deserialize(proto, signatures, is_output=False):
  '''
  Deserialize the given proto into kwargs using the given signatures.
  '''
  kwargs = {}
  for sig in signatures:
    data_proto, field = _get_data_part(proto, sig, is_output=is_output, serializing=False)
    if data_proto is None:
      # not set in proto, check if required or skip if optional arg
      if not is_output and sig.required:
        raise ValueError(f'Missing required field: {sig.name}')
      continue
    serializer = get_serializer(sig.data_type)
    data = serializer.deserialize(data_proto, field)
    kwargs[sig.name] = data
  if is_output:
    if len(kwargs) == 1 and 'return' in kwargs:  # case for single return value
      return kwargs['return']
    if kwargs and 'return.0' in kwargs:  # case for tuple return values
      return tuple(kwargs[f'return.{i}'] for i in range(len(kwargs)))
    return data_types.Output(kwargs)
  kwargs = unflatten_nested_keys(kwargs, signatures, is_output)
  return kwargs


def get_serializer(data_type: str) -> Serializer:
  if data_type in _SERIALIZERS_BY_TYPE_STRING:
    return _SERIALIZERS_BY_TYPE_STRING[data_type]
  if data_type.startswith('List['):
    inner_type_string = data_type[len('List['):-1]
    inner_serializer = get_serializer(inner_type_string)
    return ListSerializer(inner_serializer)
  raise ValueError(f'Unsupported type: "{data_type}"')


def flatten_nested_keys(kwargs, signatures, is_output):
  '''
  Flatten nested keys into a single key with a dot, e.g. {'a': {'b': 1}} -> {'a.b': 1}
  in the kwargs, using the given signatures to determine which keys are nested.
  '''
  nested_keys = [sig.name for sig in signatures if '.' in sig.name]
  outer_keys = set(key.split('.')[0] for key in nested_keys)
  for outer in outer_keys:
    if outer not in kwargs:
      continue
    kwargs.update({outer + '.' + k: v for k, v in kwargs.pop(outer).items()})
  return kwargs


def unflatten_nested_keys(kwargs, signatures, is_output):
  '''
  Unflatten nested keys in kwargs into a dict, e.g. {'a.b': 1} -> {'a': {'b': 1}}
  Uses the signatures to determine which keys are nested.
  The dict subclass is Input or Output, depending on the is_output flag.
  Preserves the order of args from the signatures.
  '''
  unflattened = OrderedDict()
  for sig in signatures:
    if '.' not in sig.name:
      if sig.name in kwargs:
        unflattened[sig.name] = kwargs[sig.name]
      continue
    if sig.name not in kwargs:
      continue
    parts = sig.name.split('.')
    assert len(parts) == 2, 'Only one level of nested keys is supported'
    if parts[0] not in unflattened:
      unflattened[parts[0]] = data_types.Output() if is_output else data_types.Input()
    unflattened[parts[0]][parts[1]] = kwargs[sig.name]
  return unflattened


def get_stream_from_signature(signatures):
  streaming_signatures = [var for var in signatures if var.streaming]
  if not streaming_signatures:
    return None, []
  stream_argname = set([var.name.split('.', 1)[0] for var in streaming_signatures])
  assert len(stream_argname) == 1, 'streaming methods must have exactly one streaming function arg'
  stream_argname = stream_argname.pop()
  return stream_argname, streaming_signatures


def _is_empty_proto_data(data):
  if isinstance(data, np.ndarray):
    return False
  if isinstance(data, MessageProto):
    return not data.ByteSize()
  return not data


def _get_data_part(proto, sig, is_output, serializing, force_named_part=False):
  field = sig.data_field

  # check if we need to force a named part, to distinguish between empty and unset values
  if force_named_part and not field.startswith('parts['):
    field = f'parts[{sig.name}].{field}'

  # gets the named part from the proto, according to the field path
  # note we only support one level of named parts
  #parts = field.replace(' ', '').split('.')
  # split on . but not if it is inside brackets, e.g. parts[outer.inner].field
  parts = re.split(r'\.(?![^\[]*\])', field.replace(' ', ''))

  if len(parts) not in (1, 2, 3):  # field, parts[name].field, parts[name].parts[].field
    raise ValueError('Invalid field: %s' % field)

  if len(parts) == 1:
    # also need to check if there is an explicitly named part, e.g. for empty values
    part = next((part for part in proto.parts if part.id == sig.name), None)
    if part:
      return part.data, field
    if not serializing and not is_output and _is_empty_proto_data(getattr(proto, field)):
      return None, field
    return proto, field

  # list
  if parts[0] == 'parts[]':
    if len(parts) != 2:
      raise ValueError('Invalid field: %s' % field)
    return proto, field  # return the data that contains the list itself

  # named part
  if not (m := re.match(r'parts\[([\w.]+)\]', parts[0])):
    raise ValueError('Invalid field: %s' % field)
  if not (name := m.group(1)):
    raise ValueError('Invalid field: %s' % field)
  assert len(parts) in (2, 3)  # parts[name].field, parts[name].parts[].field
  part = next((part for part in proto.parts if part.id == name), None)
  if part is None:
    if not serializing:
      raise ValueError('Missing part: %s' % name)
    part = proto.parts.add()
    part.id = name
  return part.data, '.'.join(parts[1:])


def _normalize_types(param, is_output=False):
  '''
  Normalize the types for the given parameter.  Returns a dict of names to types,
  including named return values for outputs, and a flag indicating if streaming is used.
  '''
  tp = param.annotation

  # stream type indicates streaming, not part of the data itself
  streaming = (get_origin(tp) == data_types.Stream)
  if streaming:
    tp = get_args(tp)[0]

  if is_output or streaming:  # named types can be used for outputs or streaming inputs
    # output type used for named return values, each with their own data type
    if isinstance(tp, (dict, data_types.Output, data_types.Input)):
      return {param.name + '.' + name: _normalize_data_type(val)
              for name, val in tp.items()}, streaming
    if tp == data_types.Output:  # check for Output type without values
      if not is_output:
        raise TypeError('Output types can only be used for output values')
      raise TypeError('Output types must be instantiated with inner type values for each key')
    if tp == data_types.Input:  # check for Output type without values
      if is_output:
        raise TypeError('Input types can only be used for input values')
      raise TypeError(
          'Stream[Input(...)] types must be instantiated with inner type values for each key')

  return {param.name: _normalize_data_type(tp)}, streaming


def _normalize_data_type(tp):
  # check if list, and if so, get inner type
  is_list = (get_origin(tp) == list)
  if is_list:
    tp = get_args(tp)[0]

  # check if numpy array, and if so, use ndarray
  if get_origin(tp) == np.ndarray:
    tp = np.ndarray

  # check for PIL images (sometimes types use the module, sometimes the class)
  # set these to use the Image data handler
  if tp in (PIL.Image, PIL.Image.Image):
    tp = data_types.Image

  # put back list
  if is_list:
    tp = List[tp]

  # check if supported type
  if tp not in _DATA_TYPES:
    raise ValueError(f'Unsupported type: {tp}')

  return tp


class _NamedFields(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


# data_type: name of the data type
# data_field: name of the field in the data proto
# serializer: serializer for the data type
_DataType = namedtuple('_DataType', ('data_type', 'data_field', 'serializer'))

# mapping of supported python types to data type names, fields, and serializers
_DATA_TYPES = {
    str:
        _DataType('str', 'string_value', AtomicFieldSerializer()),
    bytes:
        _DataType('bytes', 'bytes_value', AtomicFieldSerializer()),
    int:
        _DataType('int', 'int_value', AtomicFieldSerializer()),
    float:
        _DataType('float', 'float_value', AtomicFieldSerializer()),
    bool:
        _DataType('bool', 'bool_value', AtomicFieldSerializer()),
    None:
        _DataType('None', '', NullValueSerializer()),
    np.ndarray:
        _DataType('ndarray', 'ndarray', NDArraySerializer()),
    data_types.Text:
        _DataType('Text', 'text', MessageSerializer(data_types.Text)),
    data_types.Image:
        _DataType('Image', 'image', ImageSerializer()),
    data_types.Concept:
        _DataType('Concept', 'concepts', MessageSerializer(data_types.Concept)),
    data_types.Region:
        _DataType('Region', 'regions', MessageSerializer(data_types.Region)),
    data_types.Frame:
        _DataType('Frame', 'frames', MessageSerializer(data_types.Frame)),
    data_types.Audio:
        _DataType('Audio', 'audio', MessageSerializer(data_types.Audio)),
    data_types.Video:
        _DataType('Video', 'video', MessageSerializer(data_types.Video)),

    # lists handled specially, not as generic lists using parts
    List[int]:
        _DataType('ndarray', 'ndarray', NDArraySerializer()),
    List[float]:
        _DataType('ndarray', 'ndarray', NDArraySerializer()),
    List[bool]:
        _DataType('ndarray', 'ndarray', NDArraySerializer()),
}


# add generic lists using parts, for all supported types
def _add_list_fields():
  for tp in list(_DATA_TYPES.keys()):
    if List[tp] in _DATA_TYPES:
      # already added as special case
      continue

    # check if data field is repeated, and if so, use repeated field for list
    field_name = _DATA_TYPES[tp].data_field
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name.get(field_name)
    repeated = descriptor and descriptor.label == descriptor.LABEL_REPEATED

    # add to supported types
    data_type = 'List[%s]' % _DATA_TYPES[tp].data_type
    data_field = field_name if repeated else 'parts[].' + field_name
    serializer = ListSerializer(_DATA_TYPES[tp].serializer)

    _DATA_TYPES[List[tp]] = _DataType(data_type, data_field, serializer)


_add_list_fields()
_SERIALIZERS_BY_TYPE_STRING = {dt.data_type: dt.serializer for dt in _DATA_TYPES.values()}
