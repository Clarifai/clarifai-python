import inspect
import json
import re
import types
from collections import namedtuple
from typing import List, get_args, get_origin

import numpy as np
import PIL.Image
import yaml
from clarifai_grpc.grpc.api import resources_pb2

from clarifai.runners.utils import data_handler
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
    if not (len(output_vars) == 1 and output_vars[0].streaming):
      raise TypeError('Generate methods must return a stream')
  elif method_type == 'stream':
    # TODO handle initial non-stream inputs, check for one stream input and one stream output
    if len(input_vars) != 1:
      raise TypeError('Stream methods must take a single Stream input')
    if not input_vars[0].streaming:
      raise TypeError('Stream methods must take a stream input')
    if len(output_vars) != 1 or not output_vars[0].streaming:
      raise TypeError('Stream methods must return a single Stream')
  else:
    raise TypeError('Invalid method type: %s' % method_type)

  #method_signature = resources_pb2.MethodSignature()   # TODO
  method_signature = _NamedFields()  #for now

  method_signature.name = func.__name__
  #method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)
  assert method_type in ('predict', 'generate', 'stream')
  method_signature.method_type = method_type

  #method_signature.inputs.extend(input_vars)
  #method_signature.outputs.extend(output_vars)
  method_signature.inputs = input_vars
  method_signature.outputs = output_vars
  return method_signature


def build_variables_signature(var_types: List[inspect.Parameter], is_output=False):
  '''
  Build a data proto signature for the given variable or return type annotation.
  '''

  vars = []

  # check valid names (should already be constrained by python naming, but check anyway)
  for param in var_types:
    if not param.name.isidentifier() and not (is_output and
                                              re.match(r'return(\.\d+)?', param.name)):
      raise ValueError(f'Invalid variable name: {param.name}')

  # get fields for each variable based on type
  for param in var_types:
    tp, streaming = _normalize_type(param.annotation, is_output=is_output)
    required = (param.default == inspect.Parameter.empty)

    #var = resources_pb2.MethodVariable()   # TODO
    var = _NamedFields()
    var.name = param.name
    var.data_type = _DATA_TYPES[tp].data_type
    var.data_field = _DATA_TYPES[tp].data_field
    var.streaming = streaming
    if not is_output:
      if required:
        var.required = True
      else:
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
  return json.dumps(signatures)


def signatures_from_json(json_str):
  return json.loads(json_str, object_pairs_hook=_NamedFields)


def signatures_to_yaml(signatures):
  # XXX go in/out of json to get the correct format and python dict types
  d = json.loads(signatures_to_json(signatures))
  return yaml.dump(d, default_flow_style=False)


def signatures_from_yaml(yaml_str):
  d = yaml.safe_load(yaml_str)
  return signatures_from_json(json.dumps(d))


def serialize(kwargs, signatures, proto=None):
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
    data_proto, field = _get_named_part(proto, sig.data_field, add_parts=True)
    serializer = get_serializer(sig.data_type)
    serializer.serialize(data_proto, field, data)
  return proto


def deserialize(proto, signatures):
  '''
  Deserialize the given proto into kwargs using the given signatures.
  '''
  kwargs = {}
  for sig in signatures:
    data_proto, field = _get_named_part(proto, sig.data_field, add_parts=False)
    serializer = get_serializer(sig.data_type)
    data = serializer.deserialize(data_proto, field)
    kwargs[sig.name] = data
  if len(kwargs) == 1 and 'return' in kwargs:  # case for single return value
    return kwargs['return']
  if kwargs and 'return.0' in kwargs:  # case for tuple return values
    return tuple(kwargs[f'return.{i}'] for i in range(len(kwargs)))
  return kwargs


def get_serializer(data_type: str) -> Serializer:
  if data_type in _SERIALIZERS_BY_TYPE_STRING:
    return _SERIALIZERS_BY_TYPE_STRING[data_type]
  if data_type.startswith('List['):
    inner_type_string = data_type[len('List['):-1]
    inner_serializer = get_serializer(inner_type_string)
    return ListSerializer(inner_serializer)
  raise ValueError(f'Unsupported type: "{data_type}"')


def _get_named_part(proto, field, add_parts):
  # gets the named part from the proto, according to the field path
  # note we only support one level of named parts
  parts = field.replace(' ', '').split('.')

  if len(parts) not in (1, 2, 3):  # field, parts[name].field, parts[name].parts[].field
    raise ValueError('Invalid field: %s' % field)

  if len(parts) == 1:
    return proto, field

  # list
  if parts[0] == 'parts[]':
    if len(parts) != 2:
      raise ValueError('Invalid field: %s' % field)
    return proto, field  # return the data that contains the list itself

  # named part
  if not (m := re.match(r'parts\[(\w+)\]', parts[0])):
    raise ValueError('Invalid field: %s' % field)
  if not (name := m.group(1)):
    raise ValueError('Invalid field: %s' % field)
  assert len(parts) in (2, 3)  # parts[name].field, parts[name].parts[].field
  part = next((part for part in proto.parts if part.id == name), None)
  if part is None:
    if not add_parts:
      raise ValueError('Missing part: %s' % name)
    part = proto.parts.add()
    part.id = name
  return part.data, '.'.join(parts[1:])


def _normalize_type(tp, is_output=False):
  '''
  Normalize the given type.
  '''
  # stream type indicates streaming, not part of the data itself
  streaming = (get_origin(tp) == data_handler.Stream)
  if streaming:
    tp = get_args(tp)[0]

  if is_output:
    # output type used for named return values, each with their own data type
    if isinstance(tp, (dict, data_handler.Output)):
      return {name: _normalize_data_type(val) for name, val in tp.items()}, streaming
    if tp == data_handler.Output:  # check for Output type without values
      raise TypeError('Output types must be instantiated with inner type values for each key')

  return _normalize_data_type(tp), streaming


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
    tp = data_handler.Image

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
    data_handler.Text:
        _DataType('Text', 'text', MessageSerializer(data_handler.Text)),
    data_handler.Image:
        _DataType('Image', 'image', ImageSerializer()),
    data_handler.Concept:
        _DataType('Concept', 'concepts', MessageSerializer(data_handler.Concept)),
    data_handler.Region:
        _DataType('Region', 'regions', MessageSerializer(data_handler.Region)),
    data_handler.Frame:
        _DataType('Frame', 'frames', MessageSerializer(data_handler.Frame)),
    data_handler.Audio:
        _DataType('Audio', 'audio', MessageSerializer(data_handler.Audio)),
    data_handler.Video:
        _DataType('Video', 'video', MessageSerializer(data_handler.Video)),

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
