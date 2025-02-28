import inspect
import re
from typing import List, get_args, get_origin

import numpy as np
import PIL.Image
from clarifai_grpc.grpc.api import resources_pb2

from clarifai.runners.utils import data_handler
from clarifai.runners.utils.serializers import get_serializer


def build_function_signature(func):
  '''
  Build a signature for the given function.
  '''
  sig = inspect.signature(func)

  return_annotation = sig.return_annotation
  if not isinstance(return_annotation, dict):
    return_annotation = {'return': return_annotation}

  input_vars = build_variables_signature(sig.parameters.values())
  output_vars = build_variables_signature([
      inspect.Parameter(name=name, kind=0, annotation=tp)
      for name, tp in return_annotation.items()
  ])

  #method_signature = resources_pb2.MethodSignature()   # TODO
  method_signature = _NamedFields()  #for now

  method_signature.name = func.__name__
  method_type = 'UNARY_UNARY'  # for now
  method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)

  #method_signature.input_variables.extend(input_vars)
  #method_signature.output_variables.extend(output_vars)
  method_signature.input_variables = input_vars
  method_signature.output_variables = output_vars
  return method_signature


def build_variables_signature(var_types: List[inspect.Parameter]):
  '''
  Build a data proto signature for the given variable or return type annotation.
  '''

  vars = []

  # check valid names (should already be constrained by python naming, but check anyway)
  for param in var_types:
    if not param.name.isidentifier():
      raise ValueError(f'Invalid variable name: {param.name}')

  # get fields for each variable based on type
  for param in var_types:
    tp = _normalize_type(param.annotation)
    # TODO: check default is compatible with type and figure out how to represent in the signature proto
    default = param.default  #if param.default != inspect.Parameter.empty else None

    #var = resources_pb2.MethodVariable()   # TODO
    var = _NamedFields()
    var.name = param.name
    var.python_type = _PYTHON_TYPES[tp]
    var.data_field = _DATA_FIELDS[tp]
    var.default = default
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


def serialize(kwargs, signatures, proto=None):
  '''
  Serialize the given kwargs into the proto using the given signatures.
  '''
  if proto is None:
    proto = resources_pb2.Data()
  unknown = set(kwargs.keys()) - set(sig.name for sig in signatures)
  if unknown:
    raise TypeError('Got unexpected argument: %s' % ', '.join(unknown))
  for sig in signatures:
    if sig.name not in kwargs:
      continue  # skip missing fields, they can be set to default on the server
    data = kwargs[sig.name]
    data_proto, field = _get_named_part(proto, sig.data_field)
    serializer = get_serializer(sig.python_type)
    serializer.serialize(data_proto, field, data)
  return proto


def deserialize(proto, signatures):
  '''
    Deserialize the given proto into kwargs using the given signatures.
    '''
  kwargs = {}
  for sig in signatures:
    data_proto, field = _get_named_part(proto, sig.data_field)
    serializer = get_serializer(sig.python_type)
    data = serializer.deserialize(data_proto, field, _PYTHON_TYPES.reverse_map[sig.python_type])
    kwargs[sig.name] = data
  return kwargs


def _get_named_part(proto, field):
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
  return proto.parts[name].data, '.'.join(parts[1:])


def _normalize_type(tp):
  '''
    Normalize the given type.
    '''

  # check if list, and if so, get inner type
  is_list = (get_origin(tp) == list)
  if is_list:
    tp = get_args(tp)[0]

  # check if numpy array, and if so, use ndarray
  if get_origin(tp) == np.ndarray:
    tp = np.ndarray

  # check for PIL images (sometimes types use the module, sometimes the class)
  if tp in (PIL.Image, PIL.Image.Image):
    tp = PIL.Image.Image

  # put back list
  if is_list:
    tp = List[tp]

  # check if supported type
  if tp not in _PYTHON_TYPES:
    raise ValueError(f'Unsupported type: {tp}')

  return tp


class _NamedFields(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


class Input(_NamedFields):
  pass


class Output(_NamedFields):
  pass


class _ReversableDict(dict):
  '''
    Reversable dictionary, allowing reverse lookups.
    '''

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reverse_map = {v: k for k, v in self.items()}

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.reverse_map[value] = key


# names for supported python types
_PYTHON_TYPES = _ReversableDict({
    # common python types
    str: 'str',
    #bytes: 'bytes',
    int: 'int',
    float: 'float',
    bool: 'bool',
    np.ndarray: 'ndarray',
    PIL.Image.Image: 'PIL.Image.Image',
    data_handler.Text: 'Text',
    data_handler.Image: 'Image',
    data_handler.Video: 'Video',
    data_handler.Concept: 'Concept',
    data_handler.Region: 'Region',
    data_handler.Frame: 'Frame',
})

# data fields for supported python types
_DATA_FIELDS = {
    # common python types
    str: 'string_value',
    bytes: 'bytes_value',
    int: 'int_value',
    float: 'float_value',
    bool: 'bool_value',
    np.ndarray: 'ndarray',
    PIL.Image.Image: 'image',

    # protos, copied as-is
    data_handler.Text: 'text',
    data_handler.Image: 'image',
    data_handler.Video: 'video',
    data_handler.Concept: 'concepts',
    data_handler.Region: 'regions',
    data_handler.Frame: 'frames',

    # lists handled specially, not as generic lists using parts
    List[int]: 'ndarray',
    List[float]: 'ndarray',
    List[bool]: 'ndarray',
    #List[PIL.Image.Image]: 'frames',  # TODO use this or generic parts list?
}


# add generic lists using parts, for all supported types
def _add_list_fields():
  for tp in list(_PYTHON_TYPES.keys()):
    assert get_origin(tp) != list, 'List type already exists'
    # add to supported types
    _PYTHON_TYPES[List[tp]] = 'List[%s]' % _PYTHON_TYPES[tp]

    # add field mapping
    list_tp = List[tp]
    if list_tp in _DATA_FIELDS:
      # already added as special case
      continue
    field_name = _DATA_FIELDS[tp]
    # check if repeated field (we can use repeated fields for lists directly)
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name.get(field_name)
    repeated = descriptor and descriptor.label == descriptor.LABEL_REPEATED
    _DATA_FIELDS[list_tp] = field_name if repeated else 'parts[].' + field_name


_add_list_fields()
