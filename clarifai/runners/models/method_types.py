from typing import List, get_args, get_origin

import numpy as np
import PIL.Image
from clarifai_grpc.grpc.api import resources_pb2


def build_function_signature(func):
  '''
    Build a signature for the given function.
    '''
  input_type = dict(func.__annotations__)
  output_type = input_type.pop('return', None)
  if not isinstance(output_type, dict):
    output_type = {'return': output_type}

  input_vars = build_variables_signature(input_type)
  output_vars = build_variables_signature(output_type)

  method_signature = resources_pb2.MethodSignature()

  method_signature.name = func.__name__
  method_type = 'UNARY_UNARY'  # for now
  method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)

  method_signature.input_variables.extend(input_vars)
  method_signature.output_variables.extend(output_vars)
  return method_signature


def build_variables_signature(var_types):
  '''
    Build a data proto signature for the given type annotation.
    '''
  assert isinstance(var_types, dict)

  vars = []

  # check valid names (should already be constrained by python naming, but check anyway)
  for name in var_types.keys():
    if not name.isidentifier():
      raise ValueError(f'Invalid variable name: {name}')

  # get fields for each variable based on type
  for name, tp in var_types.items():
    tp = _normalize_type(tp)

    var = resources_pb2.MethodVariable()
    var.name = name
    var.python_type = _PYTHON_TYPES[tp]
    var.data_field = _DATA_FIELDS[tp]
    vars.append(var)

  # check if any fields are used more than once, and if so, use parts
  fields_unique = (len(set(var.data_field for var in vars)) == len(vars))
  if not fields_unique:
    for var in vars:
      var.data_field = 'parts[%s].%s' % (var.name, var.data_field)

  return vars


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


# names for supported python types
_PYTHON_TYPES = {
    # common python types
    str: 'str',
    bytes: 'bytes',
    int: 'int',
    float: 'float',
    bool: 'bool',
    np.ndarray: 'np.ndarray',
    PIL.Image.Image: 'PIL.Image.Image',

    # protos, copied as-is
    resources_pb2.Text: 'Text',
    resources_pb2.Image: 'Image',
    resources_pb2.Video: 'Video',
    resources_pb2.Concept: 'Concept',
    resources_pb2.Region: 'Region',
    resources_pb2.Frame: 'Frame',
}

# data fields for supported python types
_DATA_FIELDS = {
    # common python types
    str: 'text',
    bytes: 'bytes',
    int: 'ndarray',
    float: 'ndarray',
    bool: 'ndarray',
    np.ndarray: 'ndarray',
    PIL.Image.Image: 'image',

    # protos, copied as-is
    resources_pb2.Text: 'text',
    resources_pb2.Image: 'image',
    resources_pb2.Video: 'video',
    resources_pb2.Concept: 'concepts',
    resources_pb2.Region: 'regions',
    resources_pb2.Frame: 'frames',

    # lists handled specially, not as generic lists using parts
    List[int]: 'ndarray',
    List[float]: 'ndarray',
    List[bool]: 'ndarray',
    List[PIL.Image.Image]: 'frames',  # TODO use this or generic list?
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
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name]
    repeated = (descriptor.label == descriptor.LABEL_REPEATED)
    _DATA_FIELDS[list_tp] = field_name if repeated else 'parts[].' + field_name


_add_list_fields()
