from typing import List, get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from PIL.Image import Image as PILImage


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
    is_list = (get_origin(tp) == list)
    if is_list:
      tp = get_args(tp)[0]
    if get_origin(tp) == np.ndarray:
      tp = np.ndarray
    if tp not in _PYTHON_TYPES:
      raise ValueError(f'Unsupported type: {tp}')

    var = resources_pb2.MethodVariable()
    var.name = name
    var.data_field = _DATA_FIELDS[tp]
    if is_list:
      var.python_type = 'List[%s]' % _PYTHON_TYPES[tp]
    else:
      var.python_type = _PYTHON_TYPES[tp]
    vars.append(var)

  # check if any fields are used more than once, and if so, use parts
  fields_unique = (len(set(var.data_field for var in vars)) == len(vars))
  if not fields_unique:
    for var in vars:
      var.data_field = 'parts[%s].%s' % (var.name, var.data_field)

  return vars


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
    PILImage: 'PIL.Image.Image',

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
    PILImage: 'image',

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
    List[PILImage]: 'frames',  # TODO use this or generic list?
}


# add generic lists using parts, for all supported types
def _add_list_fields():
  for tp in _PYTHON_TYPES.keys():
    list_tp = List[tp]
    field_name = _DATA_FIELDS[tp]
    if list_tp in _DATA_FIELDS:
      # already added as special case
      continue
    # check if repeated field (we can use repeated fields for lists directly)
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name]
    repeated = (descriptor.label == descriptor.LABEL_REPEATED)
    _DATA_FIELDS[list_tp] = field_name if repeated else 'parts[].' + field_name


_add_list_fields()
