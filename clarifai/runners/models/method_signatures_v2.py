import re
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

    #var = resources_pb2.MethodVariable()   # TODO
    var = _NamedFields()
    var.name = name
    var.python_type = _PYTHON_TYPES[tp]
    var.data_field = _DATA_FIELDS[tp]
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
    #_check_type(data, sig.python_type)
    data_proto, field = _get_named_part(proto, sig.data_field)
    serializer = _get_serializer(data_field=field, python_type=sig.python_type)
    serializer.serialize(data, data_proto)
  return proto


def _check_type(data, type_string):
  tp = _PYTHON_TYPES.reverse_map[type_string]
  # TODO: can also check for compatibility with proto fields and other types that are ok to use
  if not isinstance(data, tp):
    raise TypeError('Expected type %s, got %s' % (tp, type(data)))


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


class _NestedData:
  '''
    Mixin for allowing direct access to nested data fields using dot notation.

    For example, given a proto with a field `data` containing a field `text`, you can access the text field directly using `proto.text`.
    '''

  def __getattr__(self, name):
    # check this proto first, then data
    # note we don't need to implement setattr, since all fields in Data are non-atomic messages
    if name in self.DESCRIPTOR.fields_by_name:
      return super().__getattr__(name)
    if name in self.data.DESCRIPTOR.fields_by_name:
      return getattr(self.data, name)
    return super().__getattr__(name)  # raise AttributeError


def _MessageProxy(proto_cls):
  '''
    Create a proxy class for the given proto class, which allows direct access to nested data fields using dot notation.
    '''

  class Proxy:

    def __init__(self, *args, **kwargs):
      self._proto = proto_cls(*args, **kwargs)

    def __getattr__(self, name):
      try:
        return getattr(self._proto, name)
      except AttributeError:
        return super().__getattr__(name)

    def __setattr__(self, name, value):
      if name in self._proto.DESCRIPTOR.fields_by_name:
        setattr(self._proto, name, value)
      else:
        super().__setattr__(name, value)

  return Proxy


class Concept(_MessageProxy(resources_pb2.Concept)):
  '''
  Concept proto, containing the following fields:

  * id (str): concept id
  * name (str): concept name
  * value (float): concept value, e.g. 0.9
  '''


class Region(_MessageProxy(resources_pb2.Region), _NestedData):
  '''
  Region proto, containing the following fields:

  * id (str): region id, internally set
  * region_info (RegionInfo): shape and position of the region, e.g. bounding box
  * data (Data): region data, e.g. concepts, text, etc

  This class also contains convenience methods for getting and setting region data.
  '''

  @property
  def box(self):
    '''
    Get the bounding box of the region in xyxy format.
    '''
    b = self.region_info.bounding_box
    return (b.left_col, b.top_row, b.right_col, b.bottom_row)

  @box.setter
  def box(self, box):
    '''
    Set the bounding box of the region, given as a tuple (x1, y1, x2, y2).

    Args:
      box (tuple): bounding box in xyxy format
    '''
    b = self.region_info.bounding_box
    b.left_col = box[0]
    b.top_row = box[1]
    b.right_col = box[2]
    b.bottom_row = box[3]

  # TODO similar methods for polygon, mask, etc


class Part(_MessageProxy(resources_pb2.Part), _NestedData):
  '''
    Part proto, containing the following fields:

    * name (str): part name
    * data (Data): part data, e.g. concepts, text, etc (nested data, can be accessed directly)
    '''
  pass


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
    np.ndarray: 'np.ndarray',
    PIL.Image.Image: 'PIL.Image.Image',

    # protos, copied as-is
    resources_pb2.Text: 'Text',
    #resources_pb2.Bytes: 'Bytes',
    resources_pb2.Image: 'Image',
    resources_pb2.Video: 'Video',
    resources_pb2.Concept: 'Concept',
    resources_pb2.Region: 'Region',
    resources_pb2.Frame: 'Frame',
})

# data fields for supported python types
_DATA_FIELDS = {
    # common python types
    str: 'text',
    #bytes: 'bytes',
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
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name]
    repeated = (descriptor.label == descriptor.LABEL_REPEATED)
    _DATA_FIELDS[list_tp] = field_name if repeated else 'parts[].' + field_name


_add_list_fields()
