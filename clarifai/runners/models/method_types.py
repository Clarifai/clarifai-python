import io
from typing import List, Tuple, get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Concept, Image
from PIL import Image as PILImage


def build_function_serializers(func):
  '''
    Build serializers for the given function.
    '''
  types = dict(func.__annotations__)
  output_type = types.pop('return', None)
  if len(types) == 1:
    input_type = list(types.values())[0]
  else:
    input_type = Input(**types)
  input_serializer = build_serializer(input_type)
  output_serializer = build_serializer(output_type)
  return input_serializer, output_serializer


def build_serializer(type_annotation):
  '''
    Build a serializer for the given type annotation.
    '''
  if type_annotation in _SERIALIZERS:
    return _SERIALIZERS[type_annotation]

  # TODO check json-able types

  # lists of data fields
  # note we only support one level of nesting, more might get difficult to maintain
  if get_origin(type_annotation) == list:
    inner_type = get_args(type_annotation)[0]
    if inner_type in _SERIALIZERS:
      return ListSerializer(_SERIALIZERS[inner_type])
    raise NotImplementedError(f'List of {inner_type} not supported')

  # named parts fields
  if isinstance(type_annotation, Parts):
    cls = type_annotation.__class__
    #names, types = zip(*type_annotation.fields.items())
    names = type_annotation._fields
    types = [getattr(type_annotation, name) for name in names]
    return PartsSerializer(names, types, cls)


class Parts:

  def __init__(self, **fields):
    self._fields = list(fields.keys())
    for name, value in fields.items():
      setattr(self, name, value)

  def __setattr__(self, name, value):
    if name != '_fields':
      assert name in self._fields, f'Field {name} not found in {self.__class__.__name__}'
    super().__setattr__(name, value)


class Input(Parts):
  pass


class Output(Parts):
  pass


class Serializer:

  def serialize(self, data, proto=None):
    raise NotImplementedError

  def deserialize(self, proto):
    raise NotImplementedError


class PartsSerializer(Serializer):

  def __init__(self, names, types, python_type=dict):
    self.python_type = python_type
    self.fields = {}
    for name, type in zip(names, types):
      self.fields[name] = build_serializer(type)

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Data()
    for name, serializer in self.fields.items():
      part = proto.parts.add()
      #part.id = name  # TODO add id to parts
      part.data.metadata['name'] = name
      part_data = getattr(data, name)
      serializer.serialize(part_data, part.data)
    return proto

  def deserialize(self, proto):
    data = {}
    for part in proto.parts:
      #part_name = part.id  # TODO add id field to parts
      part_name = part.data.metadata['name']
      serializer = self.fields[part_name]
      data[part_name] = serializer.deserialize(part.data)
    return self.python_type(**data)


class ListSerializer(Serializer):

  def __init__(self, serializer, python_type=list):
    self.serializer = serializer
    self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Data()
    for i, item in enumerate(data):
      part = proto.parts.add()
      #part.id = str(i)  # TODO add id to parts
      self.serializer.serialize(item, part.data)
    return proto

  def deserialize(self, proto):
    ret = [self.serializer.deserialize(part.data) for part in proto.parts]
    if self.python_type is not list:
      return self.python_type(ret)
    return ret


class DataField(Serializer):

  def __init__(self, field_name, serializer, is_list=False, is_tuple=False):
    assert field_name in resources_pb2.Data.DESCRIPTOR.fields_by_name, f'Field {field_name} not found in Data proto'
    self.field_name = field_name
    self.serializer = serializer
    self.is_list = is_list
    self.is_tuple = is_tuple
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name]
    self.field_is_message = (descriptor.type == descriptor.TYPE_MESSAGE)
    self.field_is_repeated = (descriptor.label == descriptor.LABEL_REPEATED)
    if self.is_list or self.is_tuple:
      assert self.field_is_repeated, f'Field {field_name} must be repeated to be a list or tuple'
    if self.field_is_repeated and (self.is_list or self.is_tuple):
      self._serialize_field = self._serialize_repeated_list
    elif self.field_is_repeated:
      self._serialize_field = self._serialize_repeated_singleton
    else:
      self._serialize_field = self._serialize_singleton

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Data()
    dst = getattr(proto, self.field_name)
    self._serialize_field(data, dst, proto)
    return proto

  def _serialize_repeated_list(self, data, dst, proto):
    # list with repeated field
    if self.field_is_message:
      for item in data:
        self.serializer.serialize(item, dst.add())
    else:
      # list of atomic types
      dst.extend(data)

  def _serialize_repeated_singleton(self, data, dst, proto):
    # single data, but repeated field
    if self.field_is_message:
      self.serializer.serialize(data, dst.add())
    else:
      dst.append(data)

  def _serialize_singleton(self, data, dst, proto):
    # single field
    if self.field_is_message:
      self.serializer.serialize(data, dst)
    else:
      setattr(proto, self.field_name, self.serializer.serialize(data))

  def deserialize(self, proto):
    src = getattr(proto, self.field_name)
    if self.field_is_repeated:
      data = [self.serializer.deserialize(item) for item in src]
      if self.is_tuple:
        return tuple(data)
      if self.is_list:
        return data
      return data[0]  # singleton case
    else:
      return self.serializer.deserialize(src)


class TextSerializer(Serializer):

  def __init__(self, python_type=str):
    self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Text()
    proto.raw = data
    return proto

  def deserialize(self, proto):
    if self.python_type is None:
      return proto
    if self.python_type is str:
      return proto.raw
    raise ValueError(f'Unsupported text type: {self.python_type}')


class BytesSerializer(Serializer):

  def serialize(self, data, proto=None):
    assert proto is None
    return data

  def deserialize(self, proto):
    return proto  # atomic bytes


class NDArraySerializer(Serializer):

  def __init__(self, python_type=np.ndarray):
    if get_origin(python_type) == list:
      tp = get_args(python_type)[0]
      self.python_type = lambda arr: [tp(a) for a in arr]
    else:
      self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.NDArray()
    data = np.asarray(data)
    proto.buffer = np.ascontiguousarray(data).tobytes()
    proto.shape.extend(data.shape)
    proto.dtype = str(data.dtype)
    return proto

  def deserialize(self, proto):
    array = np.frombuffer(proto.buffer, dtype=proto.dtype).reshape(proto.shape)
    if self.python_type is not np.ndarray:
      return self.python_type(array)
    return array


class ImageSerializer(Serializer):

  def __init__(self, python_type=None):
    self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Image()
    if isinstance(data, PILImage):
      self.write_image(data, proto)
    elif isinstance(data, np.ndarray):
      self.write_image(PILImage.fromarray(data), proto)
    elif isinstance(data, resources_pb2.Image):
      proto.CopyFrom(data)
    else:
      raise ValueError(f'Unsupported image type: {type(data)}')
    return proto

  def write_image(self, image, proto):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    proto.base64 = buf.getvalue()
    proto.image_info.width, proto.image_info.height = image.size

  def deserialize(self, proto):
    if self.python_type is None:
      return proto
    img = PILImage.open(io.BytesIO(proto.base64))
    if self.python_type is PILImage:
      return img
    if self.python_type is np.ndarray:
      return np.asarray(img)
    raise ValueError(f'Unsupported image type: {self.python_type}')


class ConceptSerializer(Serializer):

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Concept()
    proto.CopyFrom(data)

  def deserialize(self, proto):
    return proto  # TODO


_SERIALIZERS = {
    # basic atomic types
    str:
        DataField('text', TextSerializer(str)),
    #bytes:
    #    DataField('ndarray', BytesSerializer(bytes)),
    float:
        DataField('ndarray', NDArraySerializer(float)),  # TODO too much overhead?
    int:
        DataField('ndarray', NDArraySerializer(int)),
    np.ndarray:
        DataField('ndarray', NDArraySerializer(np.ndarray)),

    # specialized proto message types
    #Text:
    #    DataField('text', TextSerializer()),
    Image:
        DataField('image', ImageSerializer()),
    #Video:
    #    DataField('video', VideoSerializer()),
    Concept:
        DataField('concepts', ConceptSerializer()),
    #Region:
    #    DataField('regions', RegionSerializer()),
    #Frame:
    #    DataField('frames', FrameSerializer()),

    # common python types
    PILImage:
        DataField('image', ImageSerializer(PILImage)),

    # lists of basic atomic types
    #    List[str]: json? TODO
    #    List[bytes]: json? TODO
    List[int]:
        DataField('ndarray', NDArraySerializer(List[int])),
    Tuple[int]:
        DataField('ndarray', NDArraySerializer(Tuple[int])),
    List[float]:
        DataField('ndarray', NDArraySerializer(List[float])),
    Tuple[float]:
        DataField('ndarray', NDArraySerializer(Tuple[float])),
}

# add serializers lists of things that are in repeated fields
for tp, serializer in list(_SERIALIZERS.items()):
  if serializer.field_is_repeated and not serializer.is_list:
    _SERIALIZERS[List[tp]] = DataField(serializer.field_name, serializer, is_list=True)
    _SERIALIZERS[Tuple[tp]] = DataField(serializer.field_name, serializer, is_tuple=True)
