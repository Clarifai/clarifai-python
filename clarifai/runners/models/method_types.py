import io
from typing import List, get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Concept, Image
from PIL import Image as PILImage


def build_function_serializers(func):
  '''
    Build serializers for the given function.
    '''
  input_type = dict(func.__annotations__)
  output_type = input_type.pop('return', None)
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
      return PartsList(_SERIALIZERS[inner_type])
    raise NotImplementedError(f'List of {inner_type} not supported')

  # named parts fields
  if isinstance(type_annotation, Parts):
    cls = type_annotation.__class__
    #names, types = zip(*type_annotation.fields.items())
    names = type_annotation._fields
    types = [getattr(type_annotation, name) for name in names]
    return PartsDict(names, types, cls)


class Parts:

  def __init__(self, **fields):
    self._fields = list(fields.keys())
    for name, value in fields.items():
      setattr(self, name, value)

  def __setattr__(self, name, value):
    if name != '_fields':
      assert name in self._fields, f'Field {name} not found in {self.__class__.__name__}'
    super().__setattr__(name, value)


class Message:

  def serialize(self, data, proto=None):
    raise NotImplementedError

  def deserialize(self, proto):
    raise NotImplementedError


class PartsDict(Message):

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
      part.name = name
      part_data = getattr(data, name)
      serializer.serialize(part_data, part.data)
    return proto

  def deserialize(self, proto):
    data = {}
    for part in proto.parts:
      serializer = self.fields[part.name]
      data[part.name] = serializer.deserialize(part.data)
    return self.python_type(**data)


class PartsList(Message):

  def __init__(self, serializer, python_type=list):
    self.serializer = serializer
    self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Data()
    for i, item in enumerate(data):
      part = proto.parts.add()
      #part.name = str(i)  # TODO should we put this in for lists?
      self.serializer.serialize(item, part.data)
    return proto

  def deserialize(self, proto):
    ret = [self.serializer.deserialize(part.data) for part in proto.parts]
    if self.python_type is not list:
      return self.python_type(ret)
    return ret


class DataMessageField(Message):

  def __init__(self, field_name, serializer, is_list=False):
    assert field_name in resources_pb2.Data.DESCRIPTOR.fields_by_name, f'Field {field_name} not found in Data proto'
    self.field_name = field_name
    self.serializer = serializer
    self.is_list = is_list
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name]
    self.field_is_message = (descriptor.type == descriptor.TYPE_MESSAGE)
    self.field_is_repeated = (descriptor.label == descriptor.LABEL_REPEATED)
    if self.is_list:
      assert self.field_is_repeated, f'Field {field_name} must be repeated to be a list'
    if self.field_is_repeated and self.is_list:
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
      if self.is_list:
        return data
      return data[0]  # singleton case
    else:
      return self.serializer.deserialize(src)


class TextMessage(Message):

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


class BytesMessage(Message):

  def serialize(self, data, proto=None):
    assert proto is None
    return data

  def deserialize(self, proto):
    return proto  # atomic bytes


class NDArrayMessage(Message):

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


class ImageMessage(Message):

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


class ConceptMessage(Message):

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Concept()
    proto.CopyFrom(data)

  def deserialize(self, proto):
    return proto  # TODO


_SERIALIZERS = {
    # str and bytes types
    str:
        DataMessageField('text', TextMessage(str)),
    bytes:
        DataMessageField('bytes', BytesMessage(bytes)),
    List[str]:
        PartsList(DataMessageField('text', TextMessage(str))),

    # numeric types
    float:
        DataMessageField('ndarray', NDArrayMessage(float)),  # TODO too much overhead?
    int:
        DataMessageField('ndarray', NDArrayMessage(int)),
    np.ndarray:
        DataMessageField('ndarray', NDArrayMessage(np.ndarray)),
    # list of floats or ints are handled by ndarrays
    # (lists of ndarrays are handled generically as enumerated parts)
    List[float]:
        DataMessageField('ndarray', NDArrayMessage(List[float])),
    List[int]:
        DataMessageField('ndarray', NDArrayMessage(List[int])),

    # specialized proto message types
    #Text:
    #    DataMessageField('text', TextMessage()),
    Image:
        DataMessageField('image', ImageMessage()),
    #Video:
    #    DataMessageField('video', VideoMessage()),
    Concept:
        DataMessageField('concepts', ConceptMessage()),
    #Region:
    #    DataMessageField('regions', RegionMessage()),
    #Frame:
    #    DataMessageField('frames', FrameMessage()),

    # common python types
    PILImage:
        DataMessageField('image', ImageMessage(PILImage)),

    # lists of basic atomic types
    #    List[str]: json? TODO
    #    List[bytes]: json? TODO
    List[str]:
        DataMessageField('text', TextMessage(str), is_list=True),
}

# add serializers for lists of things that are in repeated fields
for tp, serializer in list(_SERIALIZERS.items()):
  if serializer.field_is_repeated and not serializer.is_list:
    _SERIALIZERS[List[tp]] = DataMessageField(serializer.field_name, serializer, is_list=True)
