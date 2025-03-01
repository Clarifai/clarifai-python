from typing import get_origin

import numpy as np
from PIL import Image as PILImage

from clarifai.runners.utils.data_handler import Image, MessageData


def get_serializer(python_type_string):
  if python_type_string in _SERIALIZERS_BY_TYPE_STRING:
    return _SERIALIZERS_BY_TYPE_STRING[python_type_string]
  if python_type_string.startswith('List['):
    inner_type_string = python_type_string[len('List['):-1]
    inner_serializer = get_serializer(inner_type_string)
    return ListSerializer(inner_serializer)
  raise ValueError(f"Unsupported type {python_type_string}")


def is_repeated(field):
  return hasattr(field, 'add')


class Serializer:

  def serialize(self, data_proto, field, value):
    pass

  def deserialize(self, data_proto, field, python_type):
    pass


class AtomicFieldSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    setattr(data_proto, field, value)

  def deserialize(self, data_proto, field, python_type):
    return python_type(getattr(data_proto, field))


class MessageSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    if isinstance(value, MessageData):
      value = value.to_proto()
    getattr(data_proto, field).CopyFrom(value)

  def deserialize(self, data_proto, field, python_type):
    value = getattr(data_proto, field)
    if issubclass(python_type, MessageData):
      return python_type.from_proto(value)
    if python_type is None:
      return value
    raise ValueError(f"Unsupported type {python_type} for message")


class ImageSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    if isinstance(value, PILImage.Image):
      value = Image.from_pil(value)
    if isinstance(value, MessageData):
      value = value.to_proto()
    getattr(data_proto, field).CopyFrom(value)

  def deserialize(self, data_proto, field, python_type):
    value = getattr(data_proto, field)
    if python_type in (PILImage.Image, PILImage):
      return Image.from_proto(value).to_pil()
    if issubclass(python_type, MessageData):
      return python_type.from_proto(value)
    raise ValueError(f"Unsupported type {python_type} for image")


class NDArraySerializer(Serializer):

  def serialize(self, data_proto, field, value):
    value = np.asarray(value)
    proto = getattr(data_proto, field)
    proto.buffer = value.tobytes()
    proto.shape.extend(value.shape)
    proto.dtype = str(value.dtype)

  def deserialize(self, data_proto, field, python_type):
    proto = getattr(data_proto, field)
    array = np.frombuffer(proto.buffer, dtype=np.dtype(proto.dtype)).reshape(proto.shape)
    if python_type == np.ndarray:
      return array
    if get_origin(python_type) == list:
      return array.tolist()
    raise ValueError(f"Unsupported type {python_type} for ndarray proto")


class ListSerializer(Serializer):

  def __init__(self, inner_serializer):
    self.inner_serializer = inner_serializer

  def serialize(self, data_proto, field, value):
    if field.startswith('parts[].'):
      inner_field = field[len('parts[].'):]
      for item in value:
        part = data_proto.parts.add()
        self.inner_serializer.serialize(part.data, inner_field, item)
      return
    repeated = getattr(data_proto, field)
    assert is_repeated(repeated), f"Field {field} is not repeated"
    for item in value:
      self.inner_serializer.serialize(repeated.add(), item)

  def deserialize(self, data_proto, field, python_type):
    if field.startswith('parts[].'):
      inner_field = field[len('parts[].'):]
      return [
          self.inner_serializer.deserialize(part.data, inner_field, python_type)
          for part in data_proto.parts
      ]
    repeated = getattr(data_proto, field)
    assert is_repeated(repeated), f"Field {field} is not repeated"
    return [self.inner_serializer.deserialize(item, python_type) for item in repeated]


# TODO dict serializer, maybe json only?

_SERIALIZERS_BY_TYPE_STRING = {
    'str': AtomicFieldSerializer(),
    'bytes': AtomicFieldSerializer(),
    'int': AtomicFieldSerializer(),
    'float': AtomicFieldSerializer(),
    'bool': AtomicFieldSerializer(),
    'ndarray': NDArraySerializer(),
    'Image': ImageSerializer(),
    'PIL.Image.Image': ImageSerializer(),
    'Text': MessageSerializer(),
    'Audio': MessageSerializer(),
    'Video': MessageSerializer(),

    # special cases for lists of numeric types serialized as ndarrays
    'List[int]': NDArraySerializer(),
    'List[float]': NDArraySerializer(),
    'List[bool]': NDArraySerializer(),
}
