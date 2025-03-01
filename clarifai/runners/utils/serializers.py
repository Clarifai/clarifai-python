import numpy as np
from PIL import Image as PILImage

from clarifai.runners.utils import data_handler
from clarifai.runners.utils.data_handler import Image, MessageData


class Serializer:

  def serialize(self, data_proto, field, value):
    pass

  def deserialize(self, data_proto, field):
    pass


def get_serializer(data_type: str) -> Serializer:
  if data_type in _SERIALIZERS_BY_TYPE_STRING:
    return _SERIALIZERS_BY_TYPE_STRING[data_type]
  if data_type.startswith('List['):
    inner_type_string = data_type[len('List['):-1]
    inner_serializer = get_serializer(inner_type_string)
    return ListSerializer(inner_serializer)
  raise ValueError(f'Unsupported type: "{data_type}"')


def is_repeated(field):
  return hasattr(field, 'add')


class AtomicFieldSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    setattr(data_proto, field, value)

  def deserialize(self, data_proto, field):
    return getattr(data_proto, field)


class MessageSerializer(Serializer):

  def __init__(self, message_class):
    self.message_class = message_class

  def serialize(self, data_proto, field, value):
    if isinstance(value, MessageData):
      value = value.to_proto()
    dst = getattr(data_proto, field)
    if is_repeated(dst):
      dst.add().CopyFrom(value)
    else:
      dst.CopyFrom(value)

  def deserialize(self, data_proto, field):
    src = getattr(data_proto, field)
    if is_repeated(src):
      return [self.message_class.from_proto(item) for item in src]
    else:
      return self.message_class.from_proto(src)


class ImageSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    if isinstance(value, PILImage.Image):
      value = Image.from_pil(value)
    if isinstance(value, MessageData):
      value = value.to_proto()
    getattr(data_proto, field).CopyFrom(value)

  def deserialize(self, data_proto, field):
    value = getattr(data_proto, field)
    return Image.from_proto(value)


class NDArraySerializer(Serializer):

  def serialize(self, data_proto, field, value):
    value = np.asarray(value)
    proto = getattr(data_proto, field)
    proto.buffer = value.tobytes()
    proto.shape.extend(value.shape)
    proto.dtype = str(value.dtype)

  def deserialize(self, data_proto, field):
    proto = getattr(data_proto, field)
    array = np.frombuffer(proto.buffer, dtype=np.dtype(proto.dtype)).reshape(proto.shape)
    return array


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
      self.inner_serializer.serialize(data_proto, field, item)  # appends to repeated field

  def deserialize(self, data_proto, field):
    if field.startswith('parts[].'):
      inner_field = field[len('parts[].'):]
      return [
          self.inner_serializer.deserialize(part.data, inner_field) for part in data_proto.parts
      ]
    repeated = getattr(data_proto, field)
    assert is_repeated(repeated), f"Field {field} is not repeated"
    return self.inner_serializer.deserialize(data_proto, field)  # returns repeated field list


# TODO dict serializer, maybe json only?

_SERIALIZERS_BY_TYPE_STRING = {
    'str': AtomicFieldSerializer(),
    'bytes': AtomicFieldSerializer(),
    'int': AtomicFieldSerializer(),
    'float': AtomicFieldSerializer(),
    'bool': AtomicFieldSerializer(),
    'ndarray': NDArraySerializer(),
    'Image': ImageSerializer(),
    'Text': MessageSerializer(data_handler.Text),
    'Concept': MessageSerializer(data_handler.Concept),
    'Region': MessageSerializer(data_handler.Region),
    'Audio': MessageSerializer(data_handler.Audio),
    'Video': MessageSerializer(data_handler.Video),

    # special cases for lists of numeric types serialized as ndarrays
    'List[int]': NDArraySerializer(),
    'List[float]': NDArraySerializer(),
    'List[bool]': NDArraySerializer(),
}
