import json
from typing import Iterable

import numpy as np


class Serializer:

  def serialize(self, data_proto, field, value):
    pass

  def deserialize(self, data_proto, field):
    pass


def is_repeated(field):
  return hasattr(field, 'add')


class AtomicFieldSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    try:
      setattr(data_proto, field, value)
    except TypeError as e:
      raise TypeError(f"Incompatible type for {field}: {type(value)}") from e

  def deserialize(self, data_proto, field):
    return getattr(data_proto, field)


class MessageSerializer(Serializer):

  def __init__(self, message_class):
    self.message_class = message_class

  def serialize(self, data_proto, field, value):
    value = self.message_class.from_value(value).to_proto()
    dst = getattr(data_proto, field)
    try:
      if is_repeated(dst):
        dst.add().CopyFrom(value)
      else:
        dst.CopyFrom(value)
    except TypeError as e:
      raise TypeError(f"Incompatible type for {field}: {type(value)}") from e

  def deserialize(self, data_proto, field):
    src = getattr(data_proto, field)
    if is_repeated(src):
      return [self.message_class.from_proto(item) for item in src]
    else:
      return self.message_class.from_proto(src)


class NDArraySerializer(Serializer):

  def __init__(self, as_list=False):
    self.as_list = as_list

  def serialize(self, data_proto, field, value):
    if self.as_list and not isinstance(value, Iterable):
      raise TypeError(f"Expected list, got {type(value)}")
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number):
      raise TypeError(f"Expected number array, got {value.dtype}")
    proto = getattr(data_proto, field)
    proto.buffer = value.tobytes()
    proto.shape.extend(value.shape)
    proto.dtype = str(value.dtype)

  def deserialize(self, data_proto, field):
    proto = getattr(data_proto, field)
    array = np.frombuffer(proto.buffer, dtype=np.dtype(proto.dtype)).reshape(proto.shape)
    if self.as_list:
      return array.tolist()
    return array


class JSONSerializer(Serializer):

  def __init__(self, type=None):
    self.type = type

  def serialize(self, data_proto, field, value):
    if self.type is not None and not isinstance(value, self.type):
      raise TypeError(f"Expected {self.type}, got {type(value)}")
    try:
      setattr(data_proto, field, json.dumps(value))
    except TypeError as e:
      raise TypeError(f"Incompatible type for {field}: {type(value)}") from e

  def deserialize(self, data_proto, field):
    return json.loads(getattr(data_proto, field))


class ListSerializer(Serializer):

  def __init__(self, inner_serializer):
    self.inner_serializer = inner_serializer

  def serialize(self, data_proto, field, value):
    if not isinstance(value, Iterable):
      raise TypeError(f"Expected iterable, got {type(value)}")
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
