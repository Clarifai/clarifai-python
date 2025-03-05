import json
from typing import Dict, Iterable

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2

from clarifai.runners.utils import data_types


class Serializer:

  def serialize(self, data_proto, value):
    pass

  def deserialize(self, data_proto):
    pass


class AtomicFieldSerializer(Serializer):

  def __init__(self, field_name):
    self.field_name = field_name

  def serialize(self, data_proto, value):
    try:
      setattr(data_proto, self.field_name, value)
    except TypeError as e:
      raise TypeError(f"Incompatible type for {self.field_name}: {type(value)}") from e

  def deserialize(self, data_proto):
    return getattr(data_proto, self.field_name)


class MessageSerializer(Serializer):

  def __init__(self, field_name, message_class):
    self.field_name = field_name
    self.message_class = message_class
    descriptor = resources_pb2.Data.DESCRIPTOR.fields_by_name.get(field_name)
    self.is_repeated_field = descriptor and descriptor.label == descriptor.LABEL_REPEATED

  def serialize(self, data_proto, value):
    value = self.message_class.from_value(value).to_proto()
    dst = getattr(data_proto, self.field_name)
    try:
      if self.is_repeated_field:
        dst.add().CopyFrom(value)
      else:
        dst.CopyFrom(value)
    except TypeError as e:
      raise TypeError(f"Incompatible type for {self.field_name}: {type(value)}") from e

  def deserialize(self, data_proto):
    src = getattr(data_proto, self.field_name)
    if self.is_repeated_field:
      assert len(src) == 1
      return self.message_class.from_proto(src[0])
    else:
      return self.message_class.from_proto(src)


class NDArraySerializer(Serializer):

  def __init__(self, field_name, as_list=False):
    self.field_name = field_name
    self.as_list = as_list

  def serialize(self, data_proto, value):
    if self.as_list and not isinstance(value, Iterable):
      raise TypeError(f"Expected list, got {type(value)}")
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number):
      raise TypeError(f"Expected number array, got {value.dtype}")
    proto = getattr(data_proto, self.field_name)
    proto.buffer = value.tobytes()
    proto.shape.extend(value.shape)
    proto.dtype = str(value.dtype)

  def deserialize(self, data_proto):
    proto = getattr(data_proto, self.field_name)
    array = np.frombuffer(proto.buffer, dtype=np.dtype(proto.dtype)).reshape(proto.shape)
    if self.as_list:
      return array.tolist()
    return array


class JSONSerializer(Serializer):

  def __init__(self, field_name, type=None):
    self.field_name = field_name
    self.type = type

  def serialize(self, data_proto, value):
    #if self.type is not None and not isinstance(value, self.type):
    #  raise TypeError(f"Expected {self.type}, got {type(value)}")
    try:
      setattr(data_proto, self.field_name, json.dumps(value))
    except TypeError as e:
      raise TypeError(f"Incompatible type for {self.field_name}: {type(value)}") from e

  def deserialize(self, data_proto):
    return json.loads(getattr(data_proto, self.field_name))


class ListSerializer(Serializer):

  def __init__(self, inner_serializer):
    self.field_name = 'parts'
    self.inner_serializer = inner_serializer

  def serialize(self, data_proto, value):
    if not isinstance(value, Iterable):
      raise TypeError(f"Expected iterable, got {type(value)}")
    for item in value:
      part = data_proto.parts.add()
      self.inner_serializer.serialize(part.data, item)

  def deserialize(self, data_proto):
    return [self.inner_serializer.deserialize(part.data) for part in data_proto.parts]


class TupleSerializer(Serializer):

  def __init__(self, inner_serializers):
    self.field_name = 'parts'
    self.inner_serializers = inner_serializers

  def serialize(self, data_proto, value):
    if not isinstance(value, (tuple, list)):
      raise TypeError(f"Expected tuple, got {type(value)}")
    if len(value) != len(self.inner_serializers):
      raise ValueError(f"Expected tuple of length {len(self.inner_serializers)}, got {len(value)}")
    for i, (serializer, item) in enumerate(zip(self.inner_serializers, value)):
      part = data_proto.parts.add()
      part.name = str(i)
      serializer.serialize(part.data, item)

  def deserialize(self, data_proto):
    return tuple(
        serializer.deserialize(part.data)
        for serializer, part in zip(self.inner_serializers, data_proto.parts))


class NamedFieldsSerializer(Serializer):

  def __init__(self, named_field_serializers: Dict[str, Serializer]):
    self.field_name = 'parts'
    self.named_field_serializers = named_field_serializers

  def serialize(self, data_proto, value):
    for name, serializer in self.named_field_serializers.items():
      if name not in value:
        raise KeyError(f"Missing field {name}")
      part = self._get_part(data_proto, name, add=True)
      serializer.serialize(part.data, value[name])

  def deserialize(self, data_proto):
    value = data_types.NamedFields()
    for name, serializer in self.named_field_serializers.items():
      part = self._get_part(data_proto, name)
      value[name] = serializer.deserialize(part.data)
    return value

  def _get_part(self, data_proto, name, add=False):
    for part in data_proto.parts:
      if part.name == name:
        return part
    if add:
      part = data_proto.parts.add()
      part.name = name
      return part
    raise KeyError(f"Missing part with key {name}")


# TODO dict serializer, maybe json only?
