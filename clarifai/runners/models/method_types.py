import functools
import inspect
from typing import get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2

_ATOMIC_SERIALIZERS = {
    str: DataField('text', TextSerializer()),
    Text: DataField('text', TextSerializer()),
    bytes: DataField('bytes', BytesSerializer()),
    float: DataField('ndarray', NDArraySerializer(python_type=float)),  # TODO too much overhead?
    int: DataField('ndarray', NDArraySerializer(python_type=int)),
    List[int]: DataField('ndarray', NDArraySerializer(python_type=List[int])),
    List[float]: DataField('ndarray', NDArraySerializer(python_type=List[float])),
    np.ndarray: DataField('ndarray', NDArraySerializer(python_type=np.ndarray)),
    Image: DataField('image', ImageSerializer()),
    List[Image]: DataField('frames', ListSerializer(
        FrameSerializer())),  # special case for List[Image] -> List[Frame]
    Video: DataField('video', VideoSerializer()),
    Concept: DataField('concepts', ConceptSerializer()),
    Region: DataField('regions', RegionSerializer()),
    #Embedding:  TODO
    Frame: DataField('frames', FrameSerializer()),
}


@functools.cache
def build_serializer(type_annotation):
  '''
    Build a serializer for the given type annotation.
    '''
  if type_annotation == inspect._empty:
    raise ValueError('Type annotations are required')
  for atomic_type, serializer in _ATOMIC_SERIALIZERS.items():
    if type_annotation == atomic_type:
      return serializer
  if get_origin(type_annotation) == list:
    t = get_args(type_annotation)[0]
    if t == inspect._empty:
      raise ValueError('For list types, the inner type must be specified')
    if t == Image:  # special case for List[Image] -> List[Frame] as the underlying field
      return ListSerializer(FrameSerializer())
    return ListSerializer(build_serializer(t))


class Serializer:

  def serialize(self, proto, data):
    raise NotImplementedError

  def deserialize(self, proto):
    raise NotImplementedError


class DataField(Serializer):

  def __init__(self, field_name, serializer, repeated=False):
    assert field_name in resources_pb2.Data.DESCRIPTOR.fields_by_name, f'Field {field_name} not found in Data proto'
    self.field_name = field_name
    self.serializer = serializer
    self.repeated = (resources_pb2.Data.DESCRIPTOR.fields_by_name[field_name].label ==
                     FieldDescriptor.LABEL_REPEATED)
    self.is_list = isinstance(serializer, ListSerializer)

  def serialize(self, proto, data):
    dst = getattr(proto, self.field_name)
    if self.repeated:  # repeated field
      if self.is_list:
        # actual list of data
        self.serializer.serialize(dst, data)
      else:
        # single data, but repeated field
        self.serializer.serialize(dst.add(), data)
    else:
      # single field
      assert not self.is_list, 'Cannot use a list serializer for a single field'
      self.serializer.serialize(dst, data)

  def deserialize(self, proto):
    src = getattr(proto, self.field_name)
    if self.repeated:
      data = [self.serializer.deserialize(item) for item in src]
      return data if self.is_list else data[0]
    else:
      return self.serializer.deserialize(src)


class TextSerializer(Serializer):

  def serialize(self, proto, data):
    proto.text = data

  def deserialize(self, proto):
    return proto.text


class BytesSerializer(Serializer):

  def serialize(self, proto, data):
    proto.bytes = data

  def deserialize(self, proto):
    return proto.bytes


class NDArraySerializer(Serializer):

  def __init__(self, python_type=np.ndarray):
    self.python_type = python_type

  def serialize(self, proto, data):
    data = np.asarray(data).as_contiguousarray()
    proto.ndarray.buffer = data.tobytes()
    proto.ndarray.shape.extend(data.shape)
    proto.ndarray.dtype = str(data.dtype)

  def deserialize(self, proto):
    array = np.frombuffer(proto.ndarray.buffer, dtype=proto.ndarray.dtype)
    if python_type is not np.ndarray:
      return python_type(array)
    return array


class ConceptSerializer(Serializer):

  def serialize(self, proto, data):
    proto.CopyFrom(data)

  def deserialize(self, proto):
    return proto


class ListSerializer(Serializer):

  def __init__(self, inner_serializer):
    self.inner_serializer = inner_serializer

  def serialize(self, proto, data):
    for item in data:
      self.inner_serializer.serialize(proto.add(), item)

  def deserialize(self, proto):
    return [self.inner_serializer.deserialize(item) for item in proto]
