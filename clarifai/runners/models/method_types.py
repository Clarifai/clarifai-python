import io
from typing import List, get_args, get_origin

import numpy as np
import PIL.Image
from clarifai_grpc.grpc.api import resources_pb2
from PIL.Image import Image as PILImage


def build_function_serializers(func):
  '''
    Build serializers for the given function.
    '''
  input_type = dict(func.__annotations__)
  output_type = input_type.pop('return', None)

  if len(input_type) == 1:
    input_type = list(input_type.values())[0]  # single input, no need for named parts

  input_serializer = build_serializer(input_type)
  output_serializer = build_serializer(output_type)
  return input_serializer, output_serializer


def build_serializer(data_type):
  '''
    Build a serializer for the given type annotation.
    '''
  if isinstance(data_type, dict):
    # named multi-part parts fields
    names, types = zip(*data_type.items())
    return PartsDict(names, types)

  if data_type in _SERIALIZERS:
    return _SERIALIZERS[data_type]


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
      part_data = data[name]
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
    if self.field_is_message and self.serializer is not None:
      for item in data:
        self.serializer.serialize(item, dst.add())
    else:
      # list of atomic types or messages that are already in protos
      dst.extend(data)

  def _serialize_repeated_singleton(self, data, dst, proto):
    # single data, but repeated field
    if self.field_is_message and self.serializer is not None:
      self.serializer.serialize(data, dst.add())
    else:
      dst.append(data)

  def _serialize_singleton(self, data, dst, proto):
    # single field
    if self.field_is_message and self.serializer is not None:
      self.serializer.serialize(data, dst)
    elif self.field_is_message:
      dst.CopyFrom(data)
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
    if isinstance(data, str):
      proto.raw = data
    elif isinstance(data, resources_pb2.Text):
      proto.CopyFrom(data)
    else:
      raise ValueError(f'Unsupported text type: {type(data)}')
    return proto

  def deserialize(self, proto):
    if self.python_type is None:
      return proto
    if self.python_type is str:
      return proto.raw
    raise ValueError(f'Unsupported text type: {self.python_type}')


class BytesMessage(Message):

  def __init__(self, python_type=bytes):
    self.python_type = python_type

  def serialize(self, data, proto=None):
    if proto is None:
      proto = resources_pb2.Bytes()
    proto.raw = data
    return proto

  def deserialize(self, proto):
    if self.python_type is None:
      return proto
    if self.python_type is bytes:
      return proto.raw
    raise ValueError(f'Unsupported text type: {self.python_type}')


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

    # numeric types
    float:
        DataMessageField('ndarray', NDArrayMessage(float)),  # TODO too much overhead?
    int:
        DataMessageField('ndarray', NDArrayMessage(int)),
    np.ndarray:
        DataMessageField('ndarray', NDArrayMessage(np.ndarray)),
    # list of floats or ints are handled by 1d ndarrays, not as generic lists
    # (lists of ndarrays are handled generically as enumerated parts, below)
    List[float]:
        DataMessageField('ndarray', NDArrayMessage(List[float])),
    List[int]:
        DataMessageField('ndarray', NDArrayMessage(List[int])),

    # existing resources_pb2 proto message types are serialized as they are
    resources_pb2.Text:
        DataMessageField('text', None),
    resources_pb2.Image:
        DataMessageField('image', None),
    resources_pb2.Video:
        DataMessageField('video', None),
    resources_pb2.Concept:
        DataMessageField('concepts', None),
    resources_pb2.Region:
        DataMessageField('regions', None),
    resources_pb2.Frame:
        DataMessageField('frames', None),

    # common python types
    PIL.Image.Image:
        DataMessageField('image', ImageMessage(PIL.Image.Image)),
}

# add serializers for single-level lists of non-list data fields
# only support one level of nesting, more will be difficult to maintain
for tp, data_field in list(_SERIALIZERS.items()):
  if get_origin(tp) == list:
    continue  # already a list
  if data_field.field_is_repeated:
    # lists of items that have repeated fields in the Data proto
    assert not data_field.is_list, 'List of lists not supported'
    _SERIALIZERS[List[tp]] = DataMessageField(
        data_field.field_name, data_field.serializer, is_list=True)
  else:
    # lists of items that have single fields in the Data proto -- use enumerated parts
    _SERIALIZERS[List[tp]] = PartsList(data_field)
print(_SERIALIZERS)
