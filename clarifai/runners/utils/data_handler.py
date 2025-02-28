import io

from typing import get_origin

import numpy as np
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage


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
    return value


class ImageSerializer(Serializer):

  def serialize(self, data_proto, field, value):
    if isinstance(value, PILImage):
      value = Image.from_pil(value)
    if isinstance(value, MessageData):
      value = value.to_proto()
    getattr(data_proto, field).CopyFrom(value)

  def deserialize(self, data_proto, field, python_type):
    value = getattr(data_proto, field)
    if python_type == PILImage:
      return Image.from_proto(value).to_pil()
    if issubclass(python_type, MessageData):
      return python_type.from_proto(value)
    return value


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
    elif get_origin(python_type) == list:
      return array.tolist()
    else:
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


class MessageData:

  def to_proto(self):
    raise NotImplementedError

  @classmethod
  def from_proto(cls, proto):
    raise NotImplementedError


class Text(MessageData):

  def __init__(self, text: str, url: str = None):
    self.text = text
    self.url = url

  def to_proto(self) -> TextProto:
    return TextProto(raw=self.text or '', self_url=self.url or '')

  @classmethod
  def from_proto(cls, proto: TextProto) -> "Text":
    return cls(proto.raw, proto.url or None)


class Image(MessageData):

  def __init__(self, proto_image: ImageProto):
    self.proto = proto_image

  @property
  def url(self) -> str:
    return self.proto.url

  @url.setter
  def url(self, value: str):
    self.proto.url = value

  @property
  def bytes(self) -> bytes:
    return self.proto.base64

  @bytes.setter
  def bytes(self, value: bytes):
    self.proto.base64 = value

  def __repr__(self) -> str:
    attrs = []
    if self.url:
      attrs.append(f"url={self.url!r}")
    if self.bytes:
      attrs.append(f"bytes=<{len(self.bytes)} bytes>")
    return f"Image({', '.join(attrs)})"

  @classmethod
  def from_url(cls, url: str) -> "Image":
    proto_image = ImageProto(url=url)
    return cls(proto_image)

  @classmethod
  def from_pil(cls, pil_image: PILImage.Image) -> "Image":
    with io.BytesIO() as output:
      pil_image.save(output, format="PNG")
      image_bytes = output.getvalue()
    proto_image = ImageProto(base64=image_bytes)
    return cls(proto_image)

  def to_pil(self) -> PILImage.Image:
    return PILImage.open(io.BytesIO(self.proto.base64))

  def to_numpy(self) -> np.ndarray:
    # below is very slow, need to find a better way
    # return np.array(self.to_pil())
    pass

  def to_proto(self) -> ImageProto:
    return self.proto


class Audio(MessageData):

  def __init__(self, proto_audio: AudioProto):
    self.proto = proto_audio

  @property
  def url(self) -> str:
    return self.proto.url

  @url.setter
  def url(self, value: str):
    self.proto.url = value

  @property
  def bytes(self) -> bytes:
    return self.proto.base64

  @bytes.setter
  def bytes(self, value: bytes):
    self.proto.base64 = value

  @classmethod
  def from_url(cls, url: str) -> "Audio":
    proto_audio = AudioProto(url=url)
    return cls(proto_audio)

  def __repr__(self) -> str:
    attrs = []
    if self.url:
      attrs.append(f"url={self.url!r}")
    if self.bytes:
      attrs.append(f"bytes=<{len(self.bytes)} bytes>")
    return f"Audio({', '.join(attrs)})"

  def to_proto(self) -> AudioProto:
    return self.proto


class Video(MessageData):

  def __init__(self, proto_video: VideoProto):
    self.proto = proto_video

  @property
  def url(self) -> str:
    return self.proto.url

  @url.setter
  def url(self, value: str):
    self.proto.url = value

  @property
  def bytes(self) -> bytes:
    return self.proto.base64

  @bytes.setter
  def bytes(self, value: bytes):
    self.proto.base64 = value

  @classmethod
  def from_url(cls, url: str) -> "Video":
    proto_video = VideoProto(url=url)
    return cls(proto_video)

  def __repr__(self) -> str:
    attrs = []
    if self.url:
      attrs.append(f"url={self.url!r}")
    if self.bytes:
      attrs.append(f"bytes=<{len(self.bytes)} bytes>")
    return f"Video({', '.join(attrs)})"

  def to_proto(self) -> VideoProto:
    return self.proto
