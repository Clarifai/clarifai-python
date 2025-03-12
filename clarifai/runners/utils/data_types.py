import io
import json
from typing import Iterable, List, get_args, get_origin

import numpy as np
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Concept as ConceptProto
from clarifai_grpc.grpc.api.resources_pb2 import Frame as FrameProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Region as RegionProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage


class MessageData:

  def to_proto(self):
    raise NotImplementedError

  @classmethod
  def from_proto(cls, proto):
    raise NotImplementedError

  @classmethod
  def from_value(cls, value):
    if isinstance(value, cls):
      return value
    return cls(value)

  def cast(self, python_type):
    if python_type == self.__class__:
      return self
    raise TypeError(f'Incompatible type for {self.__class__.__name__}: {python_type}')


class NamedFieldsMeta(type):
  """Metaclass to create NamedFields subclasses with __annotations__ when fields are specified."""

  def __call__(cls, *args, **kwargs):
    # Check if keyword arguments are types (used in type annotations)
    if kwargs and all(isinstance(v, type) for v in kwargs.values()):
      # Dynamically create a subclass with __annotations__
      name = f"NamedFields({', '.join(f'{k}:{v.__name__}' for k, v in kwargs.items())})"
      return type(name, (cls,), {'__annotations__': kwargs})
    else:
      # Create a normal instance for runtime data
      return super().__call__(*args, **kwargs)


class NamedFields(metaclass=NamedFieldsMeta):
  """A class that can be used to store named fields with values."""

  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

  def items(self):
    return self.__dict__.items()

  def keys(self):
    return self.__dict__.keys()

  def values(self):
    return self.__dict__.values()

  def __contains__(self, key):
    return key in self.__dict__

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __repr__(self):
    return f"{self.__class__.__name__}({', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())})"

  def __origin__(self):
    return self

  def __args__(self):
    return list(self.keys())


class Stream(Iterable):
  pass


class JSON:

  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return self.value == other

  def __bool__(self):
    return bool(self.value)

  def to_json(self):
    return json.dumps(self.value)

  @classmethod
  def from_json(cls, json_str):
    return cls(json.loads(json_str))

  @classmethod
  def from_value(cls, value):
    return cls(value)

  def cast(self, python_type):
    if not isinstance(self.value, python_type):
      raise TypeError(f'Incompatible type {type(self.value)} for {python_type}')
    return self.value


class Text(MessageData):

  def __init__(self, text: str, url: str = None):
    self.text = text
    self.url = url

  def __eq__(self, other):
    if isinstance(other, Text):
      return self.text == other.text and self.url == other.url
    if isinstance(other, str):
      return self.text == other
    return False

  def __bool__(self):
    return bool(self.text) or bool(self.url)

  def to_proto(self) -> TextProto:
    return TextProto(raw=self.text or '', url=self.url or '')

  @classmethod
  def from_proto(cls, proto: TextProto) -> "Text":
    return cls(proto.raw, proto.url or None)

  @classmethod
  def from_value(cls, value):
    if isinstance(value, str):
      return cls(value)
    if isinstance(value, Text):
      return value
    if isinstance(value, dict):
      return cls(value.get('text'), value.get('url'))
    raise TypeError(f'Incompatible type for Text: {type(value)}')

  def cast(self, python_type):
    if python_type == str:
      return self.text
    if python_type == Text:
      return self
    raise TypeError(f'Incompatible type for Text: {python_type}')


class Concept(MessageData):

  def __init__(self, name: str, value: float = 0):
    self.name = name
    self.value = value

  def __repr__(self) -> str:
    return f"Concept(name={self.name!r}, value={self.value})"

  def to_proto(self):
    return ConceptProto(name=self.name, value=self.value)

  @classmethod
  def from_proto(cls, proto: ConceptProto) -> "Concept":
    return cls(proto.name, proto.value)


class Region(MessageData):

  def __init__(self, proto_region: RegionProto):
    self.proto = proto_region

  @property
  def box(self) -> List[float]:
    bbox = self.proto.region_info.bounding_box
    return [bbox.left_col, bbox.top_row, bbox.right_col, bbox.bottom_row]  # x1, y1, x2, y2

  @box.setter
  def box(self, value: List[float]):
    bbox = self.proto.region_info.bounding_box
    bbox.left_col, bbox.top_row, bbox.right_col, bbox.bottom_row = value

  @property
  def concepts(self) -> List[Concept]:
    return [Concept.from_proto(proto) for proto in self.proto.data.concepts]

  @concepts.setter
  def concepts(self, value: List[Concept]):
    self.proto.data.concepts.extend([concept.to_proto() for concept in value])

  def __repr__(self) -> str:
    return f"Region(box={self.box}, concepts={self.concepts})"

  def to_proto(self) -> RegionProto:
    return self.proto

  @classmethod
  def from_proto(cls, proto: RegionProto) -> "Region":
    return cls(proto)


class Image(MessageData):

  def __init__(self, proto_image: ImageProto = None, url: str = None, bytes: bytes = None):
    if proto_image is None:
      proto_image = ImageProto()
    self.proto = proto_image
    # use setters for init vals
    if url:
      self.url = url
    if bytes:
      self.bytes = bytes

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
    if not self.proto.base64:
      raise ValueError("Image has no bytes")
    return PILImage.open(io.BytesIO(self.proto.base64))

  def to_numpy(self) -> np.ndarray:
    return np.asarray(self.to_pil())

  def to_proto(self) -> ImageProto:
    return self.proto

  @classmethod
  def from_proto(cls, proto: ImageProto) -> "Image":
    return cls(proto)

  @classmethod
  def from_value(cls, value):
    if isinstance(value, PILImage.Image):
      return cls.from_pil(value)
    if isinstance(value, Image):
      return value
    raise TypeError(f'Incompatible type for Image: {type(value)}')

  def cast(self, python_type):
    if python_type == Image:
      return self
    if python_type in (PILImage.Image, PILImage):
      return self.to_pil()
    if python_type == np.ndarray or get_origin(python_type) == np.ndarray:
      return self.to_numpy()
    raise TypeError(f'Incompatible type for Image: {python_type}')


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

  @classmethod
  def from_proto(cls, proto: AudioProto) -> "Audio":
    return cls(proto)


class Frame(MessageData):

  def __init__(self, proto_frame: FrameProto):
    self.proto = proto_frame

  @property
  def time(self) -> float:
    # TODO: time is a uint32, so this will overflow at 49.7 days
    # we should be using double or uint64 in the proto instead
    return self.proto.frame_info.time / 1000.0

  @time.setter
  def time(self, value: float):
    self.proto.frame_info.time = int(value * 1000)

  @property
  def image(self) -> Image:
    return Image.from_proto(self.proto.data.image)

  @image.setter
  def image(self, value: Image):
    self.proto.data.image.CopyFrom(value.to_proto())

  @property
  def regions(self) -> List[Region]:
    return [Region(region) for region in self.proto.data.regions]

  @regions.setter
  def regions(self, value: List[Region]):
    self.proto.data.regions.extend([region.proto for region in value])

  def to_proto(self) -> FrameProto:
    return self.proto

  @classmethod
  def from_proto(cls, proto: FrameProto) -> "Frame":
    return cls(proto)


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

  @classmethod
  def from_proto(cls, proto: VideoProto) -> "Video":
    return cls(proto)


def cast(value, python_type):
  list_type = (get_origin(python_type) == list)
  if isinstance(value, MessageData):
    return value.cast(python_type)
  if list_type and isinstance(value, np.ndarray):
    return value.tolist()
  if list_type and isinstance(value, list):
    if get_args(python_type):
      inner_type = get_args(python_type)[0]
      return [cast(item, inner_type) for item in value]
    if not isinstance(value, Iterable):
      raise TypeError(f'Expected list, got {type(value)}')
  return value
