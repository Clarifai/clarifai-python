import io

import numpy as np
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage


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
