import io

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage


class Output:

  def __init__(self, text=None, image=None, audio=None, video=None, metadata=None):
    self.text = text
    self.image = image
    self.audio = audio
    self.video = video

  def to_proto(self, output):
    data_proto = resources_pb2.Data()

    if output.text is not None:
      if isinstance(output.text, Text):
        data_proto.text.raw = output.text.text
      elif isinstance(output.text, str):
        data_proto.text.raw = output.text
      else:
        raise TypeError("Output text must be of type str or Text")

    if output.image is not None:
      if isinstance(output.image, Image):
        data_proto.image.bytes = output.image.bytes
      elif isinstance(output.image, bytes):
        data_proto.image.bytes = output.image
      else:
        raise TypeError("Output image must be of type bytes or Image")

    if output.audio is not None:
      if isinstance(output.audio, Audio):
        data_proto.audio = output.audio.to_proto()
      else:
        raise TypeError("Output audio must be of type Audio")
    if output.video is not None:
      if isinstance(output.video, Video):
        data_proto.video = output.video.to_proto()
      else:
        raise TypeError("Output video must be of type Video")

    return service_pb2.Output(data=data_proto)


class Text:

  def __init__(self, text: str):
    self.text = text

  def to_proto(self) -> TextProto:
    return TextProto(raw=self.text)

  @classmethod
  def from_proto(cls, proto: TextProto) -> "Text":
    return cls(proto.raw)


class Image:

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

  def to_proto(self) -> ImageProto:
    return self.proto


class Audio:

  def __init__(self, proto_audio: AudioProto):
    self.proto = proto_audio

  def to_proto(self) -> AudioProto:
    return self.proto


class Video:

  def __init__(self, proto_video: VideoProto):
    self.proto = proto_video

  def to_proto(self) -> VideoProto:
    return self.proto
