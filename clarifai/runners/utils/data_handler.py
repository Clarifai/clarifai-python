import io
import numpy as np
from typing import Any
from PIL import Image as PILImage

from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from PIL import Image as PILImage


class Output:

  def __init__(self, **kwargs: Any):
    self.parts = kwargs  # Stores named output parts

  def to_proto(self) -> resources_pb2.Output:
    """Converts the Output instance to a Clarifai protobuf Output message."""
    data_proto = resources_pb2.Data()
    for part_name, part_value in self.parts.items():
      part = data_proto.parts.add()
      part.id = part_name  # Assign the part name as the ID

      # Handle different data types and convert to proto
      if isinstance(part_value, Text):
        part.data.text.CopyFrom(part_value.to_proto())
      elif isinstance(part_value, Image):
        part.data.image.CopyFrom(part_value.to_proto())
      elif isinstance(part_value, Audio):
        part.data.audio.CopyFrom(part_value.to_proto())
      elif isinstance(part_value, Video):
        part.data.video.CopyFrom(part_value.to_proto())
      elif isinstance(part_value, str):
        part.data.text.raw = part_value
      elif isinstance(part_value, bytes):
        part.data.base64 = part_value
      elif isinstance(part_value, int):
        part.data.int_value = part_value
      elif isinstance(part_value, float):
        part.data.float_value = part_value
      elif isinstance(part_value, bool):
        part.data.boolean = part_value
      elif isinstance(part_value, np.ndarray):
        ndarray_proto = resources_pb2.NDArray(
            buffer=part_value.tobytes(),
            shape=part_value.shape,
            dtype=str(part_value.dtype))
        part.data.ndarray.CopyFrom(ndarray_proto)
      elif isinstance(part_value, PILImage.Image):
        image = Image.from_pil(part_value)
        part.data.image.CopyFrom(image.to_proto())
      else:
        raise TypeError(f"Unsupported Output type {type(part_value)} for part '{part_name}'")
      else:
        raise TypeError(f"Unsupported Output type {type(part_value)} for part '{part_name}'")

    return resources_pb2.Output(data=data_proto)


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
