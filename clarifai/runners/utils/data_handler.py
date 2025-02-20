import io
from typing import Any

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image as PILImage


def metadata_to_dict(data: resources_pb2.Data) -> dict:
  return MessageToDict(data.metadata)


def dict_to_metadata(data: resources_pb2.Data, metadata_dict: dict):
  struct = Struct()
  ParseDict(metadata_dict, struct)
  data.metadata.CopyFrom(struct)


def kwargs_to_proto(**kwargs) -> resources_pb2.Data:
  """Converts the kwargs to a Clarifai protobuf Data message."""

  def _handle_list(target_data, value_list, part_name):
    """Handles list values by processing each item into a new part."""
    if isinstance(value_list[0], dict):
      raise ValueError("List of dictionaries is not supported")

    for item in value_list:
      new_part = target_data.parts.add()
      _process_value(new_part.data, item, part_name)

  def _process_value(target_data, value, part_name):
    """Processes individual values and sets the appropriate proto field."""
    if isinstance(value, Text):
      target_data.text.CopyFrom(value.to_proto())
    elif isinstance(value, Image):
      target_data.image.CopyFrom(value.to_proto())
    elif isinstance(value, Audio):
      target_data.audio.CopyFrom(value.to_proto())
    elif isinstance(value, Video):
      target_data.video.CopyFrom(value.to_proto())
    elif isinstance(value, str):
      target_data.text.raw = value
    elif isinstance(value, bytes):
      target_data.bytes_value = value
    elif isinstance(value, int):
      target_data.int_value = value
    elif isinstance(value, float):
      target_data.float_value = value
    elif isinstance(value, bool):
      target_data.bool_value = value
    elif isinstance(value, np.ndarray):
      ndarray_proto = resources_pb2.NDArray(
          buffer=value.tobytes(), shape=value.shape, dtype=str(value.dtype))
      target_data.ndarray.CopyFrom(ndarray_proto)
    elif isinstance(value, PILImage.Image):
      image = Image.from_pil(value)
      target_data.image.CopyFrom(image.to_proto())
    else:
      raise TypeError(f"Unsupported type {type(value)} for part '{part_name}'")

  data_proto = resources_pb2.Data()
  for part_name, part_value in kwargs.items():
    part = data_proto.parts.add()
    part.id = part_name

    if isinstance(part_value, list):
      _handle_list(part.data, part_value, part_name)
    elif isinstance(part_value, dict):
      dict_to_metadata(part.data, part_value)
    else:
      _process_value(part.data, part_value, part_name)
  return data_proto


def proto_to_kwargs(data: resources_pb2.Data) -> dict:
  """Converts the Clarifai protobuf Data message to a dictionary."""

  def process_part(part, allow_metadata: bool = True) -> object:
    if part.HasField("text"):
      return Text.from_proto(part.text)
    elif part.HasField("image"):
      return Image(part.image)
    elif part.HasField("audio"):
      return Audio(part.audio)
    elif part.HasField("video"):
      return Video(part.video)
    elif part.bytes_value != b'':
      return part.bytes_value
    elif part.int_value != 0:
      return part.int_value
    elif part.float_value != 0.0:
      return part.float_value
    elif part.bool_value is not False:
      return part.bool_value
    elif part.HasField("ndarray"):
      ndarray = part.ndarray
      return np.frombuffer(ndarray.buffer, dtype=np.dtype(ndarray.dtype)).reshape(ndarray.shape)
    elif part.HasField("metadata"):
      if not allow_metadata:
        raise ValueError("Metadata in list is not supported")
      return metadata_to_dict(part)
    elif part.parts:
      return [process_part(p, allow_metadata=False) for p in part.parts]
    else:
      raise ValueError(f"Unknown part data: {part}")

  kwargs = {}
  for part in data.parts:
    part_name = part.id
    part_data = part.data
    kwargs[part_name] = process_part(part_data)
  return kwargs


class Output:

  def __init__(self, **kwargs: Any):
    if not kwargs:
      raise ValueError("Output must have at least one  key-value pair")
    if isinstance(kwargs, dict):
      kwargs = kwargs
    else:
      raise ValueError("Output must be a dictionary")
    self.parts = kwargs

  def to_proto(self) -> resources_pb2.Output:
    """Converts the Output instance to a Clarifai protobuf Output message."""
    data_proto = kwargs_to_proto(**self.parts)

    return resources_pb2.Output(
        data=data_proto, status=status_pb2.Status(code=status_code_pb2.SUCCESS))


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
