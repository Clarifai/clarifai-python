import io
from typing import Any

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image as PILImage


def metadata_to_dict(data: resources_pb2.Data) -> dict:
  return MessageToDict(data.metadata)


def dict_to_metadata(data: resources_pb2.Data, metadata_dict: dict):
  struct = Struct()
  ParseDict(metadata_dict, struct)
  data.metadata.CopyFrom(struct)


def kwargs_to_proto(*args, **kwargs) -> resources_pb2.Data:
  """Converts the kwargs to a Clarifai protobuf Data message."""
  kwargs = dict(kwargs)
  if any(k.startswith("_arg_") for k in kwargs.keys()):
    raise ValueError("Keys starting with '_arg_' are reserved for positional arguments")
  for arg_i, arg in enumerate(args):
    kwargs[f"_arg_{arg_i}"] = arg
  data_proto = resources_pb2.Data()
  for part_name, part_value in kwargs.items():
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
      part.data.bytes_value = part_value
    elif isinstance(part_value, int):
      part.data.int_value = part_value
    elif isinstance(part_value, float):
      part.data.float_value = part_value
    elif isinstance(part_value, bool):
      part.data.boolean = part_value
    elif isinstance(part_value, np.ndarray):
      ndarray_proto = resources_pb2.NDArray(
          buffer=part_value.tobytes(), shape=part_value.shape, dtype=str(part_value.dtype))
      part.data.ndarray.CopyFrom(ndarray_proto)
    elif isinstance(part_value, PILImage.Image):
      image = Image.from_pil(part_value)
      part.data.image.CopyFrom(image.to_proto())
    elif isinstance(part_value, list):
      if len(part_value) == 0:
        raise ValueError("List must have at least one element")
      if isinstance(part_value[0], dict):
        raise ValueError("List of dictionaries is not supported")
      else:
        for part_item in part_value:
          if isinstance(part_item, Text):
            part.data.parts.add().data.text.CopyFrom(part_item.to_proto())
          elif isinstance(part_item, Image):
            part.data.parts.add().data.image.CopyFrom(part_item.to_proto())
          elif isinstance(part_item, Audio):
            part.data.parts.add().data.audio.CopyFrom(part_item.to_proto())
          elif isinstance(part_item, Video):
            part.data.parts.add().data.video.CopyFrom(part_item.to_proto())
          elif isinstance(part_item, str):
            part.data.parts.add().data.text.raw = part_item
          elif isinstance(part_item, bytes):
            part.data.parts.add().data.bytes_value = part_item
          elif isinstance(part_item, int):
            part.data.parts.add().data.int_value = part_item
          elif isinstance(part_item, float):
            part.data.parts.add().data.float_value = part_item
          elif isinstance(part_item, bool):
            part.data.parts.add().data.boolean = part_item
          elif isinstance(part_item, np.ndarray):
            ndarray_proto = resources_pb2.NDArray(
                buffer=part_item.tobytes(), shape=part_item.shape, dtype=str(part_item.dtype))
            part.data.parts.add().data.ndarray.CopyFrom(ndarray_proto)
          elif isinstance(part_item, PILImage.Image):
            image = Image.from_pil(part_item)
            part.data.parts.add().data.image.CopyFrom(image.to_proto())
          else:
            raise TypeError(f"Unsupported Output type {type(part_item)} for part '{part_name}'")
    elif isinstance(part_value, dict):
      dict_to_metadata(part.data, part_value)
    else:
      raise TypeError(f"Unsupported Output type {type(part_value)} for part '{part_name}'")

  return data_proto


def proto_to_kwargs(data: resources_pb2.Data) -> dict:
  """Converts the Clarifai protobuf Data message to a dictionary."""
  kwargs = {}
  part_names = [part.id for part in data.parts]
  assert "return" not in part_names, "The key 'return' is reserved"
  for part_name in part_names + ["return"]:
    part_data = part.data if part_name != "return" else data
    if part_data.HasField("text"):
      kwargs[part_name] = Text.from_proto(part_data.text)
    elif part_data.HasField("image"):
      kwargs[part_name] = Image(part_data.image)
    elif part_data.HasField("audio"):
      kwargs[part_name] = Audio(part_data.audio)
    elif part_data.HasField("video"):
      kwargs[part_name] = Video(part_data.video)
    elif part_data.HasField("bytes_value"):
      kwargs[part_name] = part_data.bytes_value
    elif part_data.HasField("int_value"):
      kwargs[part_name] = part_data.int_value
    elif part_data.HasField("float_value"):
      kwargs[part_name] = part_data.float_value
    elif part_data.HasField("boolean"):
      kwargs[part_name] = part_data.boolean
    elif part_data.HasField("ndarray"):
      ndarray = part_data.ndarray
      kwargs[part_name] = np.frombuffer(
          ndarray.buffer, dtype=np.dtype(ndarray.dtype)).reshape(ndarray.shape)
    elif part_data.HasField("metadata"):
      kwargs[part_name] = metadata_to_dict(part_data)
    elif part_data.parts:
      kwargs[part_name] = []
      for part_item in part_data.parts:
        if part_item.HasField("text"):
          kwargs[part_name].append(Text.from_proto(part_item.text))
        elif part_item.HasField("image"):
          kwargs[part_name].append(Image(part_item.image))
        elif part_item.HasField("audio"):
          kwargs[part_name].append(Audio(part_item.audio))
        elif part_item.HasField("video"):
          kwargs[part_name].append(Video(part_item.video))
        elif part_item.HasField("bytes_value"):
          kwargs[part_name].append(part_item.bytes_value)
        elif part_item.HasField("int_value"):
          kwargs[part_name].append(part_item.int_value)
        elif part_item.HasField("float_value"):
          kwargs[part_name].append(part_item.float_value)
        elif part_item.HasField("boolean"):
          kwargs[part_name].append(part_item.boolean)
        elif part_item.HasField("ndarray"):
          ndarray = part_item.ndarray
          kwargs[part_name].append(
              np.frombuffer(ndarray.buffer, dtype=np.dtype(ndarray.dtype)).reshape(ndarray.shape))
        elif part_item.HasField("metadata"):
          raise ValueError("Metadata in list is not supported")
    else:
      raise ValueError(f"Unknown part data: {part_data}")

  args = [kwargs.pop(f"_arg_{i}") for i in range(len(kwargs)) if f"_arg_{i}" in kwargs]
  return args, kwargs


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
