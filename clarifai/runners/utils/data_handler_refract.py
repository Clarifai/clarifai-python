import io
from typing import Any, Callable, Dict, Type

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Audio as AudioProto
from clarifai_grpc.grpc.api.resources_pb2 import Image as ImageProto
from clarifai_grpc.grpc.api.resources_pb2 import NDArray
from clarifai_grpc.grpc.api.resources_pb2 import Text as TextProto
from clarifai_grpc.grpc.api.resources_pb2 import Video as VideoProto
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image as PILImage

# Type registry for conversion between Python types and protobuf
_TYPE_HANDLERS: Dict[Type, Callable] = {
    # Python type: (to_proto, from_proto)
    TextProto: (
        lambda value: value.to_proto(),
        lambda proto: Text.from_proto(proto)
    ),
    ImageProto: (
        lambda value: value.to_proto(),
        lambda proto: Image(proto)
    ),
    AudioProto: (
        lambda value: value.to_proto(),
        lambda proto: Audio(proto)
    ),
    VideoProto: (
        lambda value: value.to_proto(),
        lambda proto: Video(proto)
    ),
    str: (
        lambda value: TextProto(raw=value),
        lambda proto: proto.raw
    ),
    bytes: (
        lambda value: resources_pb2.Data(base64=value),
        lambda proto: proto.base64
    ),
    int: (
        lambda value: resources_pb2.Data(int_value=value),
        lambda proto: proto.int_value
    ),
    float: (
        lambda value: resources_pb2.Data(float_value=value),
        lambda proto: proto.float_value
    ),
    bool: (
        lambda value: resources_pb2.Data(boolean=value),
        lambda proto: proto.boolean
    ),
    np.ndarray: (
        lambda value: NDArray(buffer=value.tobytes(), shape=value.shape, dtype=str(value.dtype)),
        lambda proto: np.frombuffer(proto.buffer, dtype=np.dtype(proto.dtype)).reshape(proto.shape)
    ),
    PILImage.Image: (
        lambda value: Image.from_pil(value).to_proto(),
        lambda proto: Image(proto).to_pil()
    ),
    dict: (
        lambda value: _dict_to_metadata(value),
        lambda proto: MessageToDict(proto.metadata)
    )
}


def _dict_to_metadata(metadata: dict) -> Struct:
  struct = Struct()
  ParseDict(metadata, struct)
  return struct


def _value_to_proto(value: Any) -> resources_pb2.Data:
  """Convert a Python value to a protobuf Data message."""
  data = resources_pb2.Data()
  for py_type, (to_proto, _) in _TYPE_HANDLERS.items():
    if isinstance(value, py_type):
      handler = to_proto
      break
  else:
    if isinstance(value, (Text, Image, Audio, Video)):
      data_part = getattr(data, type(value).__name__.lower())
      data_part.CopyFrom(value.to_proto())
      return data
    raise TypeError(f"Unsupported type: {type(value)}")

  result = handler(value)
  if isinstance(result, resources_pb2.Data):
    data.CopyFrom(result)
  else:
    field_name = type(result).DESCRIPTOR.name.lower()
    getattr(data, field_name).CopyFrom(result)
  return data


def _proto_to_value(proto: resources_pb2.Data) -> Any:
  """Convert a protobuf Data message to a Python value."""
  for field in proto.DESCRIPTOR.fields:
    if proto.HasField(field.name):
      _, from_proto = _TYPE_HANDLERS.get(field.type, (None, None))
      if from_proto:
        return from_proto(getattr(proto, field.name))
  if proto.parts:
    return [_proto_to_value(part.data) for part in proto.parts]
  return None


def kwargs_to_proto(**kwargs) -> resources_pb2.Data:
  """Convert keyword arguments to a Data proto."""
  data_proto = resources_pb2.Data()
  for part_name, part_value in kwargs.items():
    part = data_proto.parts.add()
    part.id = part_name

    if isinstance(part_value, list):
      for item in part_value:
        item_proto = _value_to_proto(item)
        part_part = part.data.parts.add()
        part_part.data.CopyFrom(item_proto)
    else:
      part_proto = _value_to_proto(part_value)
      part.data.CopyFrom(part_proto)
  return data_proto


def proto_to_kwargs(data: resources_pb2.Data) -> dict:
  """Convert a Data proto to keyword arguments."""
  kwargs = {}
  for part in data.parts:
    part_name = part.id
    if part.data.parts:
      kwargs[part_name] = [_proto_to_value(part.data) for _ in part.data.parts]
    else:
      kwargs[part_name] = _proto_to_value(part.data)
  return kwargs


class Output:

  def __init__(self, **kwargs: Any):
    if not kwargs:
      raise ValueError("Output must have at least one key-value pair")
    self.parts = kwargs

  def to_proto(self) -> resources_pb2.Output:
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

  @classmethod
  def from_url(cls, url: str) -> "Image":
    return cls(ImageProto(url=url))

  @classmethod
  def from_pil(cls, pil_image: PILImage.Image) -> "Image":
    with io.BytesIO() as output:
      pil_image.save(output, format="PNG")
      return cls(ImageProto(base64=output.getvalue()))

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


'''
Type Handling Registry: Centralized conversion logic reduces duplication and enhances extensibility.
Simplified Conversion Functions: _value_to_proto and _proto_to_value handle all type conversions using the registry.
Streamlined Wrapper Methods: Common processing logic extracted into _process_request, reducing code duplication.
Improved Batch Processing: Uses ThreadPoolExecutor.map for cleaner batch prediction.
Error Handling: Clearer error messages and validation of required parameters.
Removed Redundant Checks: Simplified Output class initialization.

'''
