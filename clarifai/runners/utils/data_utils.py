from io import BytesIO
from typing import List

from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeEnumOption
from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeField as InputFieldProto
from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeRangeInfo
from PIL import Image

from clarifai.runners.utils.data_types import MessageData


def image_to_bytes(img: Image.Image, format="JPEG") -> bytes:
  buffered = BytesIO()
  img.save(buffered, format=format)
  img_str = buffered.getvalue()
  return img_str


def bytes_to_image(bytes_img) -> Image.Image:
  img = Image.open(BytesIO(bytes_img))
  return img


def is_openai_chat_format(messages):
  """
    Verify if the given argument follows the OpenAI chat messages format.

    Args:
        messages (list): A list of dictionaries representing chat messages.

    Returns:
        bool: True if valid, False otherwise.
    """
  if not isinstance(messages, list):
    return False

  valid_roles = {"system", "user", "assistant", "function"}

  for msg in messages:
    if not isinstance(msg, dict):
      return False
    if "role" not in msg or "content" not in msg:
      return False
    if msg["role"] not in valid_roles:
      return False

    content = msg["content"]

    # Content should be either a string (text message) or a multimodal list
    if isinstance(content, str):
      continue  # Valid text message

    elif isinstance(content, list):
      for item in content:
        if not isinstance(item, dict):
          return False
  return True


class InputField(MessageData):
  """A field that can be used to store input data."""

  def __init__(
      self,
      default=None,
      description=None,
      min_value=None,
      max_value=None,
      choices=None,
      #  is_param=True
  ):
    self.default = default
    self.description = description
    self.min_value = min_value
    self.max_value = max_value
    self.choices = choices
    # self.is_param = is_param

  def __repr__(self) -> str:
    attrs = []
    if self.default is not None:
      attrs.append(f"default={self.default!r}")
    if self.description is not None:
      attrs.append(f"description={self.description!r}")
    if self.min_value is not None:
      attrs.append(f"min_value={self.min_value!r}")
    if self.max_value is not None:
      attrs.append(f"max_value={self.max_value!r}")
    if self.choices is not None:
      attrs.append(f"choices={self.choices!r}")
    # attrs.append(f"is_param={self.is_param!r}")
    return f"InputField({', '.join(attrs)})"

  def to_proto(self, proto=None) -> InputFieldProto:
    if proto is None:
      proto = InputFieldProto()
    if self.description is not None:
      proto.description = self.description

    if self.choices is not None:
      for choice in self.choices:
        option = ModelTypeEnumOption(id=str(choice))
        proto.model_type_enum_options.append(option)

    proto.required = self.default is None

    if self.min_value is not None or self.max_value is not None:
      range_info = ModelTypeRangeInfo()
      if self.min_value is not None:
        range_info.min = float(self.min_value)
      if self.max_value is not None:
        range_info.max = float(self.max_value)
      proto.model_type_range_info.CopyFrom(range_info)
    # proto.is_param = self.is_param

    if self.default is not None:
      proto = self.set_default(proto, self.default)

    return proto

  @classmethod
  def from_proto(cls, proto):
    default = None
    if proto.HasField('default'):
      pb_value = proto.default
      if pb_value.HasField('string_value'):
        default = pb_value.string_value
        try:
          import json
          default = json.loads(default)
        except json.JSONDecodeError:
          pass
      elif pb_value.HasField('number_value'):
        default = pb_value.number_value
        if default.is_integer():
          default = int(default)
        else:
          default = float(default)
      elif pb_value.HasField('bool_value'):
        default = pb_value.bool_value

    choices = [option.id for option in proto.model_type_enum_options
              ] if proto.model_type_enum_options else None

    min_value = None
    max_value = None
    if proto.HasField('model_type_range_info'):
      min_value = proto.model_type_range_info.min
      max_value = proto.model_type_range_info.max
      if min_value.is_integer():
        min_value = int(min_value)
      if max_value.is_integer():
        max_value = int(max_value)

    return cls(
        default=default,
        description=proto.description if proto.description else None,
        min_value=min_value,
        max_value=max_value,
        choices=choices,
        # is_param=proto.is_param
    )

  @classmethod
  def set_default(cls, proto=None, default=None):
    try:
      import json
      if proto is None:
        proto = InputFieldProto()
      if default is not None:
        proto.default = json.dumps(default)
      return proto
    except Exception:
      if default is not None:
        proto.default = str(default)
      return proto
    except Exception as e:
      raise ValueError(
          f"Error setting default value of type, {type(default)} and value: {default}: {e}")

  @classmethod
  def get_default(cls, proto):
    default_str = proto.default
    default = None
    import json
    try:
      # Attempt to parse as JSON first (for complex types)
      return json.loads(default_str)
    except json.JSONDecodeError:
      pass
    # Check for boolean values stored as "True" or "False"
    if proto.type == resources_pb2.ModelTypeField.DataType.BOOL:
      try:
        default = bool(default_str)
      except ValueError:
        pass
    # Try to parse as integer
    elif proto.type == resources_pb2.ModelTypeField.DataType.INT:
      try:
        default = int(default_str)
      except ValueError:
        pass

    # Try to parse as float
    elif proto.type == resources_pb2.ModelTypeField.DataType.FLOAT:
      try:
        default = float(default_str)
      except ValueError:
        pass
    elif proto.type == resources_pb2.ModelTypeField.DataType.STR:
      default = default_str

    if default is None:
      # If all parsing fails, return the string value
      default = default_str
    return default


class DataConverter:
  """A class that can be used to convert data to and from a specific format."""

  @classmethod
  def convert_input_data_to_new_format(
      cls, data: resources_pb2.Data,
      input_fields: List[resources_pb2.ModelTypeField]) -> resources_pb2.Data:
    """Convert input data to new format."""
    new_data = resources_pb2.Data()
    for field in input_fields:
      part_data = cls._convert_field(data, field)
      if cls._is_data_set(part_data):
        # if the field is set, add it to the new data part
        part = new_data.parts.add()
        part.id = field.name
        part.data.CopyFrom(part_data)
      else:
        if field.required:
          raise ValueError(f"Field {field.name} is required but not set")
    return new_data

  @classmethod
  def _convert_field(cls, old_data: resources_pb2.Data,
                     field: resources_pb2.ModelTypeField) -> resources_pb2.Data:
    data_type = field.type
    new_data = resources_pb2.Data()
    if data_type == resources_pb2.ModelTypeField.DataType.STR:
      if old_data.HasField('text'):
        new_data.string_value = old_data.text.raw
        old_data.ClearField('text')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.IMAGE:
      if old_data.HasField('image'):
        new_data.image.CopyFrom(old_data.image)
        # Clear the old field to avoid duplication
        old_data.ClearField('image')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.VIDEO:
      if old_data.HasField('video'):
        new_data.video.CopyFrom(old_data.video)
        old_data.ClearField('video')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.BOOL:
      if old_data.bool_value is not False:
        new_data.bool_value = old_data.bool_value
        old_data.bool_value = False
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.INT:
      if old_data.int_value != 0:
        new_data.int_value = old_data.int_value
        old_data.int_value = 0
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.FLOAT:
      if old_data.float_value != 0.0:
        new_data.float_value = old_data.float_value
        old_data.float_value = 0.0
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.BYTES:
      if old_data.bytes_value != b"":
        new_data.bytes_value = old_data.bytes_value
        old_data.bytes_value = b""
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.NDARRAY:
      if old_data.HasField('ndarray'):
        new_data.ndarray.CopyFrom(old_data.ndarray)
        old_data.ClearField('ndarray')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.TEXT:
      if old_data.HasField('text'):
        new_data.text.CopyFrom(old_data.text)
        old_data.ClearField('text')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.AUDIO:
      if old_data.HasField('audio'):
        new_data.audio.CopyFrom(old_data.audio)
        old_data.ClearField('audio')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.CONCEPT:
      if old_data.concepts:
        new_data.concepts.extend(old_data.concepts)
        old_data.ClearField('concepts')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.REGION:
      if old_data.regions:
        new_data.regions.extend(old_data.regions)
        old_data.ClearField('regions')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.FRAME:
      if old_data.frames:
        new_data.frames.extend(old_data.frames)
        old_data.ClearField('frames')
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.LIST:
      if not field.type_args:
        raise ValueError("LIST type requires type_args")
      element_field = field.type_args[0]
      if element_field in (resources_pb2.ModelTypeField.DataType.CONCEPT,
                           resources_pb2.ModelTypeField.DataType.REGION,
                           resources_pb2.ModelTypeField.DataType.FRAME):
        # convert to new format
        new_data = cls._convert_field(old_data, element_field)
      return new_data
    else:
      return new_data
      # raise ValueError(f"Unsupported data type: {data_type}")

  @classmethod
  def is_old_format(cls, data: resources_pb2.Data) -> bool:
    """Check if the Data proto is in the old format (without parts)."""
    if len(data.parts) > 0:
      return False  # New format uses parts

    # Check if any singular field is set
    singular_fields = [
        'image', 'video', 'metadata', 'geo', 'text', 'audio', 'ndarray', 'int_value',
        'float_value', 'bytes_value', 'bool_value', 'string_value'
    ]
    for field in singular_fields:
      if data.HasField(field):
        return True

    # Check if any repeated field has elements
    repeated_fields = [
        'concepts', 'colors', 'clusters', 'embeddings', 'regions', 'frames', 'tracks',
        'time_segments', 'hits', 'heatmaps'
    ]
    for field in repeated_fields:
      if getattr(data, field):
        return True

    return False

  @classmethod
  def convert_output_data_to_old_format(cls, data: resources_pb2.Data) -> resources_pb2.Data:
    """Convert output data to old format."""
    old_data = resources_pb2.Data()
    part_data = data.parts[0].data
    # Handle text.raw specially (common case for text outputs)
    old_data = part_data
    if old_data.string_value:
      old_data.text.raw = old_data.string_value

    return old_data

  @classmethod
  def _is_data_set(cls, data_msg):
    # Singular message fields
    singular_fields = ["image", "video", "metadata", "geo", "text", "audio", "ndarray"]
    for field in singular_fields:
      if data_msg.HasField(field):
        return True

    # Repeated fields
    repeated_fields = [
        "concepts", "colors", "clusters", "embeddings", "regions", "frames", "tracks",
        "time_segments", "hits", "heatmaps", "parts"
    ]
    for field in repeated_fields:
      if getattr(data_msg, field):  # checks if the list is not empty
        return True

    # Scalar fields (proto3 default: 0 for numbers, empty for strings/bytes, False for bool)
    if (data_msg.int_value != 0 or data_msg.float_value != 0.0 or data_msg.bytes_value != b"" or
        data_msg.bool_value is True or data_msg.string_value != ""):
      return True

    return False
