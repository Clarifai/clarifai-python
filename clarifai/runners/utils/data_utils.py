from io import BytesIO

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

  def __init__(self,
               default=None,
               description=None,
               min_value=None,
               max_value=None,
               choices=None,
               visibility=True,
               is_param=False):
    self.default = default
    self.description = description
    self.min_value = min_value
    self.max_value = max_value
    self.choices = choices
    self.visibility = visibility
    self.is_param = is_param

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
    attrs.append(f"visibility={self.visibility!r}")
    attrs.append(f"is_param={self.is_param!r}")
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

    proto.visibility = self.visibility
    proto.is_param = self.is_param

    if self.default is not None:
      if isinstance(self.default, str) or isinstance(self.default, bool) or isinstance(
          self.default, (int, float)):
        proto.default = str(self.default)
      else:
        import json
        proto.default = json.dumps(self.default)

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
        visibility=proto.visibility,
        is_param=proto.is_param)

  @classmethod
  def set_default(cls, proto=None, default=None):

    if proto is None:
      proto = InputFieldProto()
    if default is not None:
      if isinstance(default, str) or isinstance(default, bool) or isinstance(
          default, (int, float)):
        proto.default = str(default)
      else:
        import json
        proto.default = json.dumps(default)
    return proto
