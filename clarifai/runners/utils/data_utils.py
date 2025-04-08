from io import BytesIO
from typing import Dict, List

from PIL import Image as PILImage

from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeEnumOption
from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeField as InputFieldProto
from clarifai_grpc.grpc.api.resources_pb2 import ModelTypeRangeInfo

from clarifai.runners.utils.data_types import MessageData, Audio, Image, Video



def image_to_bytes(img: PILImage.Image, format="JPEG") -> bytes:
  buffered = BytesIO()
  img.save(buffered, format=format)
  img_str = buffered.getvalue()
  return img_str


def bytes_to_image(bytes_img) -> PILImage.Image:
  img = PILImage.open(BytesIO(bytes_img))
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

def build_openai_chat_format(prompt: str,
                             image: Image,
                             images: List[Image],
                             audio: Audio,
                             audios: List[Audio],
                             video: Video,
                             videos: List[Video],
                             messages: List[Dict]) -> List[Dict]:
  """
  Construct OpenAI-compatible messages from input components.
    Args:
        prompt (str): The prompt text.
        image (Image): Clarifai Image object.
        images (List[Image]): List of Clarifai Image objects.
        audio (Audio): Clarifai Audio object.
        audios (List[Audio]): List of Clarifai Audio objects.
        video (Video): Clarifai Video object.
        videos (List[Video]): List of Clarifai Video objects.
        messages (List[Dict]): List of chat messages.
    Returns:
        List[Dict]: Formatted chat messages.
  """
  
  openai_messages = []
  # Add previous conversation history
  if messages:
    openai_messages.extend(messages)

  content = []
  if prompt.strip():
    # Build content array for current message
    content.append({'type': 'text', 'text': prompt})
  # Add single image if present
  if image:
    content.append(_process_image(image))
  # Add multiple images if present
  if images:
    for img in images:
      content.append(_process_image(img))
  # Add single audio if present
  if audio:
    content.append(_process_audio(audio))
  # Add multiple audios if present
  if audios:
    for audio in audios:
      content.append(_process_audio(audio))
  # Add single video if present
  if video:
    content.append(_process_video(video))
  # Add multiple videos if present
  if videos:
    for video in videos:
      content.append(_process_video(video))

  if content:
    # Append complete user message
    openai_messages.append({'role': 'user', 'content': content})

  return openai_messages
  
def _process_image(image: Image) -> Dict:
  """Convert Clarifai Image object to OpenAI image format."""
  
  if image.bytes:
    b64_img = image.to_base64_str()
    return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64_img}"}}
  elif image.url:
    return {'type': 'image_url', 'image_url': {'url': image.url}}
  else:
    raise ValueError("Image must contain either bytes or URL")

def _process_audio(audio: Audio) -> Dict:
  """Convert Clarifai Audio object to OpenAI audio format."""
  
  if audio.bytes:
    audio = audio.to_base64_str()
    audio = {
        "type": "input_audio",
        "input_audio": {
            "data": audio,
            "format": "wav"
        },
    }
  elif audio.url:
    audio = audio.url
    audio = {
        "type": "audio_url",
        "audio_url": {
            "url": audio
        },
    }
  else:
    raise ValueError("Audio must contain either bytes or URL")

  return audio

def _process_video(video: Video) -> Dict:
  """Convert Clarifai Video object to OpenAI video format."""
  
  if video.bytes:
    video = "data:video/mp4;base64," + \
        video.to_base64_str()
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  elif video.url:
    video = video.url
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  else:
    raise ValueError("Video must contain either bytes or URL")

  return video

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