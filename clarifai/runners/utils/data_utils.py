from io import BytesIO
from typing import Dict, List

from PIL import Image as PILImage
from clarifai.runners.utils.data_types import Audio, Image, Video



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