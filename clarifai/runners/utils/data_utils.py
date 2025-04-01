from io import BytesIO

from PIL import Image


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
