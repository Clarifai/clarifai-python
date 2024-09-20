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
