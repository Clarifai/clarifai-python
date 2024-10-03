import concurrent.futures

import fsspec

from clarifai.utils.logging import logger


def download_input(input):
  """
  This function will download any urls that are not already bytes.
  """
  if input.data.image.url and not input.data.image.base64:
    # Download the image
    with fsspec.open(input.data.image.url, 'rb') as f:
      input.data.image.base64 = f.read()
  if input.data.video.url and not input.data.video.base64:
    # Download the video
    with fsspec.open(input.data.video.url, 'rb') as f:
      input.data.video.base64 = f.read()
  if input.data.audio.url and not input.data.audio.base64:
    # Download the audio
    with fsspec.open(input.data.audio.url, 'rb') as f:
      input.data.audio.base64 = f.read()
  if input.data.text.url and not input.data.text.raw:
    # Download the text
    with fsspec.open(input.data.text.url, 'r') as f:
      input.data.text.raw = f.read()


def ensure_urls_downloaded(request, max_threads=128):
  """
  This function will download any urls that are not already bytes and parallelize with a thread pool.
  """
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = []
    for input in request.inputs:
      futures.append(executor.submit(download_input, input))
    for future in concurrent.futures.as_completed(futures):
      try:
        future.result()
      except Exception as e:
        logger.exception(f"Error downloading input: {e}")
