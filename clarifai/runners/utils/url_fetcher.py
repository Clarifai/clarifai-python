import concurrent.futures

import fsspec

from clarifai.utils.logging import logger


def download_input(input):
  _download_input_data(input.data)
  if input.data.parts:
    for i in range(len(input.data.parts)):
      _download_input_data(input.data.parts[i].data)


def _download_input_data(input_data):
  """
  This function will download any urls that are not already bytes.
  """
  if input_data.image.url and not input_data.image.base64:
    # Download the image
    with fsspec.open(input_data.image.url, 'rb') as f:
      input_data.image.base64 = f.read()
  if input_data.video.url and not input_data.video.base64:
    # Download the video
    with fsspec.open(input_data.video.url, 'rb') as f:
      input_data.video.base64 = f.read()
  if input_data.audio.url and not input_data.audio.base64:
    # Download the audio
    with fsspec.open(input_data.audio.url, 'rb') as f:
      input_data.audio.base64 = f.read()
  if input_data.text.url and not input_data.text.raw:
    # Download the text
    with fsspec.open(input_data.text.url, 'r') as f:
      input_data.text.raw = f.read()


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
