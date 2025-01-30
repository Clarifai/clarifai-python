import io
import tempfile

import av
import requests

from clarifai.runners.utils import stream_utils


def stream_video_from_url(url, download_ok=True):
  """
    Streams a video at the specified resolution using PyAV.

    :param url: The video URL
    :param download_ok: Whether to download the video if the URL is not a stream
    """
  protocol = url.split('://', 1)[0]
  if protocol == 'rtsp':
    # stream from RTSP and send to PyAV
    container = av.open(url)
  elif protocol in ('http', 'https'):
    if not download_ok:
      raise ValueError('Download not allowed for URL scheme')
    # download the video to the temporary file
    # TODO: download just enough to get the file header and stream to pyav if possible,
    # otherwise download the whole file
    # e.g. if linking to a streamable file format like mpegts (not mp4)
    file = tempfile.NamedTemporaryFile(delete=True)
    _download_video(url, file)
    container = av.open(file.name)
  else:
    # TODO others: s3, etc.
    raise ValueError('Unsupported URL scheme')

  # Decode video frames
  yield from container.decode(video=0)


def _download_video(url, file):
  response = requests.get(url, stream=True)
  response.raise_for_status()
  for chunk in response.iter_content(chunk_size=1024):
    file.write(chunk)
  file.flush()
  return file


def stream_video_from_bytes(bytes_iterator):
  """
    Streams a video from a sequence of chunked byte strings of a streamable video
    container format.

    :param bytes_iterator: An iterator that yields byte chunks with the video data
    """
  buffer = stream_utils.BufferStream(bytes_iterator)
  reader = io.BufferedReader(buffer)
  container = av.open(reader)
  yield from container.decode(video=0)
