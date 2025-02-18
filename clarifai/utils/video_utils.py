import io
import os
import tempfile
import threading

import requests

from clarifai.utils import stream_utils
from clarifai.utils.misc import optional_import

av = optional_import("av", pip_package="av")


def stream_frames_from_url(url, download_ok=True):
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
    download_file(url, file.name)
    container = av.open(file.name)
  else:
    # TODO others: s3, etc.
    raise ValueError('Unsupported URL scheme')

  # Decode video frames
  yield from container.decode(video=0)


def download_file(url, file_name):
  response = requests.get(url, stream=True)
  response.raise_for_status()
  with open(file_name, 'wb') as f:
    for chunk in response.iter_content(chunk_size=1024):
      f.write(chunk)


def stream_frames_from_bytes(bytes_iterator):
  """
    Streams a video from a sequence of chunked byte strings of a streamable video
    container format.

    :param bytes_iterator: An iterator that yields byte chunks with the video data
    """
  buffer = stream_utils.StreamingChunksReader(bytes_iterator)
  reader = io.BufferedReader(buffer)
  container = av.open(reader)
  yield from container.decode(video=0)


def convert_to_streamable(filepath):
  return recontain(filepath, "mpegts", {"muxpreload": "0", "muxdelay": "0"})


def recontain(input, format, options={}):
  # pyav-only implementation of "ffmpeg -i filepath -f mpegts -muxpreload 0 -muxdelay 0 pipe:"
  read_pipe_fd, write_pipe_fd = os.pipe()
  read_pipe = os.fdopen(read_pipe_fd, "rb")
  write_pipe = os.fdopen(write_pipe_fd, "wb")

  def _run_av():
    input_container = output_container = None
    try:
      # open input and output containers, using mpegts as output format
      input_container = av.open(input, options=options)
      output_container = av.open(write_pipe, mode="w", format=format)

      # Copy streams directly without re-encoding
      for stream in input_container.streams:
        output_container.add_stream_from_template(stream)

      # Read packets from input and write them to output
      for packet in input_container.demux():
        if not packet.size:
          break
        output_container.mux(packet)

    finally:
      if output_container:
        output_container.close()
      if input_container:
        input_container.close()

  t = threading.Thread(target=_run_av)
  t.start()

  return read_pipe
