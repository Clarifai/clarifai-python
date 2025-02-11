import io
import queue

import threading
from concurrent.futures import ThreadPoolExecutor

MB = 1024 * 1024


class StreamingChunksReader(io.RawIOBase):
  '''
  A buffered reader that reads data from an iterator yielding chunks of bytes, used
  to provide file-like access to a streaming data source.

  :param chunk_iterator: An iterator that yields chunks of data (bytes)
  '''

  def __init__(self, chunk_iterator):
    """
    Args:
      chunk_iterator (iterator): An iterator that yields chunks of bytes.
    """
    self._chunk_iterator = chunk_iterator
    self.response = None
    self.buffer = b''
    self.b_pos = 0
    self._eof = False

  def readable(self):
    return True

  def readinto(self, output_buf):
    if self._eof:
      return 0

    try:
      # load next chunk if necessary
      if self.b_pos == len(self.buffer):
        self.buffer = next(self._chunk_iterator)
        self.b_pos = 0

      # copy data to output buffer
      n = min(len(output_buf), len(self.buffer) - self.b_pos)
      assert n > 0

      output_buf[:n] = self.buffer[self.b_pos:self.b_pos + n]

      # advance positions
      self.b_pos += n
      assert self.b_pos <= len(self.buffer)

      return n

    except StopIteration:
      self._eof = True
      return 0


class SeekableStreamingChunksReader(io.RawIOBase):
  """
  A buffered reader that reads data from an iterator yielding chunks of bytes, used
  to provide file-like access to a streaming data source.

  This class allows supports limited seeking to positions within the stream, by buffering
  buffering chunks internally and supporting basic seek operations within the buffer.
  """

  def __init__(self, chunk_iterator, buffer_size=100 * MB):
    """
    Args:
      chunk_iterator (iterator): An iterator that yields chunks of bytes.
      buffer_size (int): Maximum buffer size in bytes before old chunks are discarded.
    """
    self._chunk_iterator = chunk_iterator
    self.buffer_size = buffer_size
    self.buffer_vec = []
    self.file_pos = 0
    self.vec_pos = 0
    self.b_pos = 0
    self._eof = False

  #### read() methods

  def readable(self):
    return True

  def readinto(self, output_buf):
    """
    Read data into the given buffer.

    Args:
      output_buf (bytearray): Buffer to read data into.

    Returns:
      int: Number of bytes read.
    """
    if self._eof:
      return 0

    assert self.vec_pos <= len(self.buffer_vec)

    try:
      # load next chunk if necessary
      if self.vec_pos == len(self.buffer_vec):
        self._load_next_chunk()

      # copy data from buffer_vec to output buffer
      n = min(len(output_buf), len(self.buffer_vec[self.vec_pos]) - self.b_pos)
      assert n > 0

      output_buf[:n] = self.buffer_vec[self.vec_pos][self.b_pos:self.b_pos + n]

      # advance positions
      self.file_pos += n
      self.b_pos += n
      assert self.b_pos <= len(self.buffer_vec[self.vec_pos])
      if self.b_pos == len(self.buffer_vec[self.vec_pos]):
        self.vec_pos += 1
        self.b_pos = 0
      return n
    except StopIteration:
      self._eof = True
      return 0

  def _load_next_chunk(self, check_bounds=True):
    self.buffer_vec.append(next(self._chunk_iterator))
    total = sum(len(chunk) for chunk in self.buffer_vec)
    while total > self.buffer_size and len(self.buffer_vec) > 1:  # keep at least the last chunk
      chunk = self.buffer_vec.pop(0)
      total -= len(chunk)
      self.vec_pos -= 1
      if check_bounds:
        assert self.vec_pos >= 0, 'current position fell outside the buffer'

  #### seek() methods (experimental)

  def seekable(self):
    return True

  def tell(self):
    return self.file_pos

  def seek(self, offset, whence=io.SEEK_SET):
    """
    Seek to a new position in the buffered stream.

    Args:
      offset (int): The offset to seek to.
      whence (int): The reference position (SEEK_SET, SEEK_CUR).
        SEEK_END is not supported.

    Returns:
      int: The new file position.

    Raises:
      ValueError: If an invalid `whence` value is provided.
      IOError: If seeking before the start of the buffer.
    """
    if whence == io.SEEK_SET:
      seek_pos = offset
    elif whence == io.SEEK_CUR:
      seek_pos = self.file_pos + offset
    elif whence == io.SEEK_END:
      raise ValueError('SEEK_END is not supported')
    else:
      raise ValueError(f"Invalid whence: {whence}")

    # set positions to start of buffer vec to begin seeking
    self.file_pos -= self.b_pos
    self.b_pos = 0
    while self.vec_pos > 0:
      self.vec_pos -= 1
      self.file_pos -= len(self.buffer_vec[self.vec_pos])

    # check if still seeking backwards off the start of the buffer
    if seek_pos < self.file_pos:
      raise IOError('seek before start of buffer')

    # seek forwards to desired position
    while self.file_pos < seek_pos:
      if self.vec_pos == len(self.buffer_vec):
        self._load_next_chunk()
      n = len(self.buffer_vec[self.vec_pos])
      if self.file_pos + n > seek_pos:
        self.b_pos = seek_pos - self.file_pos
        self.file_pos = seek_pos
        break
      self.file_pos += n
      self.vec_pos += 1

    # unset EOF flag
    self._eof = False

    return self.file_pos


def readahead(iterator, n=1, daemon=True):
  """
  Iterator wrapper that reads ahead from the underlying iterator, using a background thread.

  :Args:
    iterator (iterator): The iterator to read from.
    n (int): The maximum number of items to read ahead.
    daemon (bool): Whether the background thread should be a daemon thread.
  """
  q = queue.Queue(maxsize=n)
  _sentinel = object()

  def _read():
    for x in iterator:
      q.put(x)
    q.put(_sentinel)

  t = threading.Thread(target=_read, daemon=daemon)
  t.start()
  while True:
    x = q.get()
    if x is _sentinel:
      break
    yield x


def map(f, iterator, parallel=1):
  '''
  Apply a function to each item in an iterator, optionally using multiple threads.
  Similar to the built-in `map` function, but with support for parallel execution.
  '''
  if parallel < 1:
    return map(f, iterator)
  with ThreadPoolExecutor(max_workers=parallel) as executor:
    futures = []
    for i in range(parallel):
      futures.append(executor.submit(f, next(iterator)))
    for r in iterator:
      res = futures.pop(0).result()
      futures.append(executor.submit(f, r))  # start computing next result before yielding this one
      yield res
    for f in futures:
      yield f.result()
