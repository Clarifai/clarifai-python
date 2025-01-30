import io

import requests

MB = 1024 * 1024


class BufferStream(io.RawIOBase):
  '''
    A buffer that reads data from a chunked stream and provides a file-like interface for reading.

    :param chunk_iterator: An iterator that yields chunks of data (bytes)
    '''

  def __init__(self, chunk_iterator):
    self._chunk_iterator = chunk_iterator
    self.response = None
    self.buffer = b''
    self.file_pos = 0
    self.b_pos = 0
    self._eof = False

  #### read() methods

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
      n = min(len(output_buf), len(self.buffer - self.b_pos))
      assert n > 0

      output_buf[:n] = self.buffer[self.b_pos:self.b_pos + n]

      # advance positions
      self.b_pos += n
      assert self.b_pos <= len(self.buffer)

      return n

    except StopIteration:
      self._eof = True
      return 0


class SeekableBufferStream(io.RawIOBase):
  '''
    EXPERIMENTAL
    A buffer that reads data from a chunked stream and provides a file-like interface for reading.

    :param chunk_iterator: An iterator that yields chunks of data (bytes)
    :param buffer_size: The maximum size of the buffer in bytes
    '''

  def __init__(self, chunk_iterator, buffer_size=100 * MB):
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
    while total > self.buffer_size:
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
    #printerr(f"seek(offset={offset}, whence={('SET', 'CUR', 'END')[whence]})")
    # convert to offset from start of file stream
    if whence == io.SEEK_SET:
      seek_pos = offset
    elif whence == io.SEEK_CUR:
      seek_pos = self.file_pos + offset
    elif whence == io.SEEK_END:
      self._seek_to_end()
      seek_pos = self.file_pos + offset
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

  def _seek_to_end(self):
    try:
      # skip positions to end of the current buffer vec
      if self.b_pos > 0:
        self.file_pos += len(self.buffer_vec[self.vec_pos]) - self.b_pos
        self.vec_pos += 1
        self.b_pos = 0
      # keep loading chunks until EOF
      while True:
        while self.vec_pos < len(self.buffer_vec):
          self.file_pos += len(self.buffer_vec[self.vec_pos])
          self.vec_pos += 1
        self._load_next_chunk(check_bounds=False)
    except StopIteration:
      pass
    # advance to end of buffer vec
    while self.vec_pos < len(self.buffer_vec):
      self.file_pos += len(self.buffer_vec[self.vec_pos])
      self.vec_pos += 1


class URLStream(BufferStream):

  def __init__(self, url, chunk_size=1 * MB, buffer_size=10 * MB, requests_kwargs={}):
    self.url = url
    self.chunk_size = chunk_size
    self.response = requests.get(self.url, stream=True, **requests_kwargs)
    self.response.raise_for_status()
    super().__init__(
        self.response.iter_content(chunk_size=self.chunk_size), buffer_size=buffer_size)

  def close(self):
    super().close()
    self.response.close()
