import io
import unittest

from clarifai.utils.stream_utils import SeekableStreamingChunksReader, StreamingChunksReader


class TestStreamingChunksReader(unittest.TestCase):

  def setUp(self):
    self.chunks = [b'hello', b'world', b'12345']
    #self.reader = BufferStream(iter(self.chunks), buffer_size=10)
    self.reader = StreamingChunksReader(iter(self.chunks))

  def test_read(self):
    buffer = bytearray(5)
    self.assertEqual(self.reader.readinto(buffer), 5)
    self.assertEqual(buffer, b'hello')

  def test_read_file(self):
    self.assertEqual(self.reader.read(5), b'hello')

  def test_read_partial_chunk(self):
    """Test reading fewer bytes than a chunk contains, across multiple reads."""
    buffer = bytearray(3)
    self.assertEqual(self.reader.readinto(buffer), 3)
    self.assertEqual(buffer, b'hel')
    self.assertEqual(self.reader.readinto(buffer), 2)
    self.assertEqual(buffer[:2], b'lo')
    self.assertEqual(self.reader.readinto(buffer), 3)
    self.assertEqual(buffer, b'wor')

  def test_large_chunk(self):
    """Test handling a chunk larger than the buffer size."""
    large_chunk = b'a' * 20
    reader = StreamingChunksReader(iter([large_chunk]))
    buffer = bytearray(10)
    self.assertEqual(reader.readinto(buffer), 10)
    self.assertEqual(buffer, b'a' * 10)
    self.assertEqual(reader.readinto(buffer), 10)
    self.assertEqual(buffer, b'a' * 10)


class TestSeekableStreamingChunksReader(TestStreamingChunksReader):

  def setUp(self):
    self.chunks = [b'hello', b'world', b'12345']
    self.reader = SeekableStreamingChunksReader(iter(self.chunks), buffer_size=10)

  def test_interleaved_read_and_seek(self):
    """Test alternating read and seek operations."""
    buffer = bytearray(5)
    self.reader.readinto(buffer)
    self.assertEqual(buffer, b'hello')
    buffer[:] = b'xxxxx'
    self.reader.seek(0)
    self.assertEqual(self.reader.readinto(buffer), 5)
    self.assertEqual(buffer, b'hello')
    self.reader.seek(7)
    n = self.reader.readinto(buffer)
    assert 1 <= n <= len(buffer)
    self.assertEqual(buffer[:n], b''.join(self.chunks)[7:7 + n])

  def test_seek_and_tell(self):
    """Test seeking to a position and confirming it with tell()."""
    self.reader.seek(5)
    self.assertEqual(self.reader.tell(), 5)
    self.reader.seek(-2, io.SEEK_CUR)
    self.assertEqual(self.reader.tell(), 3)

  def test_seek_out_of_bounds(self):
    """Test seeking to a negative position, which should raise an IOError."""
    with self.assertRaises(IOError):
      self.reader.seek(-1)


if __name__ == '__main__':
  unittest.main()
