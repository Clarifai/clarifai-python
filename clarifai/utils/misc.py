from typing import List


class Chunker:
  """Split an input sequence into small chunks."""

  def __init__(self, seq: List, size: int) -> None:
    self.seq = seq
    self.size = size

  def chunk(self) -> List[List]:
    """Chunk input sequence."""
    return [self.seq[pos:pos + self.size] for pos in range(0, len(self.seq), self.size)]


class BackoffIterator:
  """Iterator that returns a sequence of backoff values."""

  def __init__(self):
    self.count = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.count < 1:
      self.count += 1
      return 0.1
    elif self.count < 7:
      self.count += 1
      return 0.01 * (2**(self.count + 4))
    else:
      return 0.01 * (2**10)  # 10 seconds
