from typing import Iterator, List

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct


class ClarifaiDataset:
  """
  Clarifai datasets base class.
  """

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    self.datagen_object = datagen_object
    self.dataset_id = dataset_id
    self.split = split
    self._all_input_protos = []

  def __len__(self) -> int:
    """
    Get size of all input protos
    """
    return len(self._all_input_protos)

  def _to_list(self, input_protos: Iterator) -> List:
    """
    Parse protos iterator to list.
    """
    return list(input_protos)

  def create_input_protos(self, image_path: str, label: str, input_id: str, dataset_id: str,
                          metadata: Struct) -> resources_pb2.Input:
    """
    Create input protos for each image, label input pair.
    Args:
    	`image_path`: path to image.
    	`label`: image label
    	`input_id: unique input id
    	`dataset_id`: Clarifai dataset id
    	`metadata`: input metadata
    Returns:
    	An input proto representing a single row input
    """
    raise NotImplementedError()

  def _get_input_protos(self) -> Iterator:
    """
    Create input protos for each row of the dataframe.
    Returns:
    	A list of input protos
    """
    raise NotImplementedError()


class Chunker:
  """
  Split an input sequence into small chunks.
  """

  def __init__(self, seq: List, size: int) -> None:
    self.seq = seq
    self.size = size

  def chunk(self) -> List[List]:
    """
    Chunk input sequence.
    """
    return [self.seq[pos:pos + self.size] for pos in range(0, len(self.seq), self.size)]
