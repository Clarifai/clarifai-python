from collections import defaultdict
from typing import Iterator, List, Tuple

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
    self.input_ids = []
    self._all_input_protos = {}
    self._all_annotation_protos = defaultdict(list)

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

  def _extract_protos(self) -> None:
    """
    Create input image protos for each data generator item.
    """
    raise NotImplementedError()

  def get_protos(self, input_ids: List[str]
                ) -> Tuple[List[resources_pb2.Input], List[resources_pb2.Annotation]]:
    """
    Get input and annotation protos based on input_ids.
    Args:
      `input_ids`: List of input IDs to retrieve the protos for.
    Returns:
      Input and Annotation proto iterators for the specified input IDs.
    """
    input_protos = [self._all_input_protos.get(input_id) for input_id in input_ids]
    annotation_protos = []
    if len(self._all_annotation_protos) > 0:
      annotation_protos = [self._annotation_protos.get(input_id) for input_id in input_ids]
      annotation_protos = [
          ann_proto for ann_protos in annotation_protos for ann_proto in ann_protos
      ]

    return input_protos, annotation_protos


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
