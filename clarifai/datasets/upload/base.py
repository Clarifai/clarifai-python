from collections import defaultdict
from typing import Iterator, List, Tuple, TypeVar, Union

from clarifai_grpc.grpc.api import resources_pb2

from clarifai.constants.dataset import DATASET_UPLOAD_TASKS
from clarifai.datasets.upload.features import (MultiModalFeatures, TextFeatures,
                                               VisualClassificationFeatures,
                                               VisualDetectionFeatures, VisualSegmentationFeatures)

OutputFeaturesType = TypeVar(
    'OutputFeaturesType',
    bound=Union[TextFeatures, VisualClassificationFeatures, VisualDetectionFeatures,
                VisualSegmentationFeatures, MultiModalFeatures])


class ClarifaiDataset:
  """Clarifai datasets base class."""

  def __init__(self, data_generator: 'ClarifaiDataLoader', dataset_id: str,
               max_workers: int = 4) -> None:
    self.data_generator = data_generator
    self.dataset_id = dataset_id
    self.max_workers = max_workers
    self.all_input_ids = {}
    self._all_input_protos = {}
    self._all_annotation_protos = defaultdict(list)

  def __len__(self) -> int:
    """Get size of all input protos"""
    return len(self.data_generator)

  def _to_list(self, input_protos: Iterator) -> List:
    """Parse protos iterator to list."""
    return list(input_protos)

  def _extract_protos(self) -> None:
    """Create input image protos for each data generator item."""
    raise NotImplementedError()

  def get_protos(self, input_ids: List[int]
                ) -> Tuple[List[resources_pb2.Input], List[resources_pb2.Annotation]]:
    """Get input and annotation protos based on input_ids.
    Args:
      input_ids: List of input IDs to retrieve the protos for.
    Returns:
      Input and Annotation proto iterators for the specified input IDs.
    """
    input_protos, annotation_protos = self._extract_protos(input_ids)

    return input_protos, annotation_protos


class ClarifaiDataLoader:
  """Clarifai data loader base class."""

  def __init__(self) -> None:
    pass

  @property
  def task(self):
    raise NotImplementedError("Task should be one of {}".format(DATASET_UPLOAD_TASKS))

  def load_data(self) -> None:
    raise NotImplementedError()

  def __len__(self) -> int:
    raise NotImplementedError()

  def __getitem__(self, index: int) -> OutputFeaturesType:
    raise NotImplementedError()
