from typing import Iterator, List

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct
from tqdm import tqdm

from .base import ClarifaiDataset


class TextClassificationDataset(ClarifaiDataset):
  """
  Upload text classification datasets to clarifai datasets
  """

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)
    self._extract_protos()

  def create_input_protos(self, text_input: str, labels: List[str], input_id: str, dataset_id: str,
                          metadata: Struct) -> resources_pb2.Input:
    """
    Create input protos for each text, label input pairs.
    Args:
    	`text_input`: text string.
    	`labels`: text labels
    	`input_id: unique input id
    	`dataset_id`: Clarifai dataset id
    	`metadata`:input metadata
    Returns:
    	An input proto representing a single row input
    """
    input_proto = resources_pb2.Input(
        id=input_id,
        dataset_ids=[dataset_id],
        data=resources_pb2.Data(
            text=resources_pb2.Text(raw=text_input),
            concepts=[
                resources_pb2.Concept(
                    id=f"id-{''.join(_label.split(' '))}", name=_label, value=1.)
                for _label in labels
            ],
            metadata=metadata))

    return input_proto

  def _extract_protos(self) -> None:
    """
    Creates input protos for each data generator item.
    """
    for i, item in tqdm(enumerate(self.datagen_object), desc="Loading text data"):
      metadata = Struct()
      text = item.text
      labels = item.labels if isinstance(item.labels, list) else [item.labels]  # clarifai concept
      input_id = f"{self.dataset_id}-{self.split}-{i}" if item.id is None else f"{self.split}-{str(item.id)}"
      metadata.update({"split": self.split})

      self.input_ids.append(input_id)
      input_proto = self.create_input_protos(text, labels, input_id, self.dataset_id, metadata)

      self._all_input_protos[input_id] = input_proto
