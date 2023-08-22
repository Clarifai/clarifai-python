from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Tuple

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct

from .base import ClarifaiDataset


class TextClassificationDataset(ClarifaiDataset):
  """Upload text classification datasets to clarifai datasets"""

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

  def _extract_protos(self, batch_input_ids: List[int]
                     ) -> Tuple[List[resources_pb2.Input], List[resources_pb2.Annotation]]:
    """Create input image and annotation protos for batch of input ids.
    Args:
      batch_input_ids: List of input IDs to retrieve the protos for.
    Returns:
      input_protos: List of input protos.
      annotation_protos: List of annotation protos.
    """
    input_protos, annotation_protos = [], []

    def process_datagen_item(id):
      datagen_item = self.datagen_object[id]
      metadata = Struct()
      text = datagen_item.text
      labels = datagen_item.labels if isinstance(
          datagen_item.labels, list) else [datagen_item.labels]  # clarifai concept
      input_id = f"{self.dataset_id}-{self.split}-{id}" if datagen_item.id is None else f"{self.split}-{str(datagen_item.id)}"
      metadata.update({"split": self.split})

      self.all_input_ids[id] = input_id
      input_protos.append(
          self.input_object.get_text_input(
              input_id=input_id,
              raw_text=text,
              dataset_id=self.dataset_id,
              labels=labels,
              metadata=metadata))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos
