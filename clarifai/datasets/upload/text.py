from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Type

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct

from clarifai.client.input import Inputs

from .base import ClarifaiDataLoader, ClarifaiDataset


class TextClassificationDataset(ClarifaiDataset):
  """Upload text classification datasets to clarifai datasets"""

  def __init__(self, data_generator: Type[ClarifaiDataLoader], dataset_id: str) -> None:
    super().__init__(data_generator, dataset_id)

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

    def process_data_item(id):
      data_item = self.data_generator[id]
      metadata = Struct()
      text = data_item.text
      labels = data_item.labels if isinstance(data_item.labels,
                                              list) else [data_item.labels]  # clarifai concept
      input_id = f"{self.dataset_id}-{id}" if data_item.id is None else f"{self.dataset_id}-{str(data_item.id)}"
      if data_item.metadata is not None:
        metadata.update(data_item.metadata)

      self.all_input_ids[id] = input_id
      input_protos.append(
          Inputs.get_text_input(
              input_id=input_id,
              raw_text=text,
              dataset_id=self.dataset_id,
              labels=labels,
              metadata=metadata))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_data_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos
