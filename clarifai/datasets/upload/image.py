import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Type

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct

from clarifai.client.input import Inputs
from clarifai.datasets.upload.base import ClarifaiDataLoader, ClarifaiDataset


class VisualClassificationDataset(ClarifaiDataset):

  def __init__(self, data_generator: Type[ClarifaiDataLoader], dataset_id: str) -> None:
    super().__init__(data_generator, dataset_id)

  def _extract_protos(self, batch_input_ids: List[str]
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
      image_path = data_item.image_path
      labels = data_item.labels if isinstance(data_item.labels,
                                              list) else [data_item.labels]  # clarifai concept
      input_id = f"{self.dataset_id}-{id}" if data_item.id is None else f"{self.dataset_id}-{str(data_item.id)}"
      geo_info = data_item.geo_info
      if data_item.metadata is not None:
        metadata.update(data_item.metadata)
      else:
        metadata.update({"filename": os.path.basename(image_path)})

      self.all_input_ids[id] = input_id
      input_protos.append(
          Inputs.get_input_from_file(
              input_id=input_id,
              image_file=image_path,
              dataset_id=self.dataset_id,
              labels=labels,
              geo_info=geo_info,
              metadata=metadata))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_data_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualDetectionDataset(ClarifaiDataset):
  """Visual detection dataset proto class."""

  def __init__(self, data_generator: Type[ClarifaiDataLoader], dataset_id: str) -> None:
    super().__init__(data_generator, dataset_id)

  def _extract_protos(self, batch_input_ids: List[int]
                     ) -> Tuple[List[resources_pb2.Input], List[resources_pb2.Annotation]]:
    """Create input image protos for each data generator item.
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
      image = data_item.image_path
      labels = data_item.labels  # list:[l1,...,ln]
      bboxes = data_item.bboxes  # [[xmin,ymin,xmax,ymax],...,[xmin,ymin,xmax,ymax]]
      input_id = f"{self.dataset_id}-{id}" if data_item.id is None else f"{self.dataset_id}-{str(data_item.id)}"
      if data_item.metadata is not None:
        metadata.update(data_item.metadata)
      else:
        metadata.update({"filename": os.path.basename(image)})
      geo_info = data_item.geo_info

      self.all_input_ids[id] = input_id
      input_protos.append(
          Inputs.get_input_from_file(
              input_id=input_id,
              image_file=image,
              dataset_id=self.dataset_id,
              geo_info=geo_info,
              metadata=metadata))
      # iter over bboxes and labels
      # one id could have more than one bbox and label
      for i in range(len(bboxes)):
        annotation_protos.append(
            Inputs.get_bbox_proto(input_id=input_id, label=labels[i], bbox=bboxes[i]))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_data_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualSegmentationDataset(ClarifaiDataset):
  """Visual segmentation dataset proto class."""

  def __init__(self, data_generator: Type[ClarifaiDataLoader], dataset_id: str) -> None:
    super().__init__(data_generator, dataset_id)

  def _extract_protos(self, batch_input_ids: List[str]
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
      image = data_item.image_path
      labels = data_item.labels
      _polygons = data_item.polygons  # list of polygons: [[[x,y],...,[x,y]],...]
      input_id = f"{self.dataset_id}-{id}" if data_item.id is None else f"{self.dataset_id}-{str(data_item.id)}"
      if data_item.metadata is not None:
        metadata.update(data_item.metadata)
      else:
        metadata.update({"filename": os.path.basename(image)})
      geo_info = data_item.geo_info

      self.all_input_ids[id] = input_id
      input_protos.append(
          Inputs.get_input_from_file(
              input_id=input_id,
              image_file=image,
              dataset_id=self.dataset_id,
              geo_info=geo_info,
              metadata=metadata))

      ## Iterate over each masked image and create a proto for upload to clarifai
      ## The length of masks/polygons-list and labels must be equal
      for i, _polygon in enumerate(_polygons):
        try:
          annotation_protos.append(
              Inputs.get_mask_proto(input_id=input_id, label=labels[i], polygons=_polygon))
        except IndexError:
          continue

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_data_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos
