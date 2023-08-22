import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Tuple

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct

from .base import ClarifaiDataset


class VisualClassificationDataset(ClarifaiDataset):

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

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

    def process_datagen_item(id):
      datagen_item = self.datagen_object[id]
      metadata = Struct()
      image_path = datagen_item.image_path
      label = datagen_item.label if isinstance(datagen_item.label,
                                               list) else [datagen_item.label]  # clarifai concept
      input_id = f"{self.dataset_id}-{self.split}-{id}" if datagen_item.id is None else f"{self.split}-{str(datagen_item.id)}"
      geo_info = datagen_item.geo_info
      metadata.update({"filename": os.path.basename(image_path), "split": self.split})

      self.all_input_ids[id] = input_id
      input_protos.append(
          self.input_object.get_input_from_file(
              input_id=input_id,
              image_file=image_path,
              dataset_id=self.dataset_id,
              labels=label,
              geo_info=geo_info,
              metadata=metadata))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualDetectionDataset(ClarifaiDataset):
  """Visual detection dataset proto class."""

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

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

    def process_datagen_item(id):
      datagen_item = self.datagen_object[id]
      metadata = Struct()
      image = datagen_item.image_path
      labels = datagen_item.classes  # list:[l1,...,ln]
      bboxes = datagen_item.bboxes  # [[xmin,ymin,xmax,ymax],...,[xmin,ymin,xmax,ymax]]
      input_id = f"{self.dataset_id}-{self.split}-{i}" if datagen_item.id is None else f"{self.split}-{str(datagen_item.id)}"
      metadata.update({"filename": os.path.basename(image), "split": self.split})
      geo_info = datagen_item.geo_info

      self.all_input_ids[id] = input_id
      input_protos.append(
          self.input_object.get_input_from_file(
              input_id=input_id,
              image_file=image,
              dataset_id=self.dataset_id,
              geo_info=geo_info,
              metadata=metadata))
      # iter over bboxes and classes
      # one id could have more than one bbox and label
      for i in range(len(bboxes)):
        annotation_protos.append(
            self.input_object.get_annotation_proto(
                input_id=input_id, label=labels[i], annotations=bboxes[i]))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualSegmentationDataset(ClarifaiDataset):
  """Visual segmentation dataset proto class."""

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

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

    def process_datagen_item(id):
      datagen_item = self.datagen_object[id]
      metadata = Struct()
      image = datagen_item.image_path
      labels = datagen_item.classes
      _polygons = datagen_item.polygons  # list of polygons: [[[x,y],...,[x,y]],...]
      input_id = f"{self.dataset_id}-{self.split}-{i}" if datagen_item.id is None else f"{self.split}-{str(datagen_item.id)}"
      metadata.update({"filename": os.path.basename(image), "split": self.split})
      geo_info = datagen_item.geo_info

      self.all_input_ids[id] = input_id
      input_protos.append(
          self.input_object.get_input_from_file(
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
              self.input_object.get_mask_proto(
                  input_id=input_id, label=labels[i], polygons=_polygon))
        except IndexError:
          continue

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos
