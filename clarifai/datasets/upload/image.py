import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Tuple, Union

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.struct_pb2 import Struct

from .base import ClarifaiDataset


class VisualClassificationDataset(ClarifaiDataset):

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

  def create_input_protos(self, image_path: str, labels: List[Union[str, int]], input_id: str,
                          dataset_id: str, geo_info: Union[List[float], None],
                          metadata: Struct) -> resources_pb2.Input:
    """Create input protos for each image, label input pair.
    Args:
      image_path: image path.
      labels: image label(s)
      input_id: unique input id
      dataset_id: Clarifai dataset id
      geo_info: image longitude, latitude info
      metadata: image metadata
    Returns:
      An input proto representing a single row input
    """
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=geo_info[0], latitude=geo_info[1])) if geo_info is not None else None

    input_proto = resources_pb2.Input(
        id=input_id,
        dataset_ids=[dataset_id],
        data=resources_pb2.Data(
            image=resources_pb2.Image(base64=open(image_path, 'rb').read(),),
            geo=geo_pb,
            concepts=[
                resources_pb2.Concept(
                  id=f"id-{''.join(_label.split(' '))}", name=_label, value=1.)\
                for _label in labels
            ],
            metadata=metadata))

    return input_proto

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
          self.create_input_protos(image_path, label, input_id, self.dataset_id, geo_info,
                                   metadata))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualDetectionDataset(ClarifaiDataset):
  """Visual detection dataset proto class."""

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

  def create_input_protos(self, image_path: str, input_id: str, dataset_id: str,
                          geo_info: Union[List[float], None],
                          metadata: Struct) -> resources_pb2.Input:
    """Create input protos for each image, label input pair.
    Args:
      image_path: file path to image
      input_id: unique input id
      dataset_id: Clarifai dataset id
      geo_info: image longitude, latitude info
      metadata: image metadata
    Returns:
      An input proto representing a single row input
    """
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=geo_info[0], latitude=geo_info[1])) if geo_info is not None else None
    input_image_proto = resources_pb2.Input(
        id=input_id,
        dataset_ids=[dataset_id],
        data=resources_pb2.Data(
            image=resources_pb2.Image(base64=open(image_path, 'rb').read(),),
            geo=geo_pb,
            metadata=metadata))

    return input_image_proto

  def create_annotation_proto(self, label: str, annotations: List, input_id: str,
                              dataset_id: str) -> resources_pb2.Annotation:
    """Create an input proto for each bounding box, label input pair.
    Args:
      label: annotation label
      annotations: a list of a single bbox's coordinates.
      input_id: unique input id
      dataset_id: Clarifai dataset id
    Returns:
      An input proto representing a single image input
    """
    input_annot_proto = resources_pb2.Annotation(
        input_id=input_id,
        data=resources_pb2.Data(regions=[
            resources_pb2.Region(
                region_info=resources_pb2.RegionInfo(bounding_box=resources_pb2.BoundingBox(
                    # Annotations ordering: [xmin, ymin, xmax, ymax]
                    # top_row must be less than bottom row
                    # left_col must be less than right col
                    top_row=annotations[1],  #y_min
                    left_col=annotations[0],  #x_min
                    bottom_row=annotations[3],  #y_max
                    right_col=annotations[2]  #x_max
                )),
                data=resources_pb2.Data(concepts=[
                    resources_pb2.Concept(
                        id=f"id-{''.join(label.split(' '))}", name=label, value=1.)
                ]))
        ]))

    return input_annot_proto

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
          self.create_input_protos(image, input_id, self.dataset_id, geo_info, metadata))
      # iter over bboxes and classes
      # one id could have more than one bbox and label
      for i in range(len(bboxes)):
        annotation_protos.append(
            self.create_annotation_proto(labels[i], bboxes[i], input_id, self.dataset_id))

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos


class VisualSegmentationDataset(ClarifaiDataset):
  """Visual segmentation dataset proto class."""

  def __init__(self, datagen_object: Iterator, dataset_id: str, split: str) -> None:
    super().__init__(datagen_object, dataset_id, split)

  def create_input_protos(self, image_path: str, input_id: str, dataset_id: str,
                          geo_info: Union[List[float], None],
                          metadata: Struct) -> resources_pb2.Input:
    """Create input protos for each image, label input pair.
    Args:
      image_path: absolute image file path
      input_id: unique input id
      dataset_id: Clarifai dataset id
      geo_info: image longitude, latitude info
      metadata: image metadata
    Returns:
      An input proto representing a single input item
    """
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=geo_info[0], latitude=geo_info[1])) if geo_info is not None else None
    input_image_proto = resources_pb2.Input(
        id=input_id,
        dataset_ids=[dataset_id],
        data=resources_pb2.Data(
            image=resources_pb2.Image(base64=open(image_path, 'rb').read(),),
            geo=geo_pb,
            metadata=metadata))

    return input_image_proto

  def create_mask_proto(self, label: str, polygons: List[List[float]], input_id: str,
                        dataset_id: str) -> resources_pb2.Annotation:
    """Create an input mask proto for an input polygon/mask and label.
    Args:
      label: image label
      polygons: Polygon x,y points iterable
      input_id: unique input id
      dataset_id: Clarifai dataset id
    Returns:
      input_mask_proto: mask proto representing a single image input
    """
    input_mask_proto = resources_pb2.Annotation(
        input_id=input_id,
        data=resources_pb2.Data(regions=[
            resources_pb2.Region(
                region_info=resources_pb2.RegionInfo(polygon=resources_pb2.Polygon(
                    points=[
                        resources_pb2.Point(
                            row=_point[1],  # row is y point
                            col=_point[0],  # col is x point
                            visibility="VISIBLE") for _point in polygons
                    ])),
                data=resources_pb2.Data(concepts=[
                    resources_pb2.Concept(
                        id=f"id-{''.join(label.split(' '))}", name=label, value=1.)
                ]))
        ]))

    return input_mask_proto

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
          self.create_input_protos(image, input_id, self.dataset_id, geo_info, metadata))

      ## Iterate over each masked image and create a proto for upload to clarifai
      ## The length of masks/polygons-list and labels must be equal
      for i, _polygon in enumerate(_polygons):
        try:
          annotation_protos.append(
              self.create_mask_proto(labels[i], _polygon, input_id, self.dataset_id))
        except IndexError:
          continue

    with ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(process_datagen_item, id) for id in batch_input_ids]
      for job in futures:
        job.result()

    return input_protos, annotation_protos