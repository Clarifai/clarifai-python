#! dataset output features (output from preprocessing & input to clarifai data proto builders)
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class TextFeatures:
  """Text classification datasets preprocessing output features."""
  text: str
  labels: List[Union[str, int]]  # List[str or int] to cater for multi-class tasks
  id: Optional[int] = None  # text_id
  metadata: Optional[dict] = None
  label_ids: Optional[List[str]] = None


@dataclass
class VisualClassificationFeatures:
  """Image classification datasets preprocessing output features."""
  image_path: str
  labels: List[Union[str, int]]  # List[str or int] to cater for multi-class tasks
  geo_info: Optional[List[float]] = None  #[Longitude, Latitude]
  id: Optional[int] = None  # image_id
  metadata: Optional[dict] = None
  image_bytes: Optional[bytes] = None
  label_ids: Optional[List[str]] = None


@dataclass
class VisualDetectionFeatures:
  """Image Detection datasets preprocessing output features."""
  image_path: str
  labels: List[Union[str, int]]
  bboxes: List[List[float]]
  geo_info: Optional[List[float]] = None  #[Longitude, Latitude]
  id: Optional[int] = None  # image_id
  metadata: Optional[dict] = None
  image_bytes: Optional[bytes] = None
  label_ids: Optional[List[str]] = None


@dataclass
class VisualSegmentationFeatures:
  """Image Segmentation datasets preprocessing output features."""
  image_path: str
  labels: List[Union[str, int]]
  polygons: List[List[List[float]]]
  geo_info: Optional[List[float]] = None  #[Longitude, Latitude]
  id: Optional[int] = None  # image_id
  metadata: Optional[dict] = None
  image_bytes: Optional[bytes] = None
  label_ids: Optional[List[str]] = None


@dataclass
class MultiModalFeatures:
  """Multi-modal datasets preprocessing output features."""
  text: str
  image_bytes: str
  labels: List[Union[str, int]] = None  # List[str or int] to cater for multi-class tasks
  id: Optional[int] = None  # image_id
  metadata: Optional[dict] = None
