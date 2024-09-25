from typing import Dict, List, Tuple, Union

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from PIL import Image

from clarifai.client.auth.helper import ClarifaiAuthHelper

from .data_utils import bytes_to_image, image_to_bytes


class BaseDataHandler:

  def __init__(self,
               proto: Union[resources_pb2.Input, resources_pb2.Output],
               auth: ClarifaiAuthHelper = None):
    self._proto = proto
    self._auth = auth

  #
  def to_python(self):
    return dict(text=self.text, image=self.image, audio=self.audio)

  # ---------------- Start get/setters ---------------- #
  # Proto
  @property
  def proto(self):
    return self._proto

  # Status
  @property
  def status(self) -> status_pb2.Status:
    return self._proto.status

  def set_status(self, code: str, description: str = ""):
    self._proto.status.code = code
    self._proto.status.description = description

  # Text
  @property
  def text(self) -> Union[None, str]:
    data = self._proto.data.text
    text = None
    if data.ByteSize():
      if data.raw:
        text = data.raw
      else:
        raise NotImplementedError
    return text

  def set_text(self, text: str):
    self._proto.data.text.raw = text

  # Image
  @property
  def image(self, format: str = "np") -> Union[None, Image.Image, np.ndarray]:
    data = self._proto.data.image
    image = None
    if data.ByteSize():
      data: resources_pb2.Image = data
      if data.base64:
        image = data.base64
      elif data.url:
        raise NotImplementedError
      image = bytes_to_image(image)
      image = image if not format == "np" else np.asarray(image).astype("uint8")
    return image

  def set_image(self, image: Union[Image.Image, np.ndarray]):
    if isinstance(image, np.ndarray):
      image = Image.fromarray(image)
    self._proto.data.image.base64 = image_to_bytes(image)

  # Audio
  @property
  def audio(self) -> bytes:
    data = self._proto.data.audio
    audio = None
    if data.ByteSize():
      if data.base64:
        audio = data.base64
    return audio

  def set_audio(self, audio: bytes):
    self._proto.data.audio.base64 = audio

  # Bboxes
  @property
  def bboxes(self, real_coord: bool = False, image_width: int = None,
             image_height: int = None) -> Tuple[List, List, List]:
    if real_coord:
      assert (image_height or image_width
             ), "image_height and image_width are required when when return real coordinates"
    xyxy = []
    scores = []
    concepts = []
    for _, each in enumerate(self._proto.data.regions):
      box = each.region_info
      score = each.value
      concept = each.data.concepts[0].id
      x1 = box.left_col
      y1 = box.top_row
      x2 = box.right_col
      y2 = box.bottom_row
      if real_coord:
        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height
      xyxy.append([x1, y1, x2, y2])
      scores.append(score)
      concepts.append(concept)

    return xyxy, scores, concepts

  def set_bboxes(self,
                 boxes: list,
                 scores: list,
                 concepts: list,
                 real_coord: bool = False,
                 image_width: int = None,
                 image_height: int = None):
    if real_coord:
      assert (image_height and
              image_width), "image_height and image_width are required when `real_coord` is set"
      bboxes = [[x[1] / image_height, x[0] / image_width, x[3] / image_height, x[2] / image_width]
                for x in boxes]  # normalize the bboxes to [0,1] and [y1 x1 y2 x2]
      bboxes = np.clip(bboxes, 0, 1.0)

    regions = []
    for ith, bbox in enumerate(bboxes):
      score = scores[ith]
      concept = concepts[ith]
      if any([each > 1.0 for each in bbox]):
        assert ValueError(
            "Box coordinates is not normalized between [0, 1]. Please set format_box to True and provide image_height and image_width to normalize"
        )
      region = resources_pb2.RegionInfo(bounding_box=resources_pb2.BoundingBox(
          top_row=bbox[0],  # y_min
          left_col=bbox[1],  # x_min
          bottom_row=bbox[2],  # y_max
          right_col=bbox[3],  # x_max
      ))
      data = resources_pb2.Data(concepts=resources_pb2.Concept(id=concept, value=score))
      regions.append(resources_pb2.Region(region_info=region, data=data))

    self._proto.data.regions = regions

  # Concepts
  @property
  def concepts(self) -> Dict[str, float]:
    con_scores = {}
    for each in self.proto.data.concepts:
      con_scores.update({each.id: each.value})
    return con_scores

  def set_concepts(self, concept_score_pairs: Dict[str, float]):
    concepts = []
    for concept, score in concept_score_pairs.items():
      con_score = resources_pb2.Concept(id=concept, name=concept, value=score)
      concepts.append(con_score)
    if concepts:
      self._proto.data.ClearField("concepts")
      for each in concepts:
        self._proto.data.concepts.append(each)

  # Embeddings
  @property
  def embeddings(self) -> List[List[float]]:
    return [each.vector for each in self.proto.data.embeddings]

  def set_embeddings(self, list_vectors: List[List[float]]):
    if list_vectors[0]:
      self._proto.data.ClearField("embeddings")
    for vec in list_vectors:
      self._proto.data.embeddings.append(
          resources_pb2.Embedding(vector=vec, num_dimensions=len(vec)))

  # ---------------- End get/setters ---------------- #

  # Constructors
  @classmethod
  def from_proto(cls, proto):
    clss = cls(proto=proto)
    return clss

  @classmethod
  def from_data(
      cls,
      status_code: int = status_code_pb2.SUCCESS,
      status_description: str = "",
      text: str = None,
      image: Union[Image.Image, np.ndarray] = None,
      audio: bytes = None,
      boxes: dict = None,
      concepts: Dict[str, float] = {},
      embeddings: List[List[float]] = [],
  ) -> 'OutputDataHandler':
    clss = cls(proto=resources_pb2.Output())
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
      clss.set_image(image)
    if text:
      clss.set_text(text)
    if audio:
      clss.set_audio(audio)
    if boxes:
      clss.set_bboxes(**boxes)
    if concepts:
      clss.set_concepts(concepts)
    if embeddings:
      clss.set_embeddings(embeddings)

    clss.set_status(code=status_code, description=status_description)
    return clss


class InputDataHandler(BaseDataHandler):

  def __init__(self,
               proto: resources_pb2.Input = resources_pb2.Input(),
               auth: ClarifaiAuthHelper = None):
    super().__init__(proto=proto, auth=auth)


class OutputDataHandler(BaseDataHandler):

  def __init__(self,
               proto: resources_pb2.Output = resources_pb2.Output(),
               auth: ClarifaiAuthHelper = None):
    super().__init__(proto=proto, auth=auth)
