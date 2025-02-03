import csv
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Generator, List, Union

import requests
from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Annotation, Audio, Image, Input, Text, Video
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.constants.dataset import MAX_RETRIES
from clarifai.constants.input import MAX_UPLOAD_BATCH_SIZE
from clarifai.errors import UserError
from clarifai.utils.logging import logger
from clarifai.utils.misc import BackoffIterator, Chunker, clean_input_id


class Inputs(Lister, BaseClient):
  """Inputs is a class that provides access to Clarifai API endpoints related to Input information."""

  def __init__(self,
               user_id: str = None,
               app_id: str = None,
               logger_level: str = "INFO",
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes an Input object.

    Args:
        user_id (str): A user ID for authentication.
        app_id (str): An app ID for the application to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the Input
    """
    self.user_id = user_id
    self.app_id = app_id
    self.kwargs = {**kwargs}
    self.input_info = resources_pb2.Input(**self.kwargs)
    self.logger = logger
    BaseClient.__init__(
        self,
        user_id=self.user_id,
        app_id=self.app_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  @staticmethod
  def _get_proto(input_id: str,
                 dataset_id: str = None,
                 imagepb: Image = None,
                 video_pb: Video = None,
                 audio_pb: Audio = None,
                 text_pb: Text = None,
                 geo_info: List = None,
                 labels: List = None,
                 label_ids: List = None,
                 metadata: Struct = None) -> Input:
    """Create input proto for image data type.
        Args:
            input_id (str): The input ID for the input to create.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            imagepb (Image): The image proto to be used for the input.
            video_pb (Video): The video proto to be used for the input.
            audio_pb (Audio): The audio proto to be used for the input.
            text_pb (Text): The text proto to be used for the input.
            geo_info (list): A list of longitude and latitude for the geo point.
            labels (list): A list of label names for the input.
            label_ids (list): A list of label ids for the input.
            metadata (Struct): A Struct of metadata for the input.
        Returns:
            Input: An Input object for the specified input ID.
        """
    assert geo_info is None or isinstance(
        geo_info, list), "geo_info must be a list of longitude and latitude"
    assert labels is None or isinstance(labels, list), "labels must be a list of strings"
    assert label_ids is None or isinstance(label_ids, list), "label_ids must be a list of strings"
    assert metadata is None or isinstance(metadata, Struct), "metadata must be a Struct"
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=geo_info[0], latitude=geo_info[1])) if geo_info else None
    if labels:
      if not label_ids:
        concepts=[
            resources_pb2.Concept(
            id=_label, name=_label, value=1.)\
            for _label in labels
        ]
      else:
        assert len(labels) == len(label_ids), "labels and label_ids must be of the same length"
        concepts=[
            resources_pb2.Concept(
            id=label_id, name=_label, value=1.)\
            for label_id, _label in zip(label_ids, labels)
        ]
    else:
      concepts = None

    if dataset_id:
      return resources_pb2.Input(
          id=input_id,
          dataset_ids=[dataset_id],
          data=resources_pb2.Data(
              image=imagepb,
              video=video_pb,
              audio=audio_pb,
              text=text_pb,
              geo=geo_pb,
              concepts=concepts,
              metadata=metadata))

    return resources_pb2.Input(
        id=input_id,
        data=resources_pb2.Data(
            image=imagepb,
            video=video_pb,
            audio=audio_pb,
            text=text_pb,
            geo=geo_pb,
            concepts=concepts,
            metadata=metadata))

  @staticmethod
  def get_input_from_url(input_id: str,
                         image_url: str = None,
                         video_url: str = None,
                         audio_url: str = None,
                         text_url: str = None,
                         dataset_id: str = None,
                         **kwargs) -> Input:
    """Create input proto from url.

    Args:
        input_id (str): The input ID for the input to create.
        image_url (str): The url for the image.
        video_url (str): The url for the video.
        audio_url (str): The url for the audio.
        text_url (str): The url for the text.
            dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_proto = Inputs.get_input_from_url(input_id = 'demo', image_url='https://samples.clarifai.com/metro-north.jpg')
    """
    if not any((image_url, video_url, audio_url, text_url)):
      raise ValueError(
          "At least one of image_url, video_url, audio_url, text_url must be provided.")
    image_pb = resources_pb2.Image(url=image_url) if image_url else None
    video_pb = resources_pb2.Video(url=video_url) if video_url else None
    audio_pb = resources_pb2.Audio(url=audio_url) if audio_url else None
    text_pb = resources_pb2.Text(url=text_url) if text_url else None
    return Inputs._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        text_pb=text_pb,
        **kwargs)

  @staticmethod
  def get_input_from_file(input_id: str,
                          image_file: str = None,
                          video_file: str = None,
                          audio_file: str = None,
                          text_file: str = None,
                          dataset_id: str = None,
                          **kwargs) -> Input:
    """Create input proto from files.

    Args:
        input_id (str): The input ID for the input to create.
        image_file (str): The file_path for the image.
        video_file (str): The file_path for the video.
        audio_file (str): The file_path for the audio.
        text_file (str): The file_path for the text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_proto = Inputs.get_input_from_file(input_id = 'demo', video_file='file_path')
    """
    if not any((image_file, video_file, audio_file, text_file)):
      raise ValueError(
          "At least one of image_file, video_file, audio_file, text_file must be provided.")
    image_pb = resources_pb2.Image(base64=open(image_file, 'rb').read()) if image_file else None
    video_pb = resources_pb2.Video(base64=open(video_file, 'rb').read()) if video_file else None
    audio_pb = resources_pb2.Audio(base64=open(audio_file, 'rb').read()) if audio_file else None
    text_pb = resources_pb2.Text(raw=open(text_file, 'rb').read()) if text_file else None
    return Inputs._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        text_pb=text_pb,
        **kwargs)

  @staticmethod
  def get_input_from_bytes(input_id: str,
                           image_bytes: bytes = None,
                           video_bytes: bytes = None,
                           audio_bytes: bytes = None,
                           text_bytes: bytes = None,
                           dataset_id: str = None,
                           **kwargs) -> Input:
    """Create input proto from bytes.

    Args:
        input_id (str): The input ID for the input to create.
        image_bytes (str): The bytes for the image.
        video_bytes (str): The bytes for the video.
        audio_bytes (str): The bytes for the audio.
        text_bytes (str): The bytes for the text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> image = open('demo.jpg', 'rb').read()
        >>> video = open('demo.mp4', 'rb').read()
        >>> input_proto = Inputs.get_input_from_bytes(input_id = 'demo',image_bytes =image, video_bytes=video)
    """
    if not any((image_bytes, video_bytes, audio_bytes, text_bytes)):
      raise ValueError(
          "At least one of image_bytes, video_bytes, audio_bytes, text_bytes must be provided.")
    image_pb = resources_pb2.Image(base64=image_bytes) if image_bytes else None
    video_pb = resources_pb2.Video(base64=video_bytes) if video_bytes else None
    audio_pb = resources_pb2.Audio(base64=audio_bytes) if audio_bytes else None
    text_pb = resources_pb2.Text(raw=text_bytes) if text_bytes else None
    return Inputs._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        text_pb=text_pb,
        **kwargs)

  @staticmethod
  def get_image_inputs_from_folder(folder_path: str, dataset_id: str = None,
                                   labels: bool = False) -> List[Input]:  #image specific
    """Create input protos for image data type from folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list of Input: A list of Input objects for the specified folder.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_protos = Inputs.get_image_inputs_from_folder(folder_path='demo_folder')
    """
    input_protos = []
    labels = [folder_path.split('/')[-1]] if labels else None
    for filename in os.listdir(folder_path):
      if filename.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'tiff', 'webp']:
        continue
      input_id = clean_input_id(filename.split('.')[0])
      image_pb = resources_pb2.Image(base64=open(os.path.join(folder_path, filename), 'rb').read())
      input_protos.append(
          Inputs._get_proto(
              input_id=input_id, dataset_id=dataset_id, imagepb=image_pb, labels=labels))
    return input_protos

  @staticmethod
  def get_text_input(input_id: str, raw_text: str, dataset_id: str = None,
                     **kwargs) -> Text:  #text specific
    """Create input proto for text data type from raw text.

    Args:
        input_id (str): The input ID for the input to create.
        raw_text (str): The raw text input.
        dataset_id (str): The dataset ID for the dataset to add the input to.
        **kwargs: Additional keyword arguments to be passed to the Input

    Returns:
        Text: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_protos = Inputs.get_text_input(input_id = 'demo', raw_text = 'This is a test')
    """
    text_pb = resources_pb2.Text(raw=raw_text)
    return Inputs._get_proto(input_id=input_id, dataset_id=dataset_id, text_pb=text_pb, **kwargs)

  @staticmethod
  def get_multimodal_input(input_id: str,
                           raw_text: str = None,
                           text_bytes: bytes = None,
                           image_url: str = None,
                           image_bytes: bytes = None,
                           dataset_id: str = None,
                           **kwargs) -> Text:
    """Create input proto for text and image from bytes or url.

    Args:
        input_id (str): The input ID for the input to create.
        raw_text (str): The raw text input.
        text_bytes (str): The bytes for the text.
        image_url (str): The url for the image.
        image_bytes (str): The bytes for the image.
        dataset_id (str): The dataset ID for the dataset to add the input to.
        **kwargs: Additional keyword arguments to be passed to the Input

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_protos = Inputs.get_multimodal_input(input_id = 'demo', raw_text = 'What time of day is it?', image_url='https://samples.clarifai.com/metro-north.jpg')
    """
    if (image_bytes and image_url) or (not image_bytes and not image_url):
      return UserError("Please supply only one of image_bytes or image_url, and not both.")
    if (text_bytes and raw_text) or (not text_bytes and not raw_text):
      return UserError("Please supply only one of text_bytes or raw_text, and not both.")

    image_pb = resources_pb2.Image(base64=image_bytes) if image_bytes else resources_pb2.Image(
        url=image_url) if image_url else None
    text_pb = resources_pb2.Text(raw=text_bytes) if text_bytes else resources_pb2.Text(
        raw=raw_text) if raw_text else None
    return Inputs._get_proto(
        input_id=input_id, dataset_id=dataset_id, imagepb=image_pb, text_pb=text_pb, **kwargs)

  @staticmethod
  def get_inputs_from_csv(csv_path: str,
                          input_type: str = 'text',
                          csv_type: str = 'raw',
                          dataset_id: str = None,
                          labels: str = True) -> List[Text]:
    """Create input protos from csv.

    Args:
        csv_path (str): Path to the csv file.
        input_type (str): Type of input. Options: 'text', 'image', 'video', 'audio'.
        csv_type (str): Type of csv file. Options: 'raw', 'url', 'file_path'.
        dataset_id (str): The dataset ID for the dataset to add the input to.
        labels (str): True if csv file has labels column.

    Returns:
        inputs: List of inputs

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_protos = Inputs.get_inputs_from_csv(csv_path='filepath', input_type='text', csv_type='raw')
    """
    input_protos = []
    with open(csv_path) as _file:
      reader = csv.DictReader(_file, delimiter=',', quotechar='"')
      columns = reader.fieldnames
      for column in columns:
        if column not in ['inputid', 'input', 'concepts', 'metadata', 'geopoints']:
          raise UserError(
              "CSV file may have 'inputid', 'input', 'concepts', 'metadata', 'geopoints' columns. Does not support '{}' column".
              format(column))
      for id, input in enumerate(reader):
        if labels:
          labels_list = input['concepts'].split(',')
          labels = labels_list if len(input['concepts']) > 0 else None
        else:
          labels = None

        if 'metadata' in columns:
          if len(input['metadata']) > 0:
            metadata_str = input['metadata'].replace("'", '"')
            try:
              metadata_dict = json.loads(metadata_str)
            except json.decoder.JSONDecodeError:
              raise UserError("metadata column in CSV file should be a valid json")
            metadata = Struct()
            metadata.update(metadata_dict)
          else:
            metadata = None
        else:
          metadata = None

        if 'geopoints' in columns:
          if len(input['geopoints']) > 0:
            geo_points = input['geopoints'].split(',')
            geo_points = [float(geo_point) for geo_point in geo_points]
            geo_info = geo_points if len(geo_points) == 2 else UserError(
                "geopoints column in CSV file should have longitude,latitude")
          else:
            geo_info = None
        else:
          geo_info = None

        input_id = input['inputid'] if 'inputid' in columns else uuid.uuid4().hex
        text = input['input'] if input_type == 'text' else None
        image = input['input'] if input_type == 'image' else None
        video = input['input'] if input_type == 'video' else None
        audio = input['input'] if input_type == 'audio' else None

        if csv_type == 'raw':
          input_protos.append(
              Inputs.get_text_input(
                  input_id=input_id,
                  raw_text=text,
                  dataset_id=dataset_id,
                  labels=labels,
                  metadata=metadata,
                  geo_info=geo_info))
        elif csv_type == 'url':
          input_protos.append(
              Inputs.get_input_from_url(
                  input_id=input_id,
                  image_url=image,
                  text_url=text,
                  audio_url=audio,
                  video_url=video,
                  dataset_id=dataset_id,
                  labels=labels,
                  metadata=metadata,
                  geo_info=geo_info))
        else:
          input_protos.append(
              Inputs.get_input_from_file(
                  input_id=input_id,
                  image_file=image,
                  text_file=text,
                  audio_file=audio,
                  video_file=video,
                  dataset_id=dataset_id,
                  labels=labels,
                  metadata=metadata,
                  geo_info=geo_info))

    return input_protos

  @staticmethod
  def get_text_inputs_from_folder(folder_path: str, dataset_id: str = None,
                                  labels: bool = False) -> List[Text]:  #text specific
    """Create input protos for text data type from folder.

    Args:
        folder_path (str): Path to the folder containing text.

    Returns:
        list of Input: A list of Input objects for the specified folder.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_protos = Inputs.get_text_inputs_from_folder(folder_path='demo_folder')
    """
    input_protos = []
    labels = [folder_path.split('/')[-1]] if labels else None
    for filename in os.listdir(folder_path):
      if filename.split('.')[-1] != 'txt':
        continue
      input_id = clean_input_id(filename.split('.')[0])
      text_pb = resources_pb2.Text(raw=open(os.path.join(folder_path, filename), 'rb').read())
      input_protos.append(
          Inputs._get_proto(
              input_id=input_id, dataset_id=dataset_id, text_pb=text_pb, labels=labels))
    return input_protos

  @staticmethod
  def get_bbox_proto(input_id: str,
                     label: str,
                     bbox: List,
                     label_id: str = None,
                     annot_id: str = None) -> Annotation:
    """Create an annotation proto for each bounding box, label input pair.

    Args:
        input_id (str): The input ID for the annotation to create.
        label (str): annotation label name
        bbox (List): a list of a single bbox's coordinates. # bbox ordering: [xmin, ymin, xmax, ymax]
        label_id (str): annotation label ID
        annot_id (str): annotation ID

    Returns:
        An annotation object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> Inputs.get_bbox_proto(input_id='demo', label='demo', bbox=[x_min, y_min, x_max, y_max])
    """
    if not isinstance(bbox, list):
      raise UserError("must be a list of bbox cooridnates")
    annot_data = resources_pb2.Data(regions=[
        resources_pb2.Region(
            region_info=resources_pb2.RegionInfo(bounding_box=resources_pb2.BoundingBox(
                # bbox ordering: [xmin, ymin, xmax, ymax]
                # top_row must be less than bottom row
                # left_col must be less than right col
                top_row=bbox[1],  #y_min
                left_col=bbox[0],  #x_min
                bottom_row=bbox[3],  #y_max
                right_col=bbox[2]  #x_max
            )),
            data=resources_pb2.Data(concepts=[
                resources_pb2.Concept(id=label, name=label, value=1.)
                if not label_id else resources_pb2.Concept(id=label_id, name=label, value=1.)
            ]))
    ])
    if annot_id:
      input_annot_proto = resources_pb2.Annotation(id=annot_id, input_id=input_id, data=annot_data)
    else:
      input_annot_proto = resources_pb2.Annotation(input_id=input_id, data=annot_data)

    return input_annot_proto

  @staticmethod
  def get_mask_proto(input_id: str,
                     label: str,
                     polygons: List[List[float]],
                     label_id: str = None,
                     annot_id: str = None) -> Annotation:
    """Create an annotation proto for each polygon box, label input pair.

    Args:
        input_id (str): The input ID for the annotation to create.
        label (str): annotation label name
        polygons (List): Polygon x,y points iterable
        label_id (str): annotation label ID
        annot_id (str): annotation ID

    Returns:
        An annotation object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> Inputs.get_mask_proto(input_id='demo', label='demo', polygons=[[[x,y],...,[x,y]],...])
    """
    if not isinstance(polygons, list):
      raise UserError("polygons must be a list of points")
    annot_data = resources_pb2.Data(regions=[
        resources_pb2.Region(
            region_info=resources_pb2.RegionInfo(polygon=resources_pb2.Polygon(
                points=[
                    resources_pb2.Point(
                        row=_point[1],  # row is y point
                        col=_point[0],  # col is x point
                        visibility="VISIBLE") for _point in polygons
                ])),
            data=resources_pb2.Data(concepts=[
                resources_pb2.Concept(id=label, name=label, value=1.)
                if not label_id else resources_pb2.Concept(id=label_id, name=label, value=1.)
            ]))
    ])
    if annot_id:
      input_mask_proto = resources_pb2.Annotation(id=annot_id, input_id=input_id, data=annot_data)
    else:
      input_mask_proto = resources_pb2.Annotation(input_id=input_id, data=annot_data)

    return input_mask_proto

  def get_input(self, input_id: str) -> Input:
    """Get Input object of input with input_id provided from the app.

    Args:
        input_id (str): The input ID for the annotation to get.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_obj = Inputs(user_id = 'user_id', app_id = 'demo_app')
        >>> input_obj.get_input(input_id='demo')
    """
    request = service_pb2.GetInputRequest(user_app_id=self.user_app_id, input_id=input_id)
    response = self._grpc_request(self.STUB.GetInput, request)
    return response.input

  def upload_from_url(self,
                      input_id: str,
                      image_url: str = None,
                      video_url: str = None,
                      audio_url: str = None,
                      text_url: str = None,
                      dataset_id: str = None,
                      **kwargs) -> str:
    """Upload input from url.

    Args:
        input_id (str): The input ID for the input to create.
        image_url (str): The url for the image.
        video_url (str): The url for the video.
        audio_url (str): The url for the audio.
        text_url (str): The url for the text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id: job id for the upload request.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_obj = Inputs(user_id = 'user_id', app_id = 'demo_app')
        >>> input_obj.upload_from_url(input_id='demo', image_url='https://samples.clarifai.com/metro-north.jpg')
    """
    input_pb = self.get_input_from_url(input_id, image_url, video_url, audio_url, text_url,
                                       dataset_id, **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_file(self,
                       input_id: str,
                       image_file: str = None,
                       video_file: str = None,
                       audio_file: str = None,
                       text_file: str = None,
                       dataset_id: str = None,
                       **kwargs) -> str:
    """Upload input from file.

    Args:
        input_id (str): The input ID for the input to create.
        image_file (str): The file for the image.
        video_file (str): The file for the video.
        audio_file (str): The file for the audio.
        text_file (str): The file for the text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id: job id for the upload request.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_obj = Inputs(user_id = 'user_id', app_id = 'demo_app')
        >>> input_obj.upload_from_file(input_id='demo', audio_file='demo.mp3')
    """
    input_pb = self.get_input_from_file(input_id, image_file, video_file, audio_file, text_file,
                                        dataset_id, **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_bytes(self,
                        input_id: str,
                        image_bytes: bytes = None,
                        video_bytes: bytes = None,
                        audio_bytes: bytes = None,
                        text_bytes: bytes = None,
                        dataset_id: str = None,
                        **kwargs) -> str:
    """Upload input from bytes.

    Args:
        input_id (str): The input ID for the input to create.
        image_bytes (str): The bytes for the image.
        video_bytes (str): The bytes for the video.
        audio_bytes (str): The bytes for the audio.
        text_bytes (str): The bytes for the text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id: job id for the upload request.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_obj = Inputs(user_id = 'user_id', app_id = 'demo_app')
        >>> image = open('demo.jpg', 'rb').read()
        >>> input_obj.upload_from_bytes(input_id='demo', image_bytes=image)
    """
    input_pb = self.get_input_from_bytes(input_id, image_bytes, video_bytes, audio_bytes,
                                         text_bytes, dataset_id, **kwargs)
    return self.upload_inputs([input_pb])

  def upload_text(self, input_id: str, raw_text: str, dataset_id: str = None,
                  **kwargs) -> str:  #text specific
    """Upload text from raw text.

    Args:
        input_id (str): The input ID for the input to create.
        raw_text (str): The raw text.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id (str): job id for the upload request.

    Example:
        >>> from clarifai.client.input import Inputs
        >>> input_obj = Inputs(user_id = 'user_id', app_id = 'demo_app')
        >>> input_obj.upload_text(input_id = 'demo', raw_text = 'This is a test')
    """
    input_pb = self._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        text_pb=resources_pb2.Text(raw=raw_text),
        **kwargs)
    return self.upload_inputs([input_pb])

  def upload_inputs(self, inputs: List[Input], show_log: bool = True) -> str:
    """Upload list of input objects to the app.

    Args:
        inputs (list): List of input objects to upload.
        show_log (bool): Show upload status log.

    Returns:
        input_job_id: job id for the upload request.
    """
    if not isinstance(inputs, list):
      raise UserError("inputs must be a list of Input objects")
    if len(inputs) > MAX_UPLOAD_BATCH_SIZE:
      raise UserError(
          f"Number of inputs to upload exceeds the maximum batch size of {MAX_UPLOAD_BATCH_SIZE}. Please reduce batch size."
      )
    input_job_id = uuid.uuid4().hex  # generate a unique id for this job
    request = service_pb2.PostInputsRequest(
        user_app_id=self.user_app_id, inputs=inputs, inputs_add_job_id=input_job_id)
    response = self._grpc_request(self.STUB.PostInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      if show_log:
        self.logger.warning(response)
      else:
        return input_job_id, response
    else:
      if show_log:
        self.logger.info("\nInputs Uploaded\n%s", response.status)

    return input_job_id, response

  def patch_inputs(self, inputs: List[Input], action: str = 'merge') -> None:
    """Patch list of input objects to the app.

    Args:
        inputs (list): List of input objects to upload.
        action (str): Action to perform on the input. Options: 'merge', 'overwrite', 'remove'.

    Returns:
        response: Response from the grpc request.
    """
    if not isinstance(inputs, list):
      raise UserError("inputs must be a list of Input objects")
    request = service_pb2.PatchInputsRequest(
        user_app_id=self.user_app_id, inputs=inputs, action=action)
    response = self._grpc_request(self.STUB.PatchInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        self.logger.warning(f"Patch inputs failed, status: {response.annotations[0].status}")
      except Exception:
        self.logger.warning(f"Patch inputs failed, status: {response.status}")
    else:
      self.logger.info("\nPatch Inputs Successful\n%s", response.status)

  def upload_annotations(self, batch_annot: List[resources_pb2.Annotation], show_log: bool = True
                        ) -> Union[List[resources_pb2.Annotation], List[None]]:
    """Upload image annotations to app.

    Args:
        batch_annot: annot batch protos

    Returns:
        retry_upload: failed annot upload
    """
    retry_upload = []  # those that fail to upload are stored for retries
    request = service_pb2.PostAnnotationsRequest(
        user_app_id=self.user_app_id, annotations=batch_annot)
    response = self._grpc_request(self.STUB.PostAnnotations, request)
    response_dict = MessageToDict(response)
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        for annot in response_dict["annotations"]:
          if annot['status']['code'] != status_code_pb2.ANNOTATION_SUCCESS:
            self.logger.warning(f"Post annotations failed, status: {annot['status']}")
      except Exception:
        self.logger.warning(f"Post annotations failed due to {response.status}")
      finally:
        retry_upload.extend(batch_annot)
    else:
      if show_log:
        self.logger.info("\nAnnotations Uploaded\n%s", response.status)

    return retry_upload

  def patch_annotations(self, batch_annot: List[resources_pb2.Annotation],
                        action: str = 'merge') -> None:
    """Patch image annotations to app.

    Args:
        batch_annot: annot batch protos
        action (str): Action to perform on the input. Options: 'merge', 'overwrite', 'remove'.

    """
    if not isinstance(batch_annot, list):
      raise UserError("batch_annot must be a list of Annotation objects")
    request = service_pb2.PatchAnnotationsRequest(
        user_app_id=self.user_app_id, annotations=batch_annot, action=action)
    response = self._grpc_request(self.STUB.PatchAnnotations, request)
    response_dict = MessageToDict(response)
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        for annot in response_dict["annotations"]:
          if annot['status']['code'] != status_code_pb2.ANNOTATION_SUCCESS:
            self.logger.warning(f"Patch annotations failed, status: {annot['status']}")
      except Exception:
        self.logger.warning(f"Patch annotations failed due to {response.status}")
    else:
      self.logger.info("\nPatch Annotations Uploaded Successful\n%s", response.status)

  def patch_concepts(self,
                     concept_ids: List[str],
                     labels: List[str] = [],
                     values: List[float] = [],
                     action: str = 'overwrite') -> None:
    """Patch concepts to app.

    Args:
        concept_ids:  A list of concept
        labels: A list of label names
        values: concept value
        action (str): Action to perform on the input. Options: 'overwrite'.

    """
    if not labels:
      labels = list(concept_ids)
    if values:
      concepts=[
              resources_pb2.Concept(
              id=concept_id, name=label, value=value)\
              for concept_id, label, value in zip(concept_ids, labels, values)
          ]
    else:
      concepts=[
              resources_pb2.Concept(
              id=concept_id, name=label, value=1.)\
              for concept_id, label in zip(concept_ids, labels)
          ]
    request = service_pb2.PatchConceptsRequest(
        user_app_id=self.user_app_id, concepts=concepts, action=action)
    response = self._grpc_request(self.STUB.PatchConcepts, request)
    if response.status.code != status_code_pb2.SUCCESS:
      self.logger.warning(f"Patch Concepts failed, status: {response.status.details}")
    else:
      self.logger.info("\nPatch Concepts Successful\n%s", response.status)

  def _upload_batch(self, inputs: List[Input]) -> List[Input]:
    """Upload a batch of input objects to the app.

    Args:
        inputs (List[Input]): List of input objects to upload.

    Returns:
        input_job_id: job id for the upload request.
    """
    input_job_id, _ = self.upload_inputs(inputs, False)
    self._wait_for_inputs(input_job_id)
    failed_inputs = self._delete_failed_inputs(inputs)

    return failed_inputs

  def delete_inputs(self, inputs: List[Input]) -> None:
    """Delete list of input objects from the app.

    Args:
        input_ids (Input): List of input objects to delete.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> input_obj.delete_inputs(list(input_obj.list_inputs()))
    """
    if not isinstance(inputs, list):
      raise UserError("input_ids must be a list of input ids")
    inputs_ids = [input.id for input in inputs]
    request = service_pb2.DeleteInputsRequest(user_app_id=self.user_app_id, ids=inputs_ids)
    response = self._grpc_request(self.STUB.DeleteInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nInputs Deleted\n%s", response.status)

  def delete_annotations(self, input_ids: List[str], annotation_ids: List[str] = []) -> None:
    """Delete list of annotations of input objects from the app.

    Args:
        input_ids (Input): List of input objects for which annotations to delete.
        annotation_ids (List[str]): List of annotation ids to delete.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> input_obj.delete_annotations(input_ids=['input_id_1', 'input_id_2'])

    Note:
        'annotation_ids' are optional but if the are provided, the number and order in
        'annotation_ids' and 'input_ids' should match
    """
    if not isinstance(input_ids, list):
      raise UserError("input_ids must be a list of input ids")
    if annotation_ids and len(input_ids) != len(annotation_ids):
      raise UserError("Number of provided annotation_ids and input_ids should match.")
    request = service_pb2.DeleteAnnotationsRequest(
        user_app_id=self.user_app_id, ids=annotation_ids, input_ids=input_ids)
    response = self._grpc_request(self.STUB.DeleteAnnotations, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nAnnotations Deleted\n%s", response.status)

  def download_inputs(self, inputs: List[Input]) -> List[bytes]:
    """Download list of input objects from the app.

    Args:
        input_ids (Input): List of input objects to download.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> input_obj.download_inputs(list(input_obj.list_inputs()))
    """
    if not isinstance(inputs, list):
      raise UserError("input_ids must be a list of input ids")
    final_inputs = []
    #initiate session
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({'Authorization': self.metadata[0][1]})
    # download inputs
    data_types = ['image', 'video', 'audio', 'text']
    for input in inputs:
      for data_type in data_types:
        url = getattr(input.data, data_type).url
        if url:
          response = session.get(url, stream=True)
          if response.status_code == 200:
            final_inputs.append(response.content)

    return final_inputs

  def list_inputs(self,
                  dataset_id: str = None,
                  page_no: int = None,
                  per_page: int = None,
                  input_type: str = None) -> Generator[Input, None, None]:
    """Lists all the inputs for the app.

    Args:
        dataset_id (str): The dataset ID for the dataset to list inputs from.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.
        input_type (str): The type of input to list. Options: 'image', 'video', 'audio', 'text'.

    Yields:
        Input: Input objects for the app.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> all_inputs = list(input_obj.list_inputs(input_type='image'))

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    if input_type not in ['image', 'text', 'video', 'audio', None]:
      raise UserError('Invalid input type, it should be image,text,audio or video')
    if dataset_id:
      request_data = dict(user_app_id=self.user_app_id, dataset_id=dataset_id)
      all_inputs_info = self.list_pages_generator(
          self.STUB.ListDatasetInputs,
          service_pb2.ListDatasetInputsRequest,
          request_data,
          per_page=per_page,
          page_no=page_no)
    else:
      request_data = dict(user_app_id=self.user_app_id)
      all_inputs_info = self.list_pages_generator(
          self.STUB.ListInputs,
          service_pb2.ListInputsRequest,
          request_data,
          per_page=per_page,
          page_no=page_no)
    for input_info in all_inputs_info:
      input_info['id'] = input_info.pop('dataset_input_id') if dataset_id else input_info.pop(
          'input_id')
      if input_type:
        if input_type not in input_info['data'].keys():
          continue
      yield resources_pb2.Input(**input_info)

  def list_annotations(self,
                       batch_input: List[Input] = None,
                       page_no: int = None,
                       per_page: int = None) -> Generator[Annotation, None, None]:
    """Lists all the annotations for the app.

    Args:
        batch_input (List[Input]): The input objects to list annotations from.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Annotation: Annotation objects for the app.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> all_inputs = list(input_obj.list_inputs(input_type='image'))
        >>> all_annotations = list(input_obj.list_annotations(batch_input=all_inputs))

    Note:
        If batch_input is not given, then lists all the annotations for the app.
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(
        user_app_id=self.user_app_id,
        input_ids=[input.id for input in batch_input] if batch_input else None)
    all_annotations_info = self.list_pages_generator(
        self.STUB.ListAnnotations,
        service_pb2.ListAnnotationsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)
    for annotations_info in all_annotations_info:
      annotations_info['id'] = annotations_info.pop('annotation_id')
      yield Annotation(**annotations_info)

  def _bulk_upload(self, inputs: List[Input], batch_size: int = 128) -> None:
    """Uploads process for large number of inputs.

    Args:
        inputs (List[Input]): input protos
        batch_size (int): batch size for each request
    """
    num_workers: int = min(10, cpu_count())  # limit max workers to 10
    batch_size = min(128, batch_size)  # limit max protos in a req
    chunked_inputs = Chunker(inputs, batch_size).chunk()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
      with tqdm(total=len(chunked_inputs), desc='Uploading inputs') as progress:
        # Submit all jobs to the executor and store the returned futures
        futures = [
            executor.submit(self._upload_batch, batch_input_ids)
            for batch_input_ids in chunked_inputs
        ]

        for job in as_completed(futures):
          retry_input_proto = job.result()
          self._retry_uploads(retry_input_proto)
          progress.update()

  def _wait_for_inputs(self, input_job_id: str) -> bool:
    """Wait for inputs to be processed. Cancel Job if timeout > 30 minutes.

    Args:
        input_job_id (str): Upload Input Job ID

    Returns:
        True if inputs are processed, False otherwise
    """
    backoff_iterator = BackoffIterator(10)
    max_retries = 10
    start_time = time.time()
    while True:
      request = service_pb2.GetInputsAddJobRequest(user_app_id=self.user_app_id, id=input_job_id)
      response = self._grpc_request(self.STUB.GetInputsAddJob, request)

      if time.time() - start_time > 60 * 30 or max_retries == 0:  # 30 minutes timeout
        self._grpc_request(self.STUB.CancelInputsAddJob,
                           service_pb2.CancelInputsAddJobRequest(
                               user_app_id=self.user_app_id, id=input_job_id))  #Cancel Job
        return False
      if response.status.code != status_code_pb2.SUCCESS:
        max_retries -= 1
        self.logger.warning(f"Get input job failed, status: {response.status.details}\n")
        continue
      if response.inputs_add_job.progress.in_progress_count == 0 and response.inputs_add_job.progress.pending_count == 0:
        return True
      else:
        time.sleep(next(backoff_iterator))

  def _retry_uploads(self, failed_inputs: List[Input]) -> None:
    """Retry failed uploads.

    Args:
        failed_inputs (List[Input]): failed input protos
    """
    for _retry in range(MAX_RETRIES):
      if failed_inputs:
        self.logger.info(f"Retrying upload for {len(failed_inputs)} Failed inputs..\n")
        failed_inputs = self._upload_batch(failed_inputs)

    self.logger.warning(f"Failed to upload {len(failed_inputs)} inputs..\n ")

  def _delete_failed_inputs(self, inputs: List[Input]) -> List[Input]:
    """Delete failed input ids from clarifai platform dataset.

    Args:
        inputs (List[Input]): batch input protos

    Returns:
        failed_inputs: failed inputs
    """
    input_ids = [input.id for input in inputs]
    success_status = status_pb2.Status(code=status_code_pb2.INPUT_DOWNLOAD_SUCCESS)
    request = service_pb2.ListInputsRequest(
        ids=input_ids,
        per_page=len(input_ids),
        user_app_id=self.user_app_id,
        status=success_status)
    response = self._grpc_request(self.STUB.ListInputs, request)
    response_dict = MessageToDict(response)
    success_inputs = response_dict.get('inputs', [])

    success_input_ids = [input.get('id') for input in success_inputs]
    failed_inputs = [input for input in inputs if input.id not in success_input_ids]
    #delete failed inputs
    self._grpc_request(self.STUB.DeleteInputs,
                       service_pb2.DeleteInputsRequest(
                           user_app_id=self.user_app_id, ids=[input.id
                                                              for input in failed_inputs]))

    return failed_inputs

  def __getattr__(self, name):
    return getattr(self.input_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.input_info, param)}" for param in init_params
        if hasattr(self.input_info, param)
    ]
    return f"Input Details: \n{', '.join(attribute_strings)}\n"
