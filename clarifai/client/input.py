import csv
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Annotation, Audio, Image, Input, Text, Video
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.utils.logging import get_logger
from clarifai.utils.misc import BackoffIterator, Chunker


class Inputs(Lister, BaseClient):
  """Inputs is a class that provides access to Clarifai API endpoints related to Input information."""

  def __init__(self, user_id: str = "", app_id: str = "", logger_level: str = "INFO", **kwargs):
    """Initializes an Input object.

    Args:
        user_id (str): A user ID for authentication.
        app_id (str): An app ID for the application to interact with.
        **kwargs: Additional keyword arguments to be passed to the Input
    """
    self.user_id = user_id
    self.app_id = app_id
    self.kwargs = {**kwargs}
    self.input_info = resources_pb2.Input(**self.kwargs)
    self.logger = get_logger(logger_level=logger_level, name=__name__)
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self)

  def _get_proto(self,
                 input_id: str,
                 dataset_id: Union[str, None],
                 imagepb: Image = None,
                 video_pb: Video = None,
                 audio_pb: Audio = None,
                 text_pb: Text = None,
                 geo_info: List = None,
                 labels: List = None,
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
            labels (list): A list of labels for the input.
            metadata (Struct): A Struct of metadata for the input.
        Returns:
            Input: An Input object for the specified input ID.
        """
    assert geo_info is None or isinstance(
        geo_info, list), "geo_info must be a list of longitude and latitude"
    assert labels is None or isinstance(labels, list), "labels must be a list of strings"
    assert metadata is None or isinstance(metadata, Struct), "metadata must be a Struct"
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=geo_info[0], latitude=geo_info[1])) if geo_info else None
    concepts=[
            resources_pb2.Concept(
            id=f"id-{''.join(_label.split(' '))}", name=_label, value=1.)\
            for _label in labels
        ]if labels else None

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

  def get_input_from_url(self,
                         input_id: str,
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
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_proto = input_obj.get_input_from_url(input_id = 'demo', image_url='https://samples.clarifai.com/metro-north.jpg')
    """
    if not any((image_url, video_url, audio_url, text_url)):
      raise ValueError(
          "At least one of image_url, video_url, audio_url, text_url must be provided.")
    image_pb = resources_pb2.Image(url=image_url) if image_url else None
    video_pb = resources_pb2.Video(url=video_url) if video_url else None
    audio_pb = resources_pb2.Audio(url=audio_url) if audio_url else None
    text_pb = resources_pb2.Text(url=text_url) if text_url else None
    return self._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        text_pb=text_pb,
        **kwargs)

  def get_input_from_file(self,
                          input_id: str,
                          image_file: str = None,
                          video_file: str = None,
                          audio_file: str = None,
                          dataset_id: str = None,
                          **kwargs) -> Input:
    """Create input proto from files.

    Args:
        input_id (str): The input ID for the input to create.
        image_file (str): The url for the image.
        video_file (str): The url for the video.
        audio_file (str): The url for the audio.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_proto = input_obj.get_input_from_file(input_id = 'demo', video_file='file_path')
    """
    if not any((image_file, video_file, audio_file)):
      raise ValueError("At least one of image_file, video_file, audio_file, must be provided.")
    image_pb = resources_pb2.Image(base64=open(image_file, 'rb').read()) if image_file else None
    video_pb = resources_pb2.Video(base64=open(video_file, 'rb').read()) if video_file else None
    audio_pb = resources_pb2.Audio(base64=open(audio_file, 'rb').read()) if audio_file else None
    return self._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        **kwargs)

  def get_input_from_bytes(self,
                           input_id: str,
                           image_bytes: bytes = None,
                           video_bytes: bytes = None,
                           audio_bytes: bytes = None,
                           dataset_id: str = None,
                           **kwargs) -> Input:
    """Create input proto from bytes.

    Args:
        input_id (str): The input ID for the input to create.
        image_bytes (str): The bytes for the image.
        video_bytes (str): The bytes for the video.
        audio_bytes (str): The bytes for the audio.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        Input: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> image = open('demo.jpg', 'rb').read()
        >>> video = open('demo.mp4', 'rb').read()
        >>> input_proto = input_obj.get_input_from_bytes(input_id = 'demo',image_bytes =image, video_bytes=video)
    """
    if not any((image_bytes, video_bytes, audio_bytes)):
      raise ValueError("At least one of image_bytes, video_bytes, audio_bytes, must be provided.")
    image_pb = resources_pb2.Image(base64=image_bytes) if image_bytes else None
    video_pb = resources_pb2.Video(base64=video_bytes) if video_bytes else None
    audio_pb = resources_pb2.Audio(base64=audio_bytes) if audio_bytes else None
    return self._get_proto(
        input_id=input_id,
        dataset_id=dataset_id,
        imagepb=image_pb,
        video_pb=video_pb,
        audio_pb=audio_pb,
        **kwargs)

  def get_image_inputs_from_folder(self,
                                   folder_path: str,
                                   dataset_id: str = None,
                                   labels: bool = False) -> List[Input]:  #image specific
    """Create input protos for image data type from folder.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        list of Input: A list of Input objects for the specified folder.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_protos = input_obj.get_image_inputs_from_folder(folder_path='demo_folder')
    """
    input_protos = []
    labels = [folder_path.split('/')[-1]] if labels else None
    for filename in os.listdir(folder_path):
      if filename.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'tiff', 'webp']:
        continue
      input_id = filename.split('.')[0]
      image_pb = resources_pb2.Image(base64=open(os.path.join(folder_path, filename), 'rb').read())
      input_protos.append(
          self._get_proto(
              input_id=input_id, dataset_id=dataset_id, imagepb=image_pb, labels=labels))
    return input_protos

  def get_text_input(self, input_id: str, raw_text: str, dataset_id: str = None,
                     **kwargs) -> Text:  #text specific
    """Create input proto for text data type from rawtext.

    Args:
        input_id (str): The input ID for the input to create.
        raw_text (str): The raw text input.
        dataset_id (str): The dataset ID for the dataset to add the input to.
        **kwargs: Additional keyword arguments to be passed to the Input

    Returns:
        Text: An Input object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_protos = input_obj.get_text_input(input_id = 'demo', raw_text = 'This is a test')
    """
    text_pb = resources_pb2.Text(raw=raw_text)
    return self._get_proto(input_id=input_id, dataset_id=dataset_id, text_pb=text_pb, **kwargs)

  def get_text_input_from_csv(self, csv_path: str, dataset_id: str = None,
                              labels: str = True) -> List[Text]:  #text specific
    """Create input proto for text data type from cscv.

    Args:
        csv_path (str): Path to the csv file.
        dataset_id (str): The dataset ID for the dataset to add the input to.
        labels (str): True if csv file has labels column.

    Returns:
        inputs: List of inputs

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_protos = input_obj.get_text_input_from_csv(csv_path = 'filepath')
    """
    input_protos = []
    with open(csv_path) as _file:
      reader = csv.reader(_file)
      next(reader, None)  # skip header
      for id, input in enumerate(reader):
        text = input[0]
        if labels:
          assert len(input) == 2, "csv file should have two columns(input, labels)"
          labels = input[1] if isinstance(input[1], list) else [input[1]]
        else:
          labels = None
        input_id = f"{dataset_id}-{id}"
        input_protos.append(
            self.get_text_input(
                input_id=input_id, raw_text=text, dataset_id=dataset_id, labels=labels))

    return input_protos

  def get_text_inputs_from_folder(self,
                                  folder_path: str,
                                  dataset_id: str = None,
                                  labels: bool = False) -> List[Text]:  #text specific
    """Create input protos for text data type from folder.

    Args:
        folder_path (str): Path to the folder containing text.

    Returns:
        list of Input: A list of Input objects for the specified folder.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_protos = input_obj.get_text_inputs_from_folder(folder_path='demo_folder')
    """
    input_protos = []
    labels = [folder_path.split('/')[-1]] if labels else None
    for filename in os.listdir(folder_path):
      if filename.split('.')[-1] != 'txt':
        continue
      input_id = filename.split('.')[0]
      text_pb = resources_pb2.Text(raw=open(os.path.join(folder_path, filename), 'rb').read())
      input_protos.append(
          self._get_proto(
              input_id=input_id, dataset_id=dataset_id, text_pb=text_pb, labels=labels))
    return input_protos

  def get_annotation_proto(self, input_id: str, label: str, annotations: List) -> Annotation:
    """Create an annotation proto for each bounding box, label input pair.

    Args:
        input_id (str): The input ID for the annotation to create.
        label (str): annotation label
        annotations (List): a list of a single bbox's coordinates. # Annotations ordering: [xmin, ymin, xmax, ymax]

    Returns:
        An annotation object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_obj.get_annotation_proto(input_id='demo', label='demo', annotations=[x_min, y_min, x_max, y_max])
    """
    if not isinstance(annotations, list):
      raise UserError("annotations must be a list of bbox cooridnates")
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

  def get_mask_proto(self, input_id: str, label: str, polygons: List[List[float]]) -> Annotation:
    """Create an annotation proto for each polygon box, label input pair.

    Args:
        input_id (str): The input ID for the annotation to create.
        label (str): annotation label
        polygons (List): Polygon x,y points iterable

    Returns:
        An annotation object for the specified input ID.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input()
        >>> input_obj.get_mask_proto(input_id='demo', label='demo', polygons=[[[x,y],...,[x,y]],...])
    """
    if not isinstance(polygons, list):
      raise UserError("polygons must be a list of points")
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
        >>> from clarifai.client.input import Input
        >>> input_obj = Input(user_id = 'user_id', app_id = 'demo_app')
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
                       dataset_id: str = None,
                       **kwargs) -> str:
    """Upload input from file.

    Args:
        input_id (str): The input ID for the input to create.
        image_file (str): The file for the image.
        video_file (str): The file for the video.
        audio_file (str): The file for the audio.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id: job id for the upload request.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input(user_id = 'user_id', app_id = 'demo_app')
        >>> input_obj.upload_from_file(input_id='demo', audio_file='demo.mp3')
    """
    input_pb = self.get_input_from_file(input_id, image_file, video_file, audio_file, dataset_id,
                                        **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_bytes(self,
                        input_id: str,
                        image_bytes: bytes = None,
                        video_bytes: bytes = None,
                        audio_bytes: bytes = None,
                        dataset_id: str = None,
                        **kwargs) -> str:
    """Upload input from bytes.

    Args:
        input_id (str): The input ID for the input to create.
        image_bytes (str): The bytes for the image.
        video_bytes (str): The bytes for the video.
        audio_bytes (str): The bytes for the audio.
        dataset_id (str): The dataset ID for the dataset to add the input to.

    Returns:
        input_job_id: job id for the upload request.

    Example:
        >>> from clarifai.client.input import Input
        >>> input_obj = Input(user_id = 'user_id', app_id = 'demo_app')
        >>> image = open('demo.jpg', 'rb').read()
        >>> input_obj.upload_from_bytes(input_id='demo', image_bytes=image)
    """
    input_pb = self.get_input_from_bytes(input_id, image_bytes, video_bytes, audio_bytes,
                                         dataset_id, **kwargs)
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
        >>> from clarifai.client.input import Input
        >>> input_obj = Input(user_id = 'user_id', app_id = 'demo_app')
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
    input_job_id = uuid.uuid4().hex  # generate a unique id for this job
    request = service_pb2.PostInputsRequest(
        user_app_id=self.user_app_id, inputs=inputs, inputs_add_job_id=input_job_id)
    response = self._grpc_request(self.STUB.PostInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        self.logger.warning(response.inputs[0].status)
      except IndexError:
        self.logger.warning(response.status)
    else:
      if show_log:
        self.logger.info("\nInputs Uploaded\n%s", response.status)

    return input_job_id

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
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        self.logger.warning(
            f"Post annotations failed, status: {response.annotations[0].status.details}")
      except:
        self.logger.warning(f"Post annotations failed, status: {response.status.details}")
      finally:
        retry_upload.extend(batch_annot)
    else:
      if show_log:
        self.logger.info("\nAnnotations Uploaded\n%s", response.status)
    return retry_upload

  def _upload_batch(self, inputs: List[Input]) -> List[Input]:
    """Upload a batch of input objects to the app.

    Args:
        inputs (List[Input]): List of input objects to upload.

    Returns:
        input_job_id: job id for the upload request.
    """
    input_job_id = self.upload_inputs(inputs, False)
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
        >>> input_obj.delete_inputs(input_obj.list_inputs())
    """
    if not isinstance(inputs, list):
      raise UserError("input_ids must be a list of input ids")
    inputs_ids = [input.id for input in inputs]
    request = service_pb2.DeleteInputsRequest(user_app_id=self.user_app_id, ids=inputs_ids)
    response = self._grpc_request(self.STUB.DeleteInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nInputs Deleted\n%s", response.status)

  def list_inputs(self) -> List[Input]:  # TODO: update lister
    """Lists all the inputs for the app.

    Returns:
        list of Input: A list of Input objects for the app.

    Example:
        >>> from clarifai.client.user import User
        >>> input_obj = User(user_id="user_id").app(app_id="app_id").inputs()
        >>> input_obj.list_inputs()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size)
    all_inputs_info = list(
        self.list_all_pages_generator(self.STUB.ListInputs, service_pb2.ListInputsRequest,
                                      request_data))
    for input_info in all_inputs_info:
      input_info['id'] = input_info.pop('input_id')
    return [resources_pb2.Input(**input_info) for input_info in all_inputs_info]

  def _bulk_upload(self, inputs: List[Input], chunk_size: int = 128) -> None:
    """Uploads process for large number of inputs.

    Args:
        inputs (List[Input]): input protos
        chunk_size (int): chunk size for each request
    """
    num_workers: int = min(10, cpu_count())  # limit max workers to 10
    chunk_size = min(128, chunk_size)  # limit max protos in a req
    chunked_inputs = Chunker(inputs, chunk_size).chunk()
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
    backoff_iterator = BackoffIterator()
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
        failed_inputs (List[Input]): failed input prots
    """
    if failed_inputs:
      self._upload_batch(failed_inputs)

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
