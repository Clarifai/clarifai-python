import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Generator, List, Tuple, Type, TypeVar, Union

import requests
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.constants.dataset import DATASET_UPLOAD_TASKS
from clarifai.datasets.export.inputs_annotations import (DatasetExportReader,
                                                         InputAnnotationDownloader)
from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.datasets.upload.image import (VisualClassificationDataset, VisualDetectionDataset,
                                            VisualSegmentationDataset)
from clarifai.datasets.upload.text import TextClassificationDataset
from clarifai.datasets.upload.utils import DisplayUploadStatus
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import get_logger
from clarifai.utils.misc import BackoffIterator, Chunker

ClarifaiDatasetType = TypeVar('ClarifaiDatasetType', VisualClassificationDataset,
                              VisualDetectionDataset, VisualSegmentationDataset,
                              TextClassificationDataset)


class Dataset(Lister, BaseClient):
  """Dataset is a class that provides access to Clarifai API endpoints related to Dataset information."""

  def __init__(self,
               url: str = None,
               dataset_id: str = None,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               **kwargs):
    """Initializes a Dataset object.

    Args:
        url (str): The URL to initialize the dataset object.
        dataset_id (str): The Dataset ID within the App to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        **kwargs: Additional keyword arguments to be passed to the Dataset.
    """
    if url and dataset_id:
      raise UserError("You can only specify one of url or dataset_id.")
    if url:
      user_id, app_id, _, dataset_id, _ = ClarifaiUrlHelper.split_clarifai_url(url)
      kwargs = {'user_id': user_id, 'app_id': app_id}
    self.kwargs = {**kwargs, 'id': dataset_id}
    self.dataset_info = resources_pb2.Dataset(**self.kwargs)
    # Related to Dataset Upload
    self.num_workers: int = min(10, cpu_count())  #15 req/sec rate limit
    self.annot_num_workers = 4
    self.max_retires = 10
    self.batch_size = 128  # limit max protos in a req
    self.task = None  # Upload dataset type
    self.input_object = Inputs(user_id=self.user_id, app_id=self.app_id, pat=pat)
    self.logger = get_logger(logger_level="INFO")
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id, base=base_url, pat=pat)
    Lister.__init__(self)

  def create_version(self, **kwargs) -> 'Dataset':
    """Creates a dataset version for the Dataset.

    Args:
        **kwargs: Additional keyword arguments to be passed to Dataset Version.
          - description (str): The description of the dataset version.
          - metadata (dict): The metadata of the dataset version.

    Returns:
        Dataset: A Dataset object for the specified dataset ID.

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(dataset_id='dataset_id', user_id='user_id', app_id='app_id')
        >>> dataset_version = dataset.create_version(description='dataset_version_description')
    """
    request = service_pb2.PostDatasetVersionsRequest(
        user_app_id=self.user_app_id,
        dataset_id=self.id,
        dataset_versions=[resources_pb2.DatasetVersion(**kwargs)])

    response = self._grpc_request(self.STUB.PostDatasetVersions, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDataset Version created\n%s", response.status)
    kwargs.update({
        'dataset_id': self.id,
        'app_id': self.app_id,
        'user_id': self.user_id,
        'version': response.dataset_versions[0],
        'base_url': self.base,
        'pat': self.pat
    })
    return Dataset(**kwargs)

  def delete_version(self, version_id: str) -> None:
    """Deletes a dataset version for the Dataset.

    Args:
        version_id (str): The version ID to delete.

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(dataset_id='dataset_id', user_id='user_id', app_id='app_id')
        >>> dataset.delete_version(version_id='version_id')
    """
    request = service_pb2.DeleteDatasetVersionsRequest(
        user_app_id=self.user_app_id, dataset_id=self.id, dataset_version_ids=[version_id])

    response = self._grpc_request(self.STUB.DeleteDatasetVersions, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nDataset Version Deleted\n%s", response.status)

  def list_versions(self, page_no: int = None,
                    per_page: int = None) -> Generator['Dataset', None, None]:
    """Lists all the versions for the dataset.

    Args:
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Dataset: Dataset objects for the versions of the dataset.

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(dataset_id='dataset_id', user_id='user_id', app_id='app_id')
        >>> all_dataset_versions = list(dataset.list_versions())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(
        user_app_id=self.user_app_id,
        dataset_id=self.id,
    )
    all_dataset_versions_info = self.list_pages_generator(
        self.STUB.ListDatasetVersions,
        service_pb2.ListDatasetVersionsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for dataset_version_info in all_dataset_versions_info:
      dataset_version_info['id'] = dataset_version_info['dataset_version_id']
      del dataset_version_info['dataset_version_id']
      del dataset_version_info['metrics']
      kwargs = {
          'dataset_id': self.id,
          'app_id': self.app_id,
          'user_id': self.user_id,
          'version': resources_pb2.DatasetVersion(**dataset_version_info),
          'base_url': self.base,
          'pat': self.pat
      }
      yield Dataset(**kwargs)

  def _concurrent_annot_upload(self, annots: List[List[resources_pb2.Annotation]]
                              ) -> Union[List[resources_pb2.Annotation], List[None]]:
    """Uploads annotations concurrently.

    Args:
      annots: annot protos

    Returns:
      retry_annot_upload: All failed annot protos during upload
    """
    annot_threads = []
    retry_annot_upload = []

    with ThreadPoolExecutor(max_workers=self.annot_num_workers) as executor:  # limit annot workers
      annot_threads = [
          executor.submit(self.input_object.upload_annotations, inp_batch, False)
          for inp_batch in annots
      ]

      for job in as_completed(annot_threads):
        result = job.result()
        if result:
          retry_annot_upload.extend(result)

    return retry_annot_upload

  def _delete_failed_inputs(self, batch_input_ids: List[int],
                            dataset_obj: ClarifaiDatasetType) -> Tuple[List[int], List[int]]:
    """Delete failed input ids from clarifai platform dataset.

    Args:
      batch_input_ids: batch input ids
      dataset_obj: ClarifaiDataset object

    Returns:
      success_inputs: upload success input ids
      failed_inputs: upload failed input ids
    """
    success_status = status_pb2.Status(code=status_code_pb2.INPUT_DOWNLOAD_SUCCESS)
    input_ids = {dataset_obj.all_input_ids[id]: id for id in batch_input_ids}
    response = self._grpc_request(
        self.STUB.ListInputs,
        service_pb2.ListInputsRequest(
            ids=list(input_ids.keys()),
            per_page=len(input_ids),
            user_app_id=self.user_app_id,
            status=success_status),
    )
    response_dict = MessageToDict(response)
    success_inputs = response_dict.get('inputs', [])

    success_input_ids = [input.get('id') for input in success_inputs]
    failed_input_ids = list(set(input_ids) - set(success_input_ids))
    #delete failed inputs
    self._grpc_request(
        self.STUB.DeleteInputs,
        service_pb2.DeleteInputsRequest(user_app_id=self.user_app_id, ids=failed_input_ids),
    )
    return [input_ids[id] for id in success_input_ids], [input_ids[id] for id in failed_input_ids]

  def _upload_inputs_annotations(self, batch_input_ids: List[int], dataset_obj: ClarifaiDatasetType
                                ) -> Tuple[List[int], List[resources_pb2.Annotation]]:
    """Uploads batch of inputs and annotations concurrently to clarifai platform dataset.

    Args:
      batch_input_ids: batch input ids
      dataset_obj: ClarifaiDataset object

    Returns:
      failed_input_ids: failed input ids
      retry_annot_protos: failed annot protos
    """
    input_protos, _ = dataset_obj.get_protos(batch_input_ids)
    input_job_id = self.input_object.upload_inputs(inputs=input_protos, show_log=False)
    retry_annot_protos = []

    self.input_object._wait_for_inputs(input_job_id)
    success_input_ids, failed_input_ids = self._delete_failed_inputs(batch_input_ids, dataset_obj)

    if self.task in ["visual_detection", "visual_segmentation"]:
      _, annotation_protos = dataset_obj.get_protos(success_input_ids)
      chunked_annotation_protos = Chunker(annotation_protos, self.batch_size).chunk()
      retry_annot_protos.extend(self._concurrent_annot_upload(chunked_annotation_protos))

    return failed_input_ids, retry_annot_protos

  def _retry_uploads(self, failed_input_ids: List[int],
                     retry_annot_protos: List[resources_pb2.Annotation],
                     dataset_obj: ClarifaiDatasetType) -> None:
    """Retry failed uploads.

    Args:
      failed_input_ids: failed input ids
      retry_annot_protos: failed annot protos
      dataset_obj: ClarifaiDataset object
    """
    if failed_input_ids:
      self._upload_inputs_annotations(failed_input_ids, dataset_obj)
    if retry_annot_protos:
      chunked_annotation_protos = Chunker(retry_annot_protos, self.batch_size).chunk()
      _ = self._concurrent_annot_upload(chunked_annotation_protos)

  def _data_upload(self, dataset_obj: ClarifaiDatasetType) -> None:
    """Uploads inputs and annotations to clarifai platform dataset.

    Args:
      dataset_obj: ClarifaiDataset object
    """
    input_ids = list(range(len(dataset_obj)))
    chunk_input_ids = Chunker(input_ids, self.batch_size).chunk()
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      with tqdm(total=len(chunk_input_ids), desc='Uploading Dataset') as progress:
        # Submit all jobs to the executor and store the returned futures
        futures = [
            executor.submit(self._upload_inputs_annotations, batch_input_ids, dataset_obj)
            for batch_input_ids in chunk_input_ids
        ]

        for job in as_completed(futures):
          retry_input_ids, retry_annot_protos = job.result()
          self._retry_uploads(retry_input_ids, retry_annot_protos, dataset_obj)
          progress.update()

  def upload_dataset(self,
                     dataloader: Type[ClarifaiDataLoader],
                     batch_size: int = 32,
                     get_upload_status: bool = False) -> None:
    """Uploads a dataset to the app.

    Args:
      dataloader (Type[ClarifaiDataLoader]): ClarifaiDataLoader object
      batch_size (int): batch size for concurrent upload of inputs and annotations (max: 128)
      get_upload_status (bool): True if you want to get the upload status of the dataset
    """
    self.batch_size = min(self.batch_size, batch_size)
    self.task = dataloader.task
    if self.task not in DATASET_UPLOAD_TASKS:
      raise UserError("Task should be one of \
                      'text_classification', 'visual_classification', \
                      'visual_detection', 'visual_segmentation', 'visual_captioning'")

    if self.task == "text_classification":
      dataset_obj = TextClassificationDataset(dataloader, self.id)

    elif self.task == "visual_detection":
      dataset_obj = VisualDetectionDataset(dataloader, self.id)

    elif self.task == "visual_segmentation":
      dataset_obj = VisualSegmentationDataset(dataloader, self.id)

    else:  # visual_classification & visual_captioning
      dataset_obj = VisualClassificationDataset(dataloader, self.id)

    self._data_upload(dataset_obj)

    if get_upload_status:
      self.get_upload_status(dataloader)

  def upload_from_csv(self,
                      csv_path: str,
                      input_type: str = 'text',
                      csv_type: str = None,
                      labels: bool = True,
                      batch_size: int = 128) -> None:
    """Uploads dataset from a csv file.

    Args:
        csv_path (str): path to the csv file
        input_type (str): type of the dataset(text, image)
        csv_type (str): type of the csv file(raw, url, file_path)
        labels (bool): True if csv file has labels column
        batch_size (int): batch size for concurrent upload of inputs and annotations

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(user_id = 'user_id', app_id = 'demo_app', dataset_id = 'demo_dataset')
        >>> dataset.upload_from_csv(csv_path='csv_path', input_type='text', csv_type='raw, labels=True)

    Note:
        CSV file supports 'inputid', 'input', 'concepts', 'metadata', 'geopoints' columns.
        All the data in the CSV should be in double quotes.
        metadata should be in single quotes format. Example: "{'key': 'value'}"
        geopoints should be in "long,lat" format.
    """
    if input_type not in ['image', 'text', 'video', 'audio']:
      raise UserError('Invalid input type, it should be image,text,audio or video')
    if csv_type not in ['raw', 'url', 'file_path']:
      raise UserError('Invalid csv type, it should be raw, url or file_path')
    assert csv_path.endswith('.csv'), 'csv_path should be a csv file'
    if csv_type == 'raw' and input_type != 'text':
      raise UserError('Only text input type is supported for raw csv type')
    batch_size = min(128, batch_size)
    input_protos = self.input_object.get_inputs_from_csv(
        csv_path=csv_path,
        input_type=input_type,
        csv_type=csv_type,
        dataset_id=self.id,
        labels=labels)
    self.input_object._bulk_upload(inputs=input_protos, batch_size=batch_size)

  def upload_from_folder(self,
                         folder_path: str,
                         input_type: str,
                         labels: bool = False,
                         batch_size: int = 128) -> None:
    """Upload dataset from folder.

    Args:
        folder_path (str): Path to the folder containing images.
        input_type (str): type of the dataset(text, image)
        labels (bool): True if folder name is the label for the inputs
        batch_size (int): batch size for concurrent upload of inputs and annotations

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(user_id = 'user_id', app_id = 'demo_app', dataset_id = 'demo_dataset')
        >>> dataset.upload_from_folder(folder_path='folder_path', input_type='text', labels=True)

    Note: The filename is used as the input_id.
    """
    if input_type not in ['image', 'text']:
      raise UserError('Invalid input type it should be image or text')
    if input_type == 'image':
      input_protos = self.input_object.get_image_inputs_from_folder(
          folder_path=folder_path, dataset_id=self.id, labels=labels)
    if input_type == 'text':
      input_protos = self.input_object.get_text_inputs_from_folder(
          folder_path=folder_path, dataset_id=self.id, labels=labels)
    self.input_object._bulk_upload(inputs=input_protos, batch_size=batch_size)

  def get_upload_status(self,
                        dataloader: Type[ClarifaiDataLoader],
                        delete_version: bool = False,
                        timeout: int = 600) -> None:
    """Creates a new dataset version and displays the upload status of the dataset.

    Args:
        dataloader (Type[ClarifaiDataLoader]): ClarifaiDataLoader object
        delete_version (bool): True if you want to delete the version after getting the upload status
        timeout (int): Timeout in seconds for getting the upload status. Default is 600 seconds.

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(dataset_id='dataset_id', user_id='user_id', app_id='app_id')
        >>> dataset.get_upload_status(dataloader)

    Note:
        This is a beta feature and is subject to change.
    """
    self.logger.info("Getting dataset upload status...")
    dataset_version_id = uuid.uuid4().hex
    _ = self.create_version(id=dataset_version_id, description="SDK Upload Status")

    request_data = dict(
        user_app_id=self.user_app_id,
        dataset_id=self.id,
        dataset_version_id=dataset_version_id,
    )

    start_time = time.time()
    backoff_iterator = BackoffIterator()
    while (True):
      dataset_metrics_response = self._grpc_request(
          self.STUB.ListDatasetVersionMetricsGroups,
          service_pb2.ListDatasetVersionMetricsGroupsRequest(**request_data),
      )

      if dataset_metrics_response.status.code != status_code_pb2.SUCCESS:
        self.delete_version(dataset_version_id)
        raise Exception("Failed to get dataset metrics {}".format(dataset_metrics_response.status))

      dict_response = MessageToDict(dataset_metrics_response)
      if len(dict_response.keys()) == 1 and time.time() - start_time < timeout:
        self.logger.info("Crunching the dataset metrics. Please wait...")
        time.sleep(next(backoff_iterator))
        continue
      else:
        if time.time() - start_time > timeout:
          self.delete_version(dataset_version_id)
          raise UserError(
              "Dataset metrics are taking too long to process. Please try again later.")
        break

    dataset_info_dict = dict(user_id=self.user_id, app_id=self.app_id, dataset_id=self.id)
    DisplayUploadStatus(dataloader, dataset_metrics_response, dataset_info_dict)

    if delete_version:
      self.delete_version(dataset_version_id)

  def export(self,
             save_path: str,
             archive_url: str = None,
             local_archive_path: str = None,
             split: str = None,
             num_workers: int = 4) -> None:
    """Exports the Clarifai protobuf dataset to a local archive.

    Args:
        save_path (str): The path to save the archive to.
        archive_url (str): The URL to the Clarifai protobuf archive.
        local_archive_path (str): The path to the local Clarifai protobuf archive.
        split (str): Export dataset inputs in the directory format {split}/{input_type}. Default is all.
        num_workers (int): Number of workers to use for downloading the archive. Default is 4.

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> Dataset().export(save_path='output.zip', local_archive_path='clarifai-data-protobuf.zip')
    """
    if local_archive_path and not os.path.exists(local_archive_path):
      raise UserError(f"Archive {local_archive_path} does not exist.")
    # Create a session object and set auth header
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({'Authorization': self.metadata[0][1]})
    with DatasetExportReader(
        session=session, archive_url=archive_url, local_archive_path=local_archive_path) as reader:
      InputAnnotationDownloader(session, reader, num_workers).download_archive(
          save_path=save_path, split=split)

  def __getattr__(self, name):
    return getattr(self.dataset_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.dataset_info, param)}" for param in init_params
        if hasattr(self.dataset_info, param)
    ]
    return f"Dataset Details: \n{', '.join(attribute_strings)}\n"
