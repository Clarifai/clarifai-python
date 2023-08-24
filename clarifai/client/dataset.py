from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Tuple, TypeVar, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.datasets.upload.image import (VisualClassificationDataset, VisualDetectionDataset,
                                            VisualSegmentationDataset)
from clarifai.datasets.upload.text import TextClassificationDataset
from clarifai.datasets.upload.utils import load_dataloader, load_module_dataloader
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.misc import Chunker

ClarifaiDatasetType = TypeVar('ClarifaiDatasetType', VisualClassificationDataset,
                              VisualDetectionDataset, VisualSegmentationDataset,
                              TextClassificationDataset)


class Dataset(Lister, BaseClient):
  """Dataset is a class that provides access to Clarifai API endpoints related to Dataset information."""

  def __init__(self, url_init: str = "", dataset_id: str = "", **kwargs):
    """Initializes a Dataset object.

    Args:
        url_init (str): The URL to initialize the dataset object.
        dataset_id (str): The Dataset ID within the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    if url_init != "" and dataset_id != "":
      raise UserError("You can only specify one of url_init or dataset_id.")
    if url_init == "" and dataset_id == "":
      raise UserError("You must specify one of url_init or dataset_id.")
    if url_init != "":
      user_id, app_id, _, dataset_id, _ = ClarifaiUrlHelper.split_clarifai_url(url_init)
      kwargs = {'user_id': user_id, 'app_id': app_id}
    self.kwargs = {**kwargs, 'id': dataset_id}
    self.dataset_info = resources_pb2.Dataset(**self.kwargs)
    # Related to Dataset Upload
    self.num_workers: int = min(10, cpu_count())  #15 req/sec rate limit
    self.annot_num_workers = 4
    self.max_retires = 10
    self.chunk_size = 128  # limit max protos in a req
    self.task = None  # Upload dataset type
    self.input_object = Inputs(user_id=self.user_id, app_id=self.app_id)
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self)

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
      chunked_annotation_protos = Chunker(annotation_protos, self.chunk_size).chunk()
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
      chunked_annotation_protos = Chunker(retry_annot_protos, self.chunk_size).chunk()
      _ = self._concurrent_annot_upload(chunked_annotation_protos)

  def _data_upload(self, dataset_obj: ClarifaiDatasetType) -> None:
    """Uploads inputs and annotations to clarifai platform dataset.

    Args:
      dataset_obj: ClarifaiDataset object
    """
    input_ids = list(range(len(dataset_obj)))
    chunk_input_ids = Chunker(input_ids, self.chunk_size).chunk()
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
                     task: str,
                     split: str,
                     module_dir: str = None,
                     dataset_loader: str = None,
                     chunk_size: int = 128) -> None:
    """Uploads a dataset to the app.

    Args:
      task (str): task type(text_clf, visual-classification, visual_detection, visual_segmentation, visual-captioning)
      split (str): split type(train, test, val)
      module_dir (str): path to the module directory
      dataset_loader (str): name of the dataset loader
      chunk_size (int): chunk size for concurrent upload of inputs and annotations
    """
    self.chunk_size = min(self.chunk_size, chunk_size)
    self.task = task
    datagen_object = None

    if module_dir is None and dataset_loader is None:
      raise UserError("One of `from_module` and `dataset_loader` must be \
      specified. Both can't be None or defined at the same time.")
    elif module_dir is not None and dataset_loader is not None:
      raise UserError("Use either of `from_module` or `dataset_loader` \
      but NOT both.")
    elif module_dir is not None:
      datagen_object = load_module_dataloader(module_dir, split)
    else:
      datagen_object = load_dataloader(dataset_loader, split)

    if self.task == "text_clf":
      dataset_obj = TextClassificationDataset(datagen_object, self.id, split)

    elif self.task == "visual_detection":
      dataset_obj = VisualDetectionDataset(datagen_object, self.id, split)

    elif self.task == "visual_segmentation":
      dataset_obj = VisualSegmentationDataset(datagen_object, self.id, split)

    else:  # visual_classification & visual_captioning
      dataset_obj = VisualClassificationDataset(datagen_object, self.id, split)

    self._data_upload(dataset_obj)

  def upload_from_csv(self,
                      csv_path: str,
                      input_type: str = 'text',
                      labels: bool = True,
                      chunk_size: int = 128) -> None:
    """Uploads dataset from a csv file.

    Args:
        csv_path (str): path to the csv file
        input_type (str): type of the dataset(text, image)
        labels (bool): True if csv file has labels column
        chunk_size (int): chunk size for concurrent upload of inputs and annotations

    Example:
        >>> from clarifai.client.dataset import Dataset
        >>> dataset = Dataset(user_id = 'user_id', app_id = 'demo_app', dataset_id = 'demo_dataset')
        >>> dataset.upload_from_csv(csv_path='csv_path', labels=True)

    Note: csv file should have either one(input) or two columns(input, labels).
    """
    if input_type not in ['image', 'text']:  #TODO: add image
      raise UserError('Invalid input type it should be image or text')
    chunk_size = min(128, chunk_size)
    if input_type == 'text':
      input_protos = self.input_object.get_text_input_from_csv(
          csv_path=csv_path, dataset_id=self.id, labels=labels)
    self.input_object._bulk_upload(inputs=input_protos, chunk_size=chunk_size)

  def upload_from_folder(self,
                         folder_path: str,
                         input_type: str,
                         labels: bool = False,
                         chunk_size: int = 128) -> None:
    """Upload dataset from folder.

    Args:
        folder_path (str): Path to the folder containing images.
        input_type (str): type of the dataset(text, image)
        labels (bool): True if folder name is the label for the inputs
        chunk_size (int): chunk size for concurrent upload of inputs and annotations

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
    self.input_object._bulk_upload(inputs=input_protos, chunk_size=chunk_size)

  def __getattr__(self, name):
    return getattr(self.dataset_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.dataset_info, param)}" for param in init_params
        if hasattr(self.dataset_info, param)
    ]
    return f"Dataset Details: \n{', '.join(attribute_strings)}\n"
