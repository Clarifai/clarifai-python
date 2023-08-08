import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Tuple, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.datasets.upload.image import (VisualClassificationDataset, VisualDetectionDataset,
                                            VisualSegmentationDataset)
from clarifai.datasets.upload.text import TextClassificationDataset
from clarifai.datasets.upload.utils import load_dataloader, load_zoo_dataloader
from clarifai.utils.misc import BackoffIterator, Chunker


class Dataset(Lister, BaseClient):
  """
  Dataset is a class that provides access to Clarifai API endpoints related to Dataset information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, dataset_id: str, **kwargs):
    """Initializes an Dataset object.
    Args:
        dataset_id (str): The Dataset ID within the App to interact with.
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
    """
    self.kwargs = {**kwargs, 'id': dataset_id}
    self.dataset_info = resources_pb2.Dataset(**self.kwargs)
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self)

  def _upload_inputs(self, batch_input: List[resources_pb2.Input]) -> str:
    """Upload inputs to clarifai platform dataset.
    Args:
      batch_input: input batch protos
    Returns:
      input_job_id: Upload Input Job ID
    """
    input_job_id = uuid.uuid4().hex  # generate a unique id for this job
    response = self._grpc_request(self.STUB.PostInputs,
                                  service_pb2.PostInputsRequest(
                                      user_app_id=self.user_app_id,
                                      inputs=batch_input,
                                      inputs_add_job_id=input_job_id))
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        print(f"Post inputs failed, status: {response.inputs[0].status.details}")
      except:
        print(f"Post inputs failed, status: {response.status.details}")

    return input_job_id

  def _upload_annotations(self, batch_annot: List[resources_pb2.Annotation]
                         ) -> Union[List[resources_pb2.Annotation], List[None]]:
    """Upload image annotations to clarifai detection dataset
    Args:
      batch_annot: annot batch protos
    Returns:
      retry_upload: failed annot upload
    """
    retry_upload = []  # those that fail to upload are stored for retries
    response = self._grpc_request(
        self.STUB.PostAnnotations,
        service_pb2.PostAnnotationsRequest(user_app_id=self.user_app_id, annotations=batch_annot),
    )
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        print(f"Post annotations failed, status: {response.annotations[0].status.details}")
      except:
        print(f"Post annotations failed, status: {response.status.details}")
      finally:
        retry_upload.extend(batch_annot)

    return retry_upload

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
          executor.submit(self._upload_annotations, inp_batch) for inp_batch in annots
      ]

      for job in as_completed(annot_threads):
        result = job.result()
        if result:
          retry_annot_upload.extend(result)

    return retry_annot_upload

  def _wait_for_inputs(self, input_job_id: str) -> bool:
    """Wait for inputs to be processed. Cancel Job if timeout > 30 minutes.
    Args:
      input_job_id: Upload Input Job ID
    Returns:
      True if inputs are processed, False otherwise
    """
    backoff_iterator = BackoffIterator()
    max_retries = self.max_retires
    start_time = time.time()
    while True:
      response = self._grpc_request(
          self.STUB.GetInputsAddJob,
          service_pb2.GetInputsAddJobRequest(user_app_id=self.user_app_id, id=input_job_id),
      )
      if time.time() - start_time > 60 * 30 or max_retries == 0:  # 30 minutes timeout
        self._grpc_request(self.STUB.CancelInputsAddJob,
                           service_pb2.CancelInputsAddJobRequest(
                               user_app_id=self.user_app_id, id=input_job_id))  #Cancel Job
        return False
      if response.status.code != status_code_pb2.SUCCESS:
        max_retries -= 1
        print(f"Get input job failed, status: {response.status.details}\n")
        continue
      if response.inputs_add_job.progress.in_progress_count == 0 and response.inputs_add_job.progress.pending_count == 0:
        return True
      else:
        time.sleep(next(backoff_iterator))

  def _delete_failed_inputs(self, batch_input_ids: List[int]) -> Tuple[List[int], List[int]]:
    """Delete failed input ids from clarifai platform dataset.
    Args:
      batch_input_ids: batch input ids
    Returns:
      success_inputs: upload success input ids
      failed_inputs: upload failed input ids
    """
    success_status = status_pb2.Status(code=status_code_pb2.INPUT_DOWNLOAD_SUCCESS)
    input_ids = {self.dataset_obj.all_input_ids[id]: id for id in batch_input_ids}
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

  def _upload_inputs_annotations(
      self, batch_input_ids: List[int]) -> Tuple[List[int], List[resources_pb2.Annotation]]:
    """Uploads batch of inputs and annotations concurrently to clarifai platform dataset.
    Args:
      batch_input_ids: batch input ids
    Returns:
      failed_input_ids: failed input ids
      retry_annot_protos: failed annot protos
    """
    input_protos, _ = self.dataset_obj.get_protos(batch_input_ids)
    input_job_id = self._upload_inputs(input_protos)
    retry_annot_protos = []

    self._wait_for_inputs(input_job_id)
    success_input_ids, failed_input_ids = self._delete_failed_inputs(batch_input_ids)

    if self.task in ["visual_detection", "visual_segmentation"]:
      _, annotation_protos = self.dataset_obj.get_protos(success_input_ids)
      chunked_annotation_protos = Chunker(annotation_protos, self.chunk_size).chunk()
      retry_annot_protos.extend(self._concurrent_annot_upload(chunked_annotation_protos))

    return failed_input_ids, retry_annot_protos

  def _retry_uploads(self, failed_input_ids: List[int],
                     retry_annot_protos: List[resources_pb2.Annotation]) -> None:
    """Retry failed uploads.
    Args:
      failed_input_ids: failed input ids
      retry_annot_protos: failed annot protos
    """
    if failed_input_ids:
      self._upload_inputs_annotations(failed_input_ids)
    if retry_annot_protos:
      chunked_annotation_protos = Chunker(retry_annot_protos, self.chunk_size).chunk()
      _ = self._concurrent_annot_upload(chunked_annotation_protos)

  def _data_upload(self) -> None:
    """Uploads inputs and annotations to clarifai platform dataset."""
    input_ids = list(range(len(self.dataset_obj)))
    chunk_input_ids = Chunker(input_ids, self.chunk_size).chunk()
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      with tqdm(total=len(chunk_input_ids), desc='Uploading Dataset') as progress:
        # Submit all jobs to the executor and store the returned futures
        futures = [
            executor.submit(self._upload_inputs_annotations, batch_input_ids)
            for batch_input_ids in chunk_input_ids
        ]

        for job in as_completed(futures):
          retry_input_ids, retry_annot_protos = job.result()
          self._retry_uploads(retry_input_ids, retry_annot_protos)
          progress.update()

  def upload_dataset(self,
                     task: str,
                     split: str,
                     module_dir: str = None,
                     zoo_dataset: str = None,
                     chunk_size: int = 128) -> None:
    """Uploads a dataset to the app.
    Args:
      task: task type(text_clf, visual-classification, visual_detection, visual_segmentation, visual-captioning)
      split: split type(train, test, val)
      module_dir: path to the module directory
      zoo_dataset: name of the zoo dataset
    """
    self.chunk_size = min(128, chunk_size)  # limit max protos in a req
    self.num_workers: int = min(10, cpu_count())  #15 req/sec rate limit
    self.annot_num_workers = 4
    self.max_retires = 10
    self.task = task
    datagen_object = None

    if module_dir is None and zoo_dataset is None:
      raise Exception("One of `from_module` and `from_zoo` must be \
      specified. Both can't be None or defined at the same time.")
    elif module_dir is not None and zoo_dataset is not None:
      raise Exception("Use either of `from_module` or `from_zoo` \
      but NOT both.")
    elif module_dir is not None:
      datagen_object = load_dataloader(module_dir, split)
    else:
      datagen_object = load_zoo_dataloader(zoo_dataset, split)

    if self.task == "text_clf":
      self.dataset_obj = TextClassificationDataset(datagen_object, self.id, split)

    elif self.task == "visual_detection":
      self.dataset_obj = VisualDetectionDataset(datagen_object, self.id, split)

    elif self.task == "visual_segmentation":
      self.dataset_obj = VisualSegmentationDataset(datagen_object, self.id, split)

    else:  # visual_classification & visual_captioning
      self.dataset_obj = VisualClassificationDataset(datagen_object, self.id, split)

    self._data_upload()

  def __getattr__(self, name):
    return getattr(self.dataset_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.dataset_info, param)}" for param in init_params
        if hasattr(self.dataset_info, param)
    ]
    return f"Dataset Details: \n{', '.join(attribute_strings)}\n"
