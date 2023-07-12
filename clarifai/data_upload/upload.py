#! Clarifai data upload

import importlib
import inspect
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Iterator, List, Optional, Tuple, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from clarifai.client import create_stub
from clarifai.data_upload.datasets.base import Chunker
from clarifai.data_upload.datasets.image import (VisualClassificationDataset,
                                                 VisualDetectionDataset, VisualSegmentationDataset)
from clarifai.data_upload.datasets.text import TextClassificationDataset


def load_dataset(module_dir: Union[str, os.PathLike], split: str) -> Iterator:
  """
  Validate and import dataset module data generator.
  Args:
    `module_dir`: relative path to the module directory
    The directory must contain a `dataset.py` script and the data itself.
    `split`: "train" or "val"/"test" dataset split
  Module Directory Structure:
  ---------------------------
      <folder_name>/
      ├──__init__.py
      ├──<Your local dir dataset>/
      └──dataset.py
  dataset.py must implement a class named following the convention,
  <dataset_name>Dataset and this class must have a dataloader()
  generator method
  """
  sys.path.append(str(module_dir))

  if not os.path.exists(os.path.join(module_dir, "__init__.py")):
    with open(os.path.join(module_dir, "__init__.py"), "w"):
      pass

  import dataset  # dataset module

  # get main module class
  main_module_cls = None
  for name, obj in dataset.__dict__.items():
    if inspect.isclass(obj) and "Dataset" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls(split).dataloader()


def load_zoo_dataset(name: str, split: str) -> Iterator:
  """
  Get dataset generator object from dataset zoo.
  Args:
    `name`: dataset module name in datasets/zoo/.
    `split`: "train" or "val"/"test" dataset split
  Returns:
    Data generator object
  """
  zoo_dataset = importlib.import_module(f"datasets.zoo.{name}")
  # get main module class
  main_module_cls = None
  for name, obj in zoo_dataset.__dict__.items():
    if inspect.isclass(obj) and "Dataset" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls(split).dataloader()


class UploadConfig:

  def __init__(
      self,
      user_id: str,
      app_id: str,
      pat: str,
      dataset_id: str,
      task: str,
      from_module: Optional[Union[str, os.PathLike]] = None,
      from_zoo: Optional[str] = None,  # load dataset from zoo
      split: str = "train",  # train or test/val
      chunk_size: int = 128,
      portal: str = "clarifai"):
    """
    Initialize upload configs.
    Args:
      `user_id`: Clarifai user id.
      `app_id`: Clarifai app id.
      `pat`: Clarifai PAT(Personal Access Token).
      `dataset_id`: Clarifai dataset id (where data is to be uploaded).
      `task`: either of `visual_clf`, `visual_detection`, `visual_segmentation` or `text_clf`.
      `from_module`: Path to dataset module directory.
        Should be left as None if `from_zoo` is to be used.
      `from_zoo`: Name of dataset to upload from the zoo.
        The name must match the dataset module name excluding the file extension.
        Should be left as None if `from_module` is to be used.
      `split`: Dataset split to upload. Either of train or test/val
      `chunk_size`: size of chunks for parallel data upload.
    """
    self.USER_ID = user_id
    self.APP_ID = app_id
    self.PAT = pat
    self.dataset_id = dataset_id
    self.task = task
    self.module_dir = from_module
    self.zoo_dataset = from_zoo
    self.split = split
    self.chunk_size = min(128, chunk_size)  # limit max protos in a req
    self.num_workers: int = min(10, cpu_count())  #15 req/sec rate limit
    self.annot_num_workers = 4
    self.max_retires = 10
    self.__base: str = ""
    if portal == "dev":
      self.__base = "https://api-dev.clarifai.com"
    elif portal == "staging":
      self.__base = "https://api-staging.clarifai.com"
    else:  #prod
      self.__base = "https://api.clarifai.com"

    # Set auth vars as env variables
    os.environ["CLARIFAI_USER_ID"] = self.USER_ID
    os.environ["CLARIFAI_APP_ID"] = self.APP_ID
    os.environ["CLARIFAI_API_BASE"] = self.__base
    os.environ["CLARIFAI_PAT"] = self.PAT

    self.STUB: service_pb2_grpc.V2Stub = create_stub()
    self.metadata: Tuple = (('authorization', 'Key ' + self.PAT),)
    self.user_app_id = resources_pb2.UserAppIDSet(user_id=self.USER_ID, app_id=self.APP_ID)

  def _upload_inputs(self, batch_input: List[resources_pb2.Input]) -> str:
    """
    Upload inputs to clarifai platform dataset.
    Args:
      batch_input: input batch protos
    Returns:
      input_job_id: Upload Input Job ID
    """
    input_job_id = uuid.uuid4().hex  # generate a unique id for this job
    response = self.STUB.PostInputs(
        service_pb2.PostInputsRequest(
            user_app_id=self.user_app_id, inputs=batch_input, inputs_add_job_id=input_job_id),)
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        print(f"Post inputs failed, status: {response.inputs[0].status.details}")
      except:
        print(f"Post inputs failed, status: {response.status.details}")

    return input_job_id

  def _upload_annotations(self, batch_annot: List[resources_pb2.Annotation]
                         ) -> Union[List[resources_pb2.Annotation], List[None]]:
    """
    Upload image annotations to clarifai detection dataset
    Args:
      batch_annot: annot batch protos
    Returns:
      retry_upload: failed annot upload
    """
    retry_upload = []  # those that fail to upload are stored for retries
    response = self.STUB.PostAnnotations(
        service_pb2.PostAnnotationsRequest(user_app_id=self.user_app_id, annotations=batch_annot),)

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
    """
    Uploads annotations concurrently.
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

  def _backoff_iterator(self) -> None:
    """
    Return iterator for exponential backoff intervals.
    """
    yield 0.1
    for i in range(5, 11):
      yield 0.01 * (2**i)
    while True:
      yield 0.01 * (2**10)  #10 sec

  def _wait_for_inputs(self, input_job_id: str) -> bool:
    """
    Wait for inputs to be processed. Cancel Job if timeout > 30 minutes.
    Args:
      input_job_id: Upload Input Job ID
    Returns:
      True if inputs are processed, False otherwise
    """
    backoff_iterator = self._backoff_iterator()
    max_retries = self.max_retires
    start_time = time.time()
    while True:
      response = self.STUB.GetInputsAddJob(
          service_pb2.GetInputsAddJobRequest(user_app_id=self.user_app_id, id=input_job_id),)

      if time.time() - start_time > 60 * 30 or max_retries == 0:  # 30 minutes timeout
        self.STUB.CancelInputsAddJob(
            service_pb2.CancelInputsAddJobRequest(user_app_id=self.user_app_id, id=input_job_id),
        )  #Cancel Job
        return False
      if response.status.code != status_code_pb2.SUCCESS:
        max_retries -= 1
        print(f"Get input job failed, status: {response.status.details}\n")
        continue
      if response.inputs_add_job.progress.in_progress_count == 0 and response.inputs_add_job.progress.pending_count == 0:
        return True
      else:
        time.sleep(next(backoff_iterator))

  def _delete_failed_inputs(self, input_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Delete failed input ids from clarifai platform dataset.
    Args:
      input_ids: batch input ids
    Returns:
      success_inputs: upload success input ids
      failed_inputs: upload failed input ids
    """
    success_status = status_pb2.Status(code=status_code_pb2.INPUT_DOWNLOAD_SUCCESS)
    response = self.STUB.ListInputs(
        service_pb2.ListInputsRequest(
            ids=input_ids,
            per_page=len(input_ids),
            user_app_id=self.user_app_id,
            status=success_status),)
    response_dict = MessageToDict(response)
    success_inputs = response_dict.get('inputs', [])

    success_input_ids = [input.get('id') for input in success_inputs]
    failed_input_ids = list(set(input_ids) - set(success_input_ids))
    #delete failed inputs
    self.STUB.DeleteInputs(
        service_pb2.DeleteInputsRequest(user_app_id=self.user_app_id, ids=failed_input_ids),)

    return success_input_ids, failed_input_ids

  def _upload_inputs_annotations(
      self, batch_input_ids: List[str]) -> Tuple[List[str], List[resources_pb2.Annotation]]:
    """
    Uploads batch of inputs and annotations concurrently to clarifai platform dataset.
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

  def _retry_uploads(self, failed_input_ids: List[str],
                     retry_annot_protos: List[resources_pb2.Annotation]) -> None:
    """
    Retry failed uploads.
    Args:
      failed_input_ids: failed input ids
      retry_annot_protos: failed annot protos
    """
    if failed_input_ids:
      self._upload_inputs_annotations(failed_input_ids)
    if retry_annot_protos:
      chunked_annotation_protos = Chunker(retry_annot_protos, self.chunk_size).chunk()
      _ = self._concurrent_annot_upload(chunked_annotation_protos)

  def _data_upload(self, input_ids: List[str]) -> None:
    """
    Uploads inputs and annotations to clarifai platform dataset.
    Args:
      input_ids: input ids
    """
    chunk_input_ids = Chunker(input_ids, self.chunk_size).chunk()
    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      with tqdm(total=len(chunk_input_ids), desc='Uploading Dataset') as progress:
        # Submit all jobs to the executor and store the returned futures
        futures = [
            executor.submit(self._upload_inputs_annotations, batch_input_ids)
            for batch_input_ids in chunk_input_ids
        ]

        for job in as_completed(futures):
          retry_input_proto, retry_annot_protos = job.result()
          self._retry_uploads(retry_input_proto, retry_annot_protos)
          progress.update()

  def upload_to_clarifai(self):
    """
    Execute data upload.
    """
    datagen_object = None
    if self.module_dir is None and self.zoo_dataset is None:
      raise Exception("One of `from_module` and `from_zoo` must be \
      specified. Both can't be None or defined at the same time.")
    elif self.module_dir is not None and self.zoo_dataset is not None:
      raise Exception("Use either of `from_module` or `from_zoo` \
      but NOT both.")
    elif self.module_dir is not None:
      datagen_object = load_dataset(self.module_dir, self.split)
    else:
      datagen_object = load_zoo_dataset(self.zoo_dataset, self.split)

    if self.task == "text_clf":
      self.dataset_obj = TextClassificationDataset(datagen_object, self.dataset_id, self.split)
      self._data_upload(self.dataset_obj.input_ids)

    elif self.task == "visual_detection":
      self.dataset_obj = VisualDetectionDataset(datagen_object, self.dataset_id, self.split)
      self._data_upload(self.dataset_obj.input_ids)  # TODO: get_img_ids or get_input_ids

    elif self.task == "visual_segmentation":
      self.dataset_obj = VisualSegmentationDataset(datagen_object, self.dataset_id, self.split)
      self._data_upload(self.dataset_obj.input_ids)

    else:  # visual-classification & visual-captioning
      self.dataset_obj = VisualClassificationDataset(datagen_object, self.dataset_id, self.split)
      self._data_upload(self.dataset_obj.input_ids)
