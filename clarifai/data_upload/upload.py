#! Clarifai data upload

import importlib
import inspect
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Iterator, List, Optional, Tuple, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
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

  def _upload_inputs(self, batch_input: List[resources_pb2.Input]
                    ) -> Union[List[resources_pb2.Input], List[None]]:
    """
    Upload inputs to clarifai platform dataset.
    Args:
      batch_input: input batch protos
    Returns:
      retry_upload: failed input upload
    """
    retry_upload = []  # those that fail to upload are stored for retries
    response = self.STUB.PostInputs(
        service_pb2.PostInputsRequest(user_app_id=self.user_app_id, inputs=batch_input),)

    MESSAGE_DUPLICATE_ID = "Input has a duplicate ID."
    if response.status.code != status_code_pb2.SUCCESS:
      try:
        if response.inputs[0].status.details != MESSAGE_DUPLICATE_ID:
          retry_upload.extend(batch_input)
        print(f"Post inputs failed, status: {response.inputs[0].status.details}\n")
      except:
        if "Duplicated inputs ID" not in response.status.details:
          retry_upload.extend(batch_input)
        print(f"Post inputs failed, status: {response.status.details}\n")

    return retry_upload

  def upload_annotations(self, batch_annot: List[resources_pb2.Annotation]
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
        print(f"Post annotations failed, status:\n{response.annotations[0].status.details}\n")
      except:
        print(f"Post annotations failed, status:\n{response.status.details}\n")
      finally:
        retry_upload.extend(batch_annot)

    return retry_upload

  def concurrent_inp_upload(
      self,
      inputs: List[List[resources_pb2.Input]],
      chunks: int,
      desc: str = "uploading inputs...") -> Union[List[resources_pb2.Input], List[None]]:
    """
    Upload images concurrently.
    Args:
      inputs: input protos
      chunks: number of inputs chunks
    Returns:
      retry_upload: All failed input protos during upload
    """
    inp_threads = []
    retry_upload = []

    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      with tqdm(total=chunks, desc=desc) as progress:
        # Submit all jobs to the executor and store the returned futures
        inp_threads = [executor.submit(self._upload_inputs, inp_batch) for inp_batch in inputs]

        for job in as_completed(inp_threads):
          result = job.result()
          if result:
            retry_upload.extend(result)
          progress.update()

    return retry_upload

  def concurrent_annot_upload(
      self,
      annots: List[List[resources_pb2.Annotation]],
      chunks: int,
      desc: str = "uploading annotations...") -> Union[List[resources_pb2.Annotation], List[None]]:
    """
    Uploads annotations concurrently.
    Args:
      annots: annot protos
      chunks: number of annots chunks
    Returns:
      retry_annot_upload: All failed annot protos during upload
    """
    annot_threads = []
    retry_annot_upload = []

    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      with tqdm(total=chunks, desc=desc) as progress:
        # Submit all jobs to the executor and store the returned futures
        annot_threads = [
            executor.submit(self.upload_annotations, inp_batch) for inp_batch in annots
        ]

        for job in as_completed(annot_threads):
          result = job.result()
          if result:
            retry_annot_upload.extend(result)
          progress.update()

    return retry_annot_upload

  def retry_concurrent_uploads(
      self,
      retry_upload_protos: Union[List[resources_pb2.Input], List[resources_pb2.Annotation]],
      upload_type: str = "input") -> None:
    """
    Retry Uploads of inputs/annotations.
    Args:
      retry_upload_protos: upload protos for retry
      upload_type: input/annot protos type
    """
    retry_chunked_upload_protos = Chunker(retry_upload_protos, self.chunk_size).chunk()
    if len(retry_upload_protos) > 0 and upload_type == "input":
      _ = self.concurrent_inp_upload(
          retry_chunked_upload_protos,
          len(retry_chunked_upload_protos),
          desc="retry uploading failed input protos...")
    elif len(retry_upload_protos) > 0 and upload_type == "annot":
      _ = self.concurrent_annot_upload(
          retry_chunked_upload_protos,
          len(retry_chunked_upload_protos),
          desc="retry uploading failed annotation protos...")

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
      dataset_obj = TextClassificationDataset(datagen_object, self.dataset_id, self.split)
      text_protos = dataset_obj._get_input_protos()
      text_protos = dataset_obj._to_list(text_protos)

      # Upload text
      chunked_text_protos = Chunker(text_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_inp_upload(chunked_text_protos,
                                                       len(chunked_text_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "input")

    elif self.task == "visual_detection":
      dataset_obj = VisualDetectionDataset(datagen_object, self.dataset_id, self.split)
      img_protos, annotation_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)

      # Upload images
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_inp_upload(chunked_img_protos, len(chunked_img_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "input")

      # Upload annotations:
      print("Uploading annotations.......")
      annotation_protos = dataset_obj._to_list(annotation_protos)
      chunked_annot_protos = Chunker(annotation_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_annot_upload(chunked_annot_protos,
                                                         len(chunked_annot_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "annot")

    elif self.task == "visual_segmentation":
      dataset_obj = VisualSegmentationDataset(datagen_object, self.dataset_id, self.split)
      img_protos, mask_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)
      mask_protos = dataset_obj._to_list(mask_protos)

      # Upload images
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_inp_upload(chunked_img_protos, len(chunked_img_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "input")

      # Upload masks:
      print("Uploading masks.......")
      chunked_mask_protos = Chunker(mask_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_annot_upload(chunked_mask_protos,
                                                         len(chunked_mask_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "annot")

    else:  # visual-classification & visual-captioning
      dataset_obj = VisualClassificationDataset(datagen_object, self.dataset_id, self.split)
      img_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)

      # Upload images
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()
      retry_upload_protos = self.concurrent_inp_upload(chunked_img_protos, len(chunked_img_protos))
      self.retry_concurrent_uploads(retry_upload_protos, "input")
