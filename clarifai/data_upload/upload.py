#! Clarifai data upload

import importlib
import inspect
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Iterator, Optional, Tuple, Union

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
      chunk_size: int = 16,
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
    self.chunk_size = chunk_size
    self.num_workers: int = cpu_count()
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

  def _upload_inputs(self, inputs):
    """
    Upload inputs to clarifai platform dataset.
    Args:
      inputs: input protos
    """
    upload_count = 0
    retry_upload = []  # those that fail to upload are stored for retries

    for inp_proto in inputs:
      response = self.STUB.PostInputs(
          service_pb2.PostInputsRequest(user_app_id=self.user_app_id, inputs=[inp_proto]),)

      MESSAGE_DUPLICATE_ID = "Input has a duplicate ID."
      if response.status.code != status_code_pb2.SUCCESS:
        try:
          if response.inputs[0].status.details != MESSAGE_DUPLICATE_ID:
            retry_upload.append(inp_proto)
          print(f"Post inputs failed, status: {response.inputs[0].status.details}\n")
          continue
        except:
          print(f"Post inputs failed, status: {response.status.details}\n")
      else:
        upload_count += 1

    return retry_upload

  def upload_annotations(self, inputs):
    """
    Upload image annotations to clarifai detection dataset
    """
    upload_count = 0
    retry_upload = []  # those that fail to upload are stored for retries

    for annot_proto in inputs:
      response = self.STUB.PostAnnotations(
          service_pb2.PostAnnotationsRequest(
              user_app_id=self.user_app_id, annotations=[annot_proto]),)

      if response.status.code != status_code_pb2.SUCCESS:
        try:
          print(f"Post annotations failed, status:\n{response.annotations[0].status.details}\n")
          continue
        except:
          print(f"Post annotations failed, status:\n{response.status.details}\n")
          retry_upload.append(annot_proto)
      else:
        upload_count += 1

    return retry_upload

  def concurrent_inp_upload(self, inputs, chunks):
    """
    Upload images concurrently.
    """
    inp_threads = []
    retry_upload = []

    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      for inp_batch in tqdm(inputs, total=chunks + 1, desc="uploading inputs..."):
        inp_threads.append(executor.submit(self._upload_inputs, inp_batch))
        time.sleep(0.1)

    for job in tqdm(
        as_completed(inp_threads), total=chunks + 1, desc="retry uploading failed protos..."):
      if job.result():
        retry_upload.extend(job.result())

    if len(
        list(retry_upload)) > 0:  ## TODO: use api_with_retries functionality via upload_inputs()
      _ = self._upload_inputs(retry_upload)

  def concurrent_annot_upload(self, inputs, chunks):
    """
    Uploads annotations concurrently.
    """
    annot_threads = []
    retry_annot_upload = []

    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
      for annot_batch in tqdm(inputs, total=chunks + 1, desc="uploading..."):
        annot_threads.append(executor.submit(self.upload_annotations, annot_batch))
        time.sleep(0.2)

    for job in tqdm(
        as_completed(annot_threads), total=chunks + 1, desc="retry uploading failed protos..."):
      if job.result():
        retry_annot_upload.extend(job.result())
    if len(retry_annot_upload) > 0:
      ## TODO: use api_with_retries functionality via upload_annotations()
      _ = self.upload_annotations(retry_annot_upload)

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
      chunks = len(text_protos) // self.num_workers
      chunked_text_protos = Chunker(text_protos, self.chunk_size).chunk()

      self.concurrent_inp_upload(chunked_text_protos, chunks)

    elif self.task == "visual_detection":
      dataset_obj = VisualDetectionDataset(datagen_object, self.dataset_id, self.split)
      img_protos, annotation_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)

      # Upload images
      chunks = len(img_protos) // self.num_workers
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()

      self.concurrent_inp_upload(chunked_img_protos, chunks)

      # Upload annotations:
      print("Uploading annotations.......")
      annotation_protos = dataset_obj._to_list(annotation_protos)
      chunks_ = len(annotation_protos) // self.num_workers
      chunked_annot_protos = Chunker(annotation_protos, self.chunk_size).chunk()

      self.concurrent_annot_upload(chunked_annot_protos, chunks_)

    elif self.task == "visual_segmentation":
      dataset_obj = VisualSegmentationDataset(datagen_object, self.dataset_id, self.split)
      img_protos, mask_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)
      mask_protos = dataset_obj._to_list(mask_protos)

      # Upload images
      chunks = len(img_protos) // self.num_workers
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()

      #self.concurrent_inp_upload(chunked_img_protos, chunks)
      # Upload masks:
      print("Uploading masks.......")
      chunks_ = len(mask_protos) // self.num_workers
      chunked_mask_protos = Chunker(mask_protos, self.chunk_size).chunk()

      self.concurrent_annot_upload(chunked_mask_protos, chunks_)
    else:  # visual-classification & visual-captioning
      dataset_obj = VisualClassificationDataset(datagen_object, self.dataset_id, self.split)
      img_protos = dataset_obj._get_input_protos()
      img_protos = dataset_obj._to_list(img_protos)

      # Upload images
      chunks = len(img_protos) // self.num_workers
      chunked_img_protos = Chunker(img_protos, self.chunk_size).chunk()

      self.concurrent_inp_upload(chunked_img_protos, chunks)
