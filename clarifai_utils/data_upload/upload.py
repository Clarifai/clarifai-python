#! Clarifai data upload

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from base import Chunker
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from datasets import ImageClassificationDataset, TextClassificationDataset, VisualDetectionDataset
from omegaconf import OmegaConf
from tqdm import tqdm


def upload_data(config, inputs, inp_stub):
  """
  Upload inputs to clarifai platform dataset.
  Args:
  	config: auth and data path info.
  	inputs: input protos
  	inp_stub: grpc stub
  """
  STUB = inp_stub
  AUTHORIZATION = (("authorization", "Key " + config.auth["pat"]),)
  USER_APP_ID = resources_pb2.UserAppIDSet(
      user_id=config.auth["user_id"], app_id=config.auth["app_id"])

  upload_count = 0
  retry_upload = []  # those that fail to upload are stored for retries

  for inp_proto in inputs:
    response = STUB.PostInputs(
        service_pb2.PostInputsRequest(user_app_id=USER_APP_ID, inputs=[inp_proto]),
        metadata=AUTHORIZATION)

    if response.status.code != status_code_pb2.SUCCESS:
      try:
        print(f"Post inputs failed, status:\n{response.inputs[0].status.details}\n")
      except:
        print(f"Post inputs failed, status:\n{response.status.details}\n")
        retry_upload.append(inp_proto)
    else:
      upload_count += 1

  return retry_upload


def upload_annotations(config, inputs, inp_stub):
  """
  Upload image annotations to clarifai detection dataset
  """
  STUB = inp_stub
  AUTHORIZATION = (("authorization", "Key " + config.auth["pat"]),)
  USER_APP_ID = resources_pb2.UserAppIDSet(
      user_id=config.auth["user_id"], app_id=config.auth["app_id"])

  upload_count = 0
  retry_upload = []  # those that fail to upload are stored for retries

  for annot_proto in inputs:
    response = STUB.PostAnnotations(
        service_pb2.PostAnnotationsRequest(user_app_id=USER_APP_ID, annotations=[annot_proto]),
        metadata=AUTHORIZATION)

    if response.status.code != status_code_pb2.SUCCESS:
      try:
        print(f"Post annotations failed, status:\n{response.inputs[0].status.details}\n")
      except:
        print(f"Post annotations failed, status:\n{response.status.details}\n")
        retry_upload.append(annot_proto)
    else:
      upload_count += 1

  return retry_upload


def concurrent_inp_upload(config, inputs, workers, chunks, stub):
  """
  Upload images concurrently for efficiency.
  """
  inp_threads = []
  retry_upload = []

  with ThreadPoolExecutor(max_workers=workers) as executor:
    for inp_batch in tqdm(inputs, total=chunks + 1, desc="uploading.."):
      inp_threads.append(executor.submit(upload_data, config, inp_batch, stub))
      time.sleep(0.1)

  for job in tqdm(
      as_completed(inp_threads), total=chunks + 1, desc="retry uploading failed protos..."):
    if job.result():
      retry_upload.extend(job.result())
  if len(list(retry_upload)) > 0:  ## TODO: use api_with_retries functionality via upload_data()
    _ = upload_data(config, retry_upload, stub)


def concurrent_annot_upload(config, inputs, workers, chunks, stub):
  """
  Upload annotations concurrently for efficiency.
  """
  annot_threads = []
  retry_annot_upload = []

  with ThreadPoolExecutor(max_workers=workers) as executor:
    for annot_batch in tqdm(inputs, total=chunks + 1, desc="uploading..."):
      annot_threads.append(executor.submit(upload_annotations, config, annot_batch, stub))
      time.sleep(0.1)

  for job in tqdm(
      as_completed(annot_threads), total=chunks + 1, desc="retry uploading failed protos..."):
    if job.result():
      retry_annot_upload.extend(job.result())
  if len(retry_annot_upload) > 0:
    ## TODO: use api_with_retries functionality via upload_annotations()
    _ = upload_annotations(config, retry_annot_upload, stub)


def upload_to_clarifai(config, task: str = "visual_clf"):
  """
  Execute data upload.
  Args:
    `config`: auth and data path info.
    `task`: Machine Learning domain task data type.
       Can be either of `visual_clf`, `visual_det` or `text_clf`.
  """
  CHANNEL = ClarifaiChannel.get_json_channel()
  STUB = service_pb2_grpc.V2Stub(CHANNEL)
  workers = cpu_count()

  if task == "text_clf":
    dataset_obj = TextClassificationDataset(config.data["clf_text_dir"], config.data["dataset_id"],
                                            config["split"])
    text_protos = dataset_obj._get_input_protos()
    text_protos = dataset_obj.to_list(text_protos)

    # Upload text
    chunks = len(text_protos) // workers
    chunked_text_protos = Chunker(text_protos, config["chunk_size"]).chunk()

    concurrent_inp_upload(config, chunked_text_protos, workers, chunks, STUB)

  elif task == "visual_det":
    dataset_obj = VisualDetectionDataset(
        config.data["visual_det_image_dir"],
        config.data["visual_det_labels_dir"],
        config.data["dataset_id"],
        config["split"],
        labels_from_text_file=False)
    img_protos, annotation_protos = dataset_obj._get_input_protos()
    img_protos = dataset_obj.to_list(img_protos)

    # Upload images
    chunks = len(img_protos) // workers
    chunked_img_protos = Chunker(img_protos, config["chunk_size"]).chunk()

    concurrent_inp_upload(config, chunked_img_protos, workers, chunks, STUB)

    # Upload annotations:
    print("Uploading annotations.......")
    annotation_protos = dataset_obj.to_list(annotation_protos)
    chunks_ = len(annotation_protos) // workers
    chunked_annot_protos = Chunker(annotation_protos, config["chunk_size"]).chunk()

    concurrent_annot_upload(config, chunked_annot_protos, workers, chunks_, STUB)

  else:  # default to visual_clf
    dataset_obj = ImageClassificationDataset(config.data["clf_image_dir"],
                                             config.data["dataset_id"], config["split"])
    img_protos = dataset_obj._get_input_protos()
    img_protos = dataset_obj.to_list(img_protos)

    # Upload images
    chunks = len(img_protos) // workers
    chunked_img_protos = Chunker(img_protos, config["chunk_size"]).chunk()

    concurrent_inp_upload(config, chunked_img_protos, workers, chunks, STUB)


if __name__ == "__main__":
  yaml_path = "./config.yaml"
  config = OmegaConf.load(yaml_path)
  upload_to_clarifai(config, task=config["task"])
