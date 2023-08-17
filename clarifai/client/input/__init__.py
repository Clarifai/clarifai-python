import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Input
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.utils.logging import get_logger
from clarifai.utils.misc import BackoffIterator, Chunker


class Inputs(Lister, BaseClient):
  """
  Inputs is a class that provides access to Clarifai API endpoints related to Input information.
  Inherits from BaseClient for authentication purposes.
  """

  def __init__(self, user_id: str, app_id: str, logger_level: str = "INFO", **kwargs):
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

  def get_input_from_url(self) -> None:
    """
    Create input protos for each data type from url.
    """
    raise NotImplementedError()

  def get_input_from_filename(self) -> None:
    """
    Create input protos for each data type from filename.
    """
    raise NotImplementedError()

  def get_input_from_bytes(self) -> None:
    """
    Create input protos for each data type from bytes.
    """
    raise NotImplementedError()

  def upload_inputs(self, inputs: List[Input], show_log: bool = True) -> str:
    """Upload list of input objects to the app.
    Args:
        inputs (list): List of input objects to upload.
        show_log (bool): Show upload status log.
    Returns:
        input_job_id: job id for the upload request.
    """
    if not isinstance(inputs, list):
      raise Exception("inputs must be a list of Input objects")
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

  def list_inputs(self) -> List[Input]:
    """Lists all the inputs for the app.
    Returns:
        list of Input: A list of Input objects for the app.
    Example:
        >>> from clarifai.client.user import User
        >>> text_obj = User(user_id="user_id").app(app_id="app_id").text()
        >>> text_obj.list_inputs()
    """
    request_data = dict(user_app_id=self.user_app_id, per_page=self.default_page_size)
    all_inputs_info = list(
        self.list_all_pages_generator(self.STUB.ListInputs, service_pb2.ListInputsRequest,
                                      request_data))
    for input_info in all_inputs_info:
      input_info['id'] = input_info.pop('input_id')
    return [resources_pb2.Input(**input_info) for input_info in all_inputs_info]

  def delete_inputs(self, inputs: List[Input]) -> None:
    """Delete list of input objects from the app.
    Args:
        input_ids (Input): List of input objects to delete.
    Example:
        >>> from clarifai.client.user import User
        >>> image_obj = User(user_id="user_id").app(app_id="app_id").image()
        >>> image_obj.delete_inputs(image_obj.list_inputs())
    """
    if not isinstance(inputs, list):
      raise Exception("input_ids must be a list of input ids")
    inputs_ids = [input.id for input in inputs]
    request = service_pb2.DeleteInputsRequest(user_app_id=self.user_app_id, ids=inputs_ids)
    response = self._grpc_request(self.STUB.DeleteInputs, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nInputs Deleted\n%s", response.status)

  def _bulk_upload(self, inputs: List[Input], chunk_size: int = 128) -> None:
    """Uploads process for large number of inputs .
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
