# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interface to Clarifai Runners API."""

import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from clarifai.client.base import BaseClient
from clarifai.errors import UserError
from clarifai.utils.logging import get_logger


class Runner(BaseClient):
  """Base class for remote inference runners. This should be subclassed with the run_input method
  implemented to process each input in the request.

  Then on the subclass call start() to start the run loop.
  """

  def __init__(self,
               runner_id: str,
               user_id: str = None,
               check_runner_exists: bool = True,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               num_parallel_polls: int = 4,
               **kwargs) -> None:
    """
    Args:
      runner_id (str): the id of the runner to use. Create the runner in the Clarifai API first
      user_id (str): Clarifai User ID
      base_url (str): Base API url. Default "https://api.clarifai.com"
      pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
      num_parallel_polls (int): the max number of threads for parallel run loops to be fetching work from
    """
    user_id = user_id or os.environ.get("CLARIFAI_USER_ID", "")

    if not user_id:
      raise UserError(
          "Set CLARIFAI_USER_ID as environment variables or pass user_id as input arguments")

    self.runner_id = runner_id
    self.logger = get_logger("INFO", __name__)
    self.kwargs = {**kwargs, 'id': runner_id, 'user_id': user_id}
    self.runner_info = resources_pb2.Runner(**self.kwargs)
    self.num_parallel_polls = min(10, num_parallel_polls)
    BaseClient.__init__(self, user_id=self.user_id, app_id="", base=base_url, pat=pat)

    # Check that the runner exists.
    if check_runner_exists:
      request = service_pb2.GetRunnerRequest(user_app_id=self.user_app_id, runner_id=runner_id)
      response = self._grpc_request(self.STUB.GetRunner, request)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(
            f"Error getting runner, are you use this is a valid runner id {runner_id} for user_id {user_id}. Error: {response.status.description}, Details: {response.status.details}"
        )

  def start(self):
    """
    Start the run loop. This will ask the Clarifai API for work, and when it gets work, it will run
    the model on the inputs and post the results back to the Clarifai API. It will then ask for more
    work again.
    """
    self._long_poll_loop()

  def _run(self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """
    Run the model on the given request. You shouldn't need to override this method, see run_input
    for the implementation to process each input in the request.

    Args:
      request: service_pb2.PostModelOutputsRequest - the request to run the model on

    Returns:
      service_pb2.MultiOutputResponse - the response from the model's run_input implementation.
    """
    outputs = []

    # TODO: Could cache the model and this conversion if the hash is the same.
    model = request.model
    output_info = None
    if request.model.model_version.id != "":
      output_info = json_format.MessageToDict(
          model.model_version.output_info, preserving_proto_field_name=True)

    # TODO: parallelize this over inputs in a single request.
    for inp in request.inputs:
      try:
        output = self.run_input(inp, output_info)
      except Exception as e:
        self.logger.exception("run_input failed:")
        traceback.print_exc()
        output = resources_pb2.Output(status=status_pb2.Status(
            code=status_code_pb2.MODEL_PREDICTION_FAILED,
            description="Failed",
            details="",
            stack_trace=traceback.format_exc().split('\n'),
            internal_details=str(e),
        ))
        self.logger.exception(output)
      outputs.append(output)

    successes = [o.status.code == status_code_pb2.SUCCESS for o in outputs]
    if all(successes):
      status = status_pb2.Status(
          code=status_code_pb2.SUCCESS,
          description="Success",
      )
    elif any(successes):
      status = status_pb2.Status(
          code=status_code_pb2.MIXED_STATUS,
          description="Mixed Status",
      )
    else:
      status = status_pb2.Status(
          code=status_code_pb2.FAILURE,
          description="Failed",
      )

    return service_pb2.MultiOutputResponse(
        status=status,
        outputs=outputs,
    )

  def run_input(self, input: resources_pb2.Input,
                output_info: resources_pb2.OutputInfo) -> resources_pb2.Output:
    """
    Run the model on the given input in the request. This is the method you should override to
    process each input in the request.

    Args:
      input: resources_pb2.Input - the input to run the model on
      output_info: resources_pb2.OutputInfo - the output info for the model which includes
    output_info.params that the model can pass in on very prediction request. These can be provided
    during PostModelVersions as default for every request or can be overridden on a per request by
    passing in output_info in the PostModelOutputs request as the model.model_version.output_info.params
    field.

    Returns:
      resources_pb2.Output - the response from the model's run_input implementation.
    """
    raise NotImplementedError("run_input() not implemented")

  def _long_poll_loop(self):
    """
    This method will long poll for work, and when it gets work, it will run the model on the inputs
    and post the results back to the Clarifai API. It will then long poll again for more work.
    """

    def _single_thread_loop():
      c = 0
      while True:
        # Long poll waiting for work.
        try:
          self.logger.info(f"Loop iteration: {c} for thread {threading.get_ident()}")
          work_response = self._grpc_request(self.STUB.ListRunnerItems,
                                             service_pb2.ListRunnerItemsRequest(
                                                 user_app_id=self.user_app_id,
                                                 runner_id=self.runner_id))
          if work_response.status.code == status_code_pb2.RUNNER_NEEDS_RETRY:
            c += 1
            continue  # immediate restart the long poll
          if work_response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Error getting work: {}".format(work_response.status.description))
          if len(work_response.items) == 0:
            self.logger.info("No work to do. Waiting...")
            continue

          # We have work to do. Run the model on each item fetched (typically 1 item at a time right
          # now is returned by ListRunnerItems).
          for item in work_response.items:
            if not item.HasField('post_model_outputs_request'):
              raise Exception("Unexpected work item type: {}".format(item))
            self.logger.info(
                f"Working on item: {item.id} with inputs {len(item.post_model_outputs_request.inputs)}"
            )
            # run this batch of data through the model.
            result = self._run(item.post_model_outputs_request)

            result_response = self._grpc_request(
                self.STUB.PostRunnerItemOutputs,
                service_pb2.PostRunnerItemOutputsRequest(
                    user_app_id=self.user_app_id,
                    item_id=item.id,
                    runner_id=self.runner_id,
                    runner_item_outputs=[
                        service_pb2.RunnerItemOutput(multi_output_response=result)
                    ]))
            if result_response.status.code != status_code_pb2.SUCCESS:
              raise Exception(
                  json_format.MessageToJson(result_response, preserving_proto_field_name=True))
              # raise Exception("Error posting result: {}".format(result_response.status.description))
        except Exception as e:  # keep the loop running
          self.logger.exception(e)
          continue

    self.logger.info(f"Starting run loop for {self.num_parallel_polls} threads...")
    if self.num_parallel_polls == 1:
      _single_thread_loop()
    else:
      pool = ThreadPoolExecutor(max_workers=self.num_parallel_polls)
      futures = [pool.submit(_single_thread_loop) for _ in range(self.num_parallel_polls)]
      for f in futures:
        self.logger.info(f.result())
      pool.shutdown()

  def __getattr__(self, name):
    return getattr(self.runner_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.runner_info, param)}" for param in init_params
        if hasattr(self.runner_info, param)
    ]
    return f"Runner Details: \n{', '.join(attribute_strings)}\n"
