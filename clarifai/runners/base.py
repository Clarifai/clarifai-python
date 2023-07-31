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

from typing import Type

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub


class BaseRunner:
  """
  Base class for remote inference runners. This should be subclassed with the run_input method
  implemented to process each input in the request.

  Then on the subclass call start() to start the run loop.
  """

  def __init__(self, auth: Type[ClarifaiAuthHelper], runner_id: str) -> None:
    """
    Args:
      auth: ClarifaiAuthHelper - the auth object to use
      runner_id: str - the id of the runner to use. Create the runner in the Clarifai API first

    """
    self.auth = auth
    self.stub = create_stub(self.auth)
    self.runner_id = runner_id

    # Check that the runner exists.
    response = self.stub.GetRunner(
        service_pb2.GetRunnerRequest(
            user_app_id=self.auth.get_user_app_id_proto(), runner_id=self.runner_id))
    if work_response.status.code != status_code_pb2.SUCCESS:
      raise Exception(
          f"Error getting runner, are you use this is a valid runner id {runner_id} at the user_id/app_id {self.auth.get_user_app_id_proto().user_id}/{self.auth.get_user_app_id_proto().app_id}. Error: {response.status.description}"
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
    # TODO: parallelize this
    for inp in request.inputs:
      # TODO: handle errors
      outputs.append(self.run_input(inp))

    return service_pb2.MultiOutputResponse(
        status=status_pb2.Status(
            code=status_code_pb2.SUCCESS,
            description="Success",
        ),
        outputs=outputs,
    )

  def run_input(self, input: resources_pb2.Input) -> resources_pb2.Output:
    """
    Run the model on the given input in the request. This is the method you should override to
    process each input in the request.

    Args:
      input: resources_pb2.Input - the input to run the model on

    Returns:
      resources_pb2.Output - the response from the model's run_input implementation.
    """
    raise NotImplementedError("run_input() not implemented")

  def _long_poll_loop(self):
    """
    This method will long poll for work, and when it gets work, it will run the model on the inputs
    and post the results back to the Clarifai API. It will then long poll again for more work.
    """
    c = 0
    # TODO: handle more errors within this loop so it never stops.
    # TODO: perhaps have multiple processes running this loop to handle more work.
    while True:
      # Long poll waiting for work.
      print("Loop iteration: {}".format(c))
      work_response = self.stub.ListRunnerItems(
          service_pb2.ListRunnerItemsRequest(
              user_app_id=self.auth.get_user_app_id_proto(), runner_id=self.runner_id))
      if work_response.status.code == status_code_pb2.RUNNER_NEEDS_RETRY:
        c += 1
        continue  # immediate restart the long poll
      if work_response.status.code != status_code_pb2.SUCCESS:
        raise Exception("Error getting work: {}".format(work_response.status.description))
      if len(work_response.items) == 0:
        print("No work to do. Waiting...")
        continue

      # We have work to do. Run the model on the inputs.
      for item in work_response.items:
        if not item.HasField('post_model_outputs_request'):
          raise Exception("Unexpected work item type: {}".format(item))
        print(
            f"Working on item: {item.id} with inputs {len(item.post_model_outputs_request.inputs)}"
        )
        result = self._run(item.post_model_outputs_request)

        result_response = self.stub.PostRunnerItemOutputs(
            service_pb2.PostRunnerItemOutputsRequest(
                user_app_id=self.auth.get_user_app_id_proto(),
                item_id=item.id,
                runner_id=self.runner_id,
                runner_item_outputs=[service_pb2.RunnerItemOutput(multi_output_response=result)]))
        if result_response.status.code != status_code_pb2.SUCCESS:
          raise Exception(
              json_format.MessageToJson(result_response, preserving_proto_field_name=True))
          # raise Exception("Error posting result: {}".format(result_response.status.description))
