from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.service_pb2_grpc import V2Stub
from clarifai_grpc.grpc.api.status import status_code_pb2


def inputs_generator(stub: V2Stub, metadata: tuple, user_id: str, app_id: str,
                     page_size: int = 16, only_successful_inputs: bool = True):
  """
  Lists all the inputs in the given userAppID user_id, app_id app. This uses the more efficient
  streaming input listing technique.

  Args:
    stub: grpc client stub.
    metadata: the auth metadata for the grpc stub.
    user_id: the user to list from.
    app_id: the app in the user_id account to list from.
    page_size: the pagination size to use while iterating.
    only_successful_inputs: only accept inputs with non-failure statuses.
  """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  input_success_status = {
      status_code_pb2.INPUT_DOWNLOAD_SUCCESS, status_code_pb2.INPUT_DOWNLOAD_PENDING,
      status_code_pb2.INPUT_DOWNLOAD_IN_PROGRESS
  }

  response_success_status = {status_code_pb2.SUCCESS, status_code_pb2.MIXED_STATUS}

  last_id = ''

  while True:
    response = stub.StreamInputs(
        service_pb2.StreamInputsRequest(
            user_app_id=userDataObject, per_page=page_size, last_id=last_id),
        metadata=metadata)
    if response.status.code not in response_success_status:
      raise Exception("Stream inputs failed with response %r" % response)
    if len(response.inputs) == 0:
      break
    else:
      for inp in response.inputs:
        last_id = inp.id
        if only_successful_inputs:
          if inp.status.code in input_success_status:
            yield inp
        else:
          yield inp
