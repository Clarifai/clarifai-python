from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import V2Stub


def models_generator(stub: V2Stub,
                     user_id: str,
                     app_id: str,
                     page_size: int = 64,
                     only_in_app: bool = False):
  """
    Lists all the models accessible in an application given a userAppID user_id, app_id app.

    Args:
      stub: client stub.
      user_id: the user to list from.
      app_id: the app in the user_id account to list from.
      page_size: the pagination size to use while iterating.
      only_in_app: if the models returned should only be ones that have been created in the app

    Returns:
      models: a list of Model protos for all the models in the app.
    """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  model_success_status = {status_code_pb2.SUCCESS}

  page = 1
  while True:
    response = stub.ListModels(
        service_pb2.ListModelsRequest(user_app_id=userDataObject, page=page, per_page=page_size),)

    if response.status.code not in model_success_status:
      raise Exception("ListModels failed with response %r" % response)
    if len(response.models) == 0:
      break
    else:
      if only_in_app:
        for item in response.models:
          if item.app_id == app_id:
            yield item
      else:
        for item in response.models:
          yield item
    page += 1
