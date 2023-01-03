from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import V2Stub


def datasets_generator(stub: V2Stub, user_id: str, app_id: str, page_size: int = 64):
  """
    Lists all the datasets in an application given a userAppID user_id, app_id app.

    Args:
      stub: client stub.
      user_id: the user to list from.
      app_id: the app in the user_id account to list from.
      page_size: the pagination size to use while iterating.

    Returns:
      datasets: a list of Datasets protos for all the Datasets in the app.
    """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  dataset_success_status = {status_code_pb2.SUCCESS}

  page = 1
  while True:
    response = stub.ListDatasets(
        service_pb2.ListDatasetsRequest(user_app_id=userDataObject, page=page, per_page=page_size),
    )

    if response.status.code not in dataset_success_status:
      raise Exception("ListDatasets failed with response %r" % response)
    if len(response.datasets) == 0:
      break
    else:
      for item in response.datasets:
        yield item
    page += 1
