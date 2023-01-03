from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import V2Stub


def concepts_generator(stub: V2Stub, user_id: str, app_id: str, page_size: int = 64):
  """
    Lists all the concept in the given userAppID user_id, app_id app.

    Args:
      stub: client stub.
      user_id: the user to list from.
      app_id: the app in the user_id account to list from.
      page_size: the pagination size to use while iterating.

    Returns:
      concepts: a list of Concept protos for all the concepts in the app.
    """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  concept_success_status = {status_code_pb2.SUCCESS}

  page = 1
  while True:
    response = stub.ListConcepts(
        service_pb2.ListConceptsRequest(user_app_id=userDataObject, page=page, per_page=page_size),
    )

    if response.status.code not in concept_success_status:
      raise Exception("ListConcepts failed with response %r" % response)
    if len(response.concepts) == 0:
      break
    else:
      for item in response.concepts:
        yield item
    page += 1
