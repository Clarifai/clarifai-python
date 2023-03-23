from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client import V2Stub


def module_versions_generator(stub: V2Stub,
                              user_id: str,
                              app_id: str,
                              module_id: str,
                              page_size: int = 64):
  """
  Lists all the module versions in the given userAppID user_id, app_id app, module_id module.

  Args:
    stub: client stub.
    user_id: the user to list from.
    app_id: the app in the user_id account to list from.
    module_id: the module in the app to list from.
    page_size: the pagination size to use while iterating.

  Returns:
    module_versions: a list of ModuleVersion protos for all the modules in the app.
  """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  success_status = {status_code_pb2.SUCCESS}

  page = 1
  while True:
    response = stub.ListModuleVersions(
        service_pb2.ListModuleVersionsRequest(
            user_app_id=userDataObject, module_id=module_id, page=page, per_page=page_size),)

    if response.status.code not in success_status:
      raise Exception("ListModuleVersions failed with response %r" % response)
    for item in response.module_versions:
      yield item
    page += 1
    # if we don't get a full page back we know we're done.
    if len(response.module_versions) < page_size:
      break
