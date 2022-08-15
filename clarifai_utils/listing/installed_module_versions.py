from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.service_pb2_grpc import V2Stub
from clarifai_grpc.grpc.api.status import status_code_pb2


def installed_module_versions_generator(stub: V2Stub,
                                        metadata: tuple,
                                        user_id: str,
                                        app_id: str,
                                        page_size: int = 64):
  """
  Lists all the installed module versions in the given userAppID user_id, app_id app.

  Args:
    stub: grpc client stub.
    user_id: the user to list from.
    app_id: the app in the user_id account to list from.
    page_size: the pagination size to use while iterating.

  Returns:
    imvs: a list of InstalledModuleVersion protos for all the installed modules versions in the app.
  """
  userDataObject = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

  imv_success_status = {status_code_pb2.SUCCESS}

  # HACK(zeiler): this is the number of default installed module versions every app has.
  # so with pagination
  seen = {
      "module_manager_install": False,
  }

  page = 1
  while True:
    response = stub.ListInstalledModuleVersions(
        service_pb2.ListInstalledModuleVersionsRequest(
            user_app_id=userDataObject, page=page, per_page=page_size),
        metadata=metadata)

    if response.status.code not in imv_success_status:
      raise Exception("ListInstalledModuleVersions failed with response %r" % response)
    for item in response.installed_module_versions:
      if item.id in seen:
        if not seen[item.id]:  # yield it once.
          seen[item.id] = True
          yield item
      else:
        yield item
    page += 1
    # if we don't get a full page back (plus the hard coded ones) we know we're done.
    if len(response.installed_module_versions) < page_size + len(seen):
      break
