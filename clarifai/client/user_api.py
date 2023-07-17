from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from rich.console import Console

from ..errors import ApiError
from ..utils.logging import get_logger, table_from_dict
from .app_api import AppApi
from .base_auth import BaseAuth

logger = get_logger(__name__)


class UserApi(BaseAuth):

  def __init__(self, id, **kwargs) -> None:
    self.kwargs = {**kwargs, 'id': id}
    self.User = resources_pb2.User(**self.kwargs)
    super().__init__(user_id=id)

  def list_apps(self):
    try:
      list_apps_response = self.STUB.ListApps(
          service_pb2.ListAppsRequest(user_app_id=self.userDataObject))

      if list_apps_response.status.code != status_code_pb2.SUCCESS:
        raise ApiError(
            resource=self.base, params=self.kwargs, method='ListApps', response=list_apps_response)

      table = table_from_dict(
          MessageToDict(list_apps_response)['apps'],
          ['userId', 'id', 'name', 'defaultWorkflowId', 'createdAt', 'description'],
          title="Apps")
      Console().print(table)

    except ApiError:
      logger.exception("ApiError")

  @classmethod
  def app(cls, app_id, **kwargs):
    return AppApi(id=app_id, **kwargs)

  def __getattr__(self, name):
    return getattr(self.User, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.User, param)}" for param in init_params
        if hasattr(self.User, param)
    ]
    return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"
