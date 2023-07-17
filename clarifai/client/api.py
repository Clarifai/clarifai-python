from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from ..errors import ApiError
from ..utils.logging import get_logger
from .base_auth import BaseAuth
from .user_api import UserApi

logger = get_logger(__name__)


class ApiClient(BaseAuth):

  def __init__(self, pat="", **kwargs) -> None:
    super().__init__(pat=pat, **kwargs)

  def list_users(self):
    try:
      list_users_response = self.STUB.ListUsers(
          service_pb2.ListUsersRequest(user_app_id=self.userDataObject))
      if list_users_response.status.code != status_code_pb2.SUCCESS:
        raise ApiError(
            resource=self.base,
            params=self.kwargs,
            method='ListUsers',
            response=list_users_response)

    except ApiError:
      logger.exception("ApiError")

  @classmethod
  def user(cls, user_id, **kwargs):
    return UserApi(id=user_id, **kwargs)
