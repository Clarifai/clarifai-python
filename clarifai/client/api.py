from .base_auth import BaseAuth
from .user_api import UserApi
from clarifai_grpc.grpc.api import service_pb2

class ApiClient(BaseAuth):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def list_users(self):
        return self.STUB.ListUsers(
            service_pb2.ListUsersRequest(
                user_app_id=self.userDataObject
            )
        )

    @classmethod
    def user(cls, user_id, **kwargs):
        return UserApi(id=user_id, **kwargs)