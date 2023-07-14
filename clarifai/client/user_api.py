from .api import ApiClient
from .app_api import AppApi
from clarifai_grpc.grpc.api import service_pb2, resources_pb2

class UserApi(ApiClient):
    def __init__(self, id, **kwargs) -> None:
        self.kwargs = kwargs
        self.User = resources_pb2.User(id=id, **self.kwargs)
        super().__init__(user_id=id)

    def list_users(self):
        return self.STUB.ListUsers(
            service_pb2.ListUsersRequest(
                user_app_id=self.userDataObject
            )
        )

    def get_app(self, app_id):
        self.kwargs['user_id'] = self.id
        return AppApi(app_id, **self.kwargs)

    def __getattr__(self, name):
        return getattr(self.User, name)

    def __str__(self):
        attribute_strings = []
        for attribute in self.Model.__slots__:
            value = getattr(self.Model, attribute)
            attribute_strings.append(f"{attribute}={value}")
        return ", ".join(attribute_strings)