from .base_auth import BaseAuth
from .app_api import AppApi
from clarifai_grpc.grpc.api import service_pb2, resources_pb2

class UserApi(BaseAuth):
    def __init__(self, id, **kwargs) -> None:
        self.kwargs = {**kwargs, 'id': id}
        self.User = resources_pb2.User(**self.kwargs)
        super().__init__(user_id=id)

    def list_apps(self):
        return self.STUB.ListApps(
            service_pb2.ListAppsRequest(
                user_app_id=self.userDataObject
            )
        )

    @classmethod
    def app(cls, app_id, **kwargs):
        return AppApi(id=app_id, **kwargs)

    def __getattr__(self, name):
        return getattr(self.User, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [f"{param}={getattr(self.User, param)}" for param in init_params if hasattr(self.User, param)]
        return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"