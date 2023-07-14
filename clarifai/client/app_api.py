from .base_auth import BaseAuth
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

class AppApi(BaseAuth):
    def __init__(self, id, **kwargs):
        self.kwargs = {**kwargs, 'id': id}
        self.App = resources_pb2.App(id=id, **kwargs)
        super().__init__(user_id=self.user_id, app_id=id)

    def __getattr__(self, name):
        return getattr(self.App, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [f"{param}={getattr(self.App, param)}" for param in init_params if hasattr(self.App, param)]
        return f"Clarifai App Details: \n{', '.join(attribute_strings)}\n"