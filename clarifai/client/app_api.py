from .api import ApiClient
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

class AppApi(ApiClient):
    def __init__(self, id, **kwargs):
        
        self.App = resources_pb2.App(id=id, **kwargs)
        super().__init__(user_id=self.user_id, app_id=id)

    def __getattr__(self, name):
        return getattr(self.App, name)

    def list_apps(self):
        return self.STUB.ListApps(
            service_pb2.ListAppsRequest(
                user_app_id=self.userDataObject
            )
        )