from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401

from clarifai.client.app_api import AppApi
from clarifai.client.base_auth import BaseAuth


class UserApi(BaseAuth):
    """
    UserApi is a class that provides access to Clarifai API endpoints related to user information.
    Inherits from BaseAuth for authentication purposes.
    """
    def __init__(self, user_id: str, **kwargs):
        """
        Initializes a UserApi object.
        Args:
            user_id (str): The user ID for the user to interact with.
            **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
        """
        self.kwargs = {**kwargs, 'id': user_id}
        self.User = resources_pb2.User(**self.kwargs)
        super().__init__(user_id=self.id)

    def list_apps(self):
        """
        Lists all the apps for the user.
        """
        pass

    def app(self, app_id: str, **kwargs):
        """
        Returns an AppApi object for the specified app ID.
        Args:
            app_id (str): The app ID for the app to interact with.
            **kwargs: Additional keyword arguments to be passed to the AppApi.
        """
        kwargs['user_id'] = self.id
        return AppApi(app_id=app_id, **kwargs)

    def __getattr__(self, name):
        return getattr(self.User, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.User, param)}" for param in init_params if hasattr(self.User, param)
        ]
        return f"Clarifai User Details: \n{', '.join(attribute_strings)}\n"