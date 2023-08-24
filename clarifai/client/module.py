from typing import Dict, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import get_logger


class Module(Lister, BaseClient):
  """Module is a class that provides access to Clarifai API endpoints related to Module information."""

  def __init__(self,
               url_init: str = "",
               module_id: str = "",
               module_version: Dict = {'id': ""},
               **kwargs):
    """Initializes a Module object.

        Args:
            url_init (str): The URL to initialize the module object.
            module_id (str): The Module ID to interact with.
            module_version (dict): The Module Version to interact with.
            **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
        """
    if url_init != "" and module_id != "":
      raise UserError("You can only specify one of url_init or module_id.")
    if url_init == "" and module_id == "":
      raise UserError("You must specify one of url_init or module_id.")
    if url_init != "":
      user_id, app_id, module_id, module_version_id = ClarifaiUrlHelper.split_module_ui_url(
          url_init)
      module_version = {'id': module_version_id}
      kwargs = {'user_id': user_id, 'app_id': app_id}

    self.kwargs = {**kwargs, 'id': module_id, 'module_version': module_version}
    self.module_info = resources_pb2.Module(**self.kwargs)
    self.logger = get_logger(logger_level="INFO")
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self)

  def list_versions(self) -> List['Module']:
    """Lists all the module versions for the module.

        Returns:
            List[Moudle]: A list of Module objects for versions of the module.

        Example:
            >>> from clarifai.client.module import Module
            >>> module = Module(module_id='module_id', user_id='user_id', app_id='app_id')
            >>> all_Module_versions = module.list_versions()
        """
    request_data = dict(
        user_app_id=self.user_app_id,
        module_id=self.id,
        per_page=self.default_page_size,
    )
    all_module_versions_info = list(
        self.list_all_pages_generator(self.STUB.ListModuleVersions,
                                      service_pb2.ListModuleVersionsRequest, request_data))

    for module_version_info in all_module_versions_info:
      module_version_info['id'] = module_version_info['module_version_id']
      del module_version_info['module_version_id']

    return [
        Module(module_id=self.id, **dict(self.kwargs, module_version=module_version_info))
        for module_version_info in all_module_versions_info
    ]

  def __getattr__(self, name):
    return getattr(self.module_info, name)

  def __str__(self):
    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.module_info, param)}" for param in init_params
        if hasattr(self.module_info, param)
    ]
    return f"Module Details: \n{', '.join(attribute_strings)}\n"
