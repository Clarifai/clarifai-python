from typing import Dict, Generator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import logger


class Module(Lister, BaseClient):
    """Module is a class that provides access to Clarifai API endpoints related to Module information."""

    def __init__(
        self,
        url: str = None,
        module_id: str = None,
        module_version: Dict = {'id': ""},
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes a Module object.

        Args:
            url (str): The URL to initialize the module object.
            module_id (str): The Module ID to interact with.
            module_version (dict): The Module Version to interact with.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT.
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN.
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the Module.
        """
        if url and module_id:
            raise UserError("You can only specify one of url or module_id.")
        if not url and not module_id:
            raise UserError("You must specify one of url or module_id.")
        if url:
            user_id, app_id, module_id, module_version_id = ClarifaiUrlHelper.split_module_ui_url(
                url
            )
            module_version = {'id': module_version_id}
            kwargs = {'user_id': user_id, 'app_id': app_id}

        self.kwargs = {**kwargs, 'id': module_id, 'module_version': module_version}
        self.module_info = resources_pb2.Module(**self.kwargs)
        self.logger = logger
        BaseClient.__init__(
            self,
            user_id=self.user_id,
            app_id=self.app_id,
            base=base_url,
            pat=pat,
            token=token,
            root_certificates_path=root_certificates_path,
        )
        Lister.__init__(self)

    def list_versions(
        self, page_no: int = None, per_page: int = None
    ) -> Generator['Module', None, None]:
        """Lists all the module versions for the module.

        Args:
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Moudle: Module objects for versions of the module.

        Example:
            >>> from clarifai.client.module import Module
            >>> module = Module(module_id='module_id', user_id='user_id', app_id='app_id')
            >>> all_Module_versions = list(module.list_versions())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(
            user_app_id=self.user_app_id,
            module_id=self.id,
        )
        all_module_versions_info = self.list_pages_generator(
            self.STUB.ListModuleVersions,
            service_pb2.ListModuleVersionsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for module_version_info in all_module_versions_info:
            module_version_info['id'] = module_version_info['module_version_id']
            del module_version_info['module_version_id']
            yield Module.from_auth_helper(
                self.auth_helper,
                module_id=self.id,
                **dict(self.kwargs, module_version=module_version_info),
            )

    def __getattr__(self, name):
        return getattr(self.module_info, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.module_info, param)}"
            for param in init_params
            if hasattr(self.module_info, param)
        ]
        return f"Module Details: \n{', '.join(attribute_strings)}\n"
