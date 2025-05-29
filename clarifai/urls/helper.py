from collections import namedtuple
from urllib.parse import urlparse

# To help with using ClarifaiUrlHelper with defaults as ClarifaiUrlHelper()
auth_obj = namedtuple("auth", ["ui", "base"])


class ClarifaiUrlHelper(object):
    """Lots of helper functionality for dealing with urls around modules."""

    def __init__(
        self,
        auth=None,
        module_manager_imv_id: str = "module_manager_install",
    ):
        """
        Args:
          auth: a ClarifaiAuthHelper object. Pass None to use the values from the current context.
          module_manager_imv_id: the ID of the module manager installed module version.
        """
        self._auth = auth
        self._module_manager_imv_id = module_manager_imv_id
        self._current_context = None
        if self._auth is None:
            self._auth = auth_obj(self.current_ctx.ui, self.current_ctx.api_base)

    @property
    def ui(self):
        return self._auth.ui

    @property
    def base(self):
        return self._auth.base

    @property
    def current_ctx(self):
        if self._current_context is None:
            from clarifai.utils.config import Config

            self._current_context = Config.from_yaml().current
        return self._current_context

    def module_ui_url(
        self,
        user_id: str = None,
        app_id: str = None,
        module_id: str = None,
        module_version_id: str = None,
    ):
        """This is the path to the module in community."""
        if user_id is None:
            user_id = self.current_ctx.user_id
        if app_id is None:
            app_id = self.current_ctx.app_id
        if module_id is None:
            module_id = self.current_ctx.module_id
        if module_version_id is None:
            module_version_id = self.current_ctx.module_version_id

        return "%s/%s/%s/modules/%s/versions/%s" % (
            self.ui,
            user_id,
            app_id,
            module_id,
            module_version_id,
        )

    def module_install_ui_url(
        self, dest_user_id: str = None, dest_app_id: str = None, module_url: str = None
    ):
        """This is a url that allows for installation of the module from the community at 'module_url'
        into the destination app_id of the destination user_id."""
        if dest_user_id is None:
            dest_user_id = self.current_ctx.user_id
        if dest_app_id is None:
            dest_app_id = self.current_ctx.app_id
        if module_url is None:
            raise ValueError("module_url must be provided to install a module.")
        return "%s/%s/%s/installed_module_versions/%s/install?install=%s" % (
            self.ui,
            dest_user_id,
            dest_app_id,
            self._module_manager_imv_id,
            module_url,
        )

    def imv_ui_url(self, dest_user_id: str = None, dest_app_id: str = None, imv_id: str = None):
        """This is the path to the resource in the UI."""
        if dest_user_id is None:
            dest_user_id = self.current_ctx.user_id
        if dest_app_id is None:
            dest_app_id = self.current_ctx.app_id
        if imv_id is None:
            raise ValueError("imv_id must be provided to get the IMV API URL.")
        return "%s/%s/%s/installed_module_versions/%s" % (
            self.ui,
            dest_user_id,
            dest_app_id,
            imv_id,
        )

    def mcp_api_url(
        self, user_id: str = None, app_id: str = None, model_id: str = None, version_id: str = None
    ):
        """We have a special endpoint for MCP hosted models.

        Example:
          https://api.clarifai.com/v2/ext/mcp/v1/users/{user_id}/apps/{app_id}/models/{model_id}/versions/{version_id}

        Args:
          user_id: the author of the resource.
          app_id: the author's app the resource was created in.
          model_id: the resource ID
          version_id: the version of the resource.
        """
        if user_id is None:
            user_id = self.current_ctx.user_id
        if app_id is None:
            app_id = self.current_ctx.app_id
        if model_id is None:
            model_id = self.current_ctx.model_id
        if version_id is None:
            return "%s/v2/ext/mcp/v1/users/%s/apps/%s/models/%s" % (
                self.base,
                user_id,
                app_id,
                model_id,
            )
        return "%s/v2/ext/mcp/v1/users/%s/apps/%s/models/%s/versions/%s" % (
            self.base,
            user_id,
            app_id,
            model_id,
            version_id,
        )

    def openai_api_url(self):
        """We have a special endpoint for openAI compatible models.

        This doesn't include the /chat/completions suffix which the openAI client automatically
        adds.

        It also doesn't incldue the model which you an set as the model arg in an openAI client call
        using the clarifai_url() method below.

        Example:
          https://api.clarifai.com/v2/ext/openai/v1
        """
        return "%s/v2/ext/openai/v1" % self.base

    def api_url(
        self,
        user_id: str = None,
        app_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        version_id: str = None,
    ):
        """This is the path to the resource in the API.

        Example:
          https://api.clarifai.com/v2/zeiler/app/modules/module1/versions/2
          https://api.clarifai.com/v2/zeiler/app/models/model1/versions/2
          https://api.clarifai.com/v2/zeiler/app/concepts/concept1
          https://api.clarifai.com/v2/zeiler/app/workflows/workflow1
          https://api.clarifai.com/v2/zeiler/app/tasks/task1
          https://api.clarifai.com/v2/zeiler/app/installed_module_versions/module_manager_install

        Args:
          user_id: the author of the resource.
          app_id: the author's app the resource was created in.
          resource_type: the type of resource. One of "modules", "models", "concepts", "inputs", "workflows", "tasks"
          resource_id: the resource ID
        """
        if user_id is None:
            user_id = self.current_ctx.user_id
        if app_id is None:
            app_id = self.current_ctx.app_id
        if resource_id is None:
            raise ValueError("resource_id must be provided to get the API URL.")
        self._validate_resource_type(resource_type)
        if version_id is None:
            return "%s/v2/users/%s/apps/%s/%s/%s" % (
                self.base,
                user_id,
                app_id,
                resource_type,
                resource_id,
            )

        if resource_type in ["concepts", "tasks", "installed_module_versions"]:
            raise ValueError(f"{resource_type} do not have versions.")
        return "%s/v2/users/%s/apps/%s/%s/%s/versions/%s" % (
            self.base,
            user_id,
            app_id,
            resource_type,
            resource_id,
            version_id,
        )

    def _validate_resource_type(self, resource_type):
        if resource_type not in [
            "modules",
            "models",
            "concepts",
            "inputs",
            "workflows",
            "tasks",
            "installed_module_versions",
        ]:
            raise ValueError(
                "resource_type must be one of modules, models, concepts, inputs, workflows, tasks, installed_module_versions but was %s"
                % resource_type
            )

    def clarifai_url(
        self,
        user_id: str = None,
        app_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        version_id: str = None,
    ):
        """This is the path to the resource in community UI.

        Example:
          https://clarifai.com/zeiler/modules/module1/versions/2
          https://clarifai.com/zeiler/models/model1/versions/2
          https://clarifai.com/zeiler/concepts/concept1
          https://clarifai.com/zeiler/workflows/workflow1
          https://clarifai.com/zeiler/tasks/task1
          https://clarifai.com/zeiler/installed_module_versions/module_manager_install

        Args:
          user_id: the author of the resource.
          app_id: the author's app the resource was created in.
          resource_type: the type of resource. One of "modules", "models", "concepts", "inputs", "workflows", "tasks", "installed_module_versions"
          resource_id: the resource ID
          version_id: the version of the resource.
        """
        if user_id is None:
            user_id = self.current_ctx.user_id
        if app_id is None:
            app_id = self.current_ctx.app_id
        if resource_id is None:
            raise ValueError("resource_id must be provided to get the API URL.")
        self._validate_resource_type(resource_type)
        if version_id is None:
            return "%s/%s/%s/%s/%s" % (self.ui, user_id, app_id, resource_type, resource_id)
        return "%s/%s/%s/%s/%s/versions/%s" % (
            self.ui,
            user_id,
            app_id,
            resource_type,
            resource_id,
            version_id,
        )

    @classmethod
    def split_clarifai_app_url(cls, url: str):
        """
        clarifai.com uses fully qualified urls to resources.
        They are in the format of:
        https://clarifai.com/{user_id}/{app_id}/
        """
        url = url.replace("https://", "", 1).replace("http://", "", 1)
        o = urlparse(url)
        path = o.path
        path = path.lstrip("/")
        parts = path.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"Provided url must have 2 parts after the domain name. The current parts are: {parts}"
            )
        return tuple(parts[1:])

    @classmethod
    def split_clarifai_url(cls, url: str):
        """
        clarifai.com uses fully qualified urls to resources.
        They are in the format of:
        https://clarifai.com/{user_id}/{app_id}/{resource_type}/{resource_id}/{resource_version_type}/{resource_version_id}
        Those last two are optional.

        """
        url = url.replace("https://", "", 1).replace("http://", "", 1)
        o = urlparse(url)
        path = o.path
        path = path.lstrip("/")
        parts = path.split("/")
        if len(parts) != 5 and len(parts) != 7:
            raise ValueError(
                "Provided url must have 4 or 6 parts after the domain name. These are: {user_id}/{app_id}/{resource_type}/{resource_id}/{resource_version_type}/{resource_version_id}"
            )
        user_id, app_id, resource_type, resource_id = parts[1:5]
        if len(parts) == 7:
            resource_version_id = parts[6]
        else:
            resource_version_id = None
        return user_id, app_id, resource_type, resource_id, resource_version_id

    @classmethod
    def split_module_ui_url(cls, install: str):
        """Takes in a path like https://clarifai.com/zeiler/app/modules/module1/versions/2 to split it apart into it's IDs.

        Returns:
          user_id: the author of the module.
          app_id: the author's app the module was created in.
          module_id: the module ID
          module_version_id: the version of the module.
        """
        user_id, app_id, resource_type, resource_id, resource_version_id = cls.split_clarifai_url(
            install
        )

        if resource_type != "modules" or resource_version_id is None:
            raise ValueError(
                "Provided install url must have 6 parts after the domain name. These are {user_id}/{app_id}/modules/{module_id}/versions/{module_version_id}"
            )
        return user_id, app_id, resource_id, resource_version_id
