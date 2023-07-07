from urllib.parse import urlparse


class ClarifaiUrlHelper(object):
  """Lots of helper functionality for dealing with urls around modules."""

  def __init__(self, auth, module_manager_imv_id="module_manager_install"):
    """
        Args:
          auth: a ClarifaiAuthHelper object.
        """
    self._auth = auth
    self._module_manager_imv_id = module_manager_imv_id

  @property
  def auth(self):
    return self._auth

  def module_ui_url(self, user_id, app_id, module_id, module_version_id):
    """This is the path to the module in community."""
    return "%s/%s/%s/modules/%s/versions/%s" % (
        self.auth.ui,
        user_id,
        app_id,
        module_id,
        module_version_id,
    )

  def module_install_ui_url(self, dest_user_id, dest_app_id, module_url):
    """This is a url that allows for installation of the module from the community at 'module_url'
        into the destination app_id of the destination user_id."""
    return ("%s/%s/%s/installed_module_versions/%s/install?install=%s" %
            (self.auth.ui, dest_user_id, dest_app_id, self._module_manager_imv_id, module_url))

  def imv_ui_url(self, dest_user_id, dest_app_id, imv_id):
    return ("%s/%s/%s/installed_module_versions/%s" % (self.auth.ui, dest_user_id, dest_app_id,
                                                       imv_id))

  def clarifai_url(self, user_id, app_id, resource_type, resource_id, version_id: str = None):
    """This is the path to the resource in community.

        Args:
          user_id: the author of the resource.
          app_id: the author's app the resource was created in.
          resource_type: the type of resource. One of "modules", "models", "concepts", "inputs", "workflows", "tasks", "installed_module_versions"
          resource_id: the resource ID
          version_id: the version of the resource.
    """
    if resource_type not in [
        "modules", "models", "concepts", "inputs", "workflows", "tasks",
        "installed_module_versions"
    ]:
      raise ValueError(
          "resource_type must be one of modules, models, concepts, inputs, workflows, tasks, installed_module_versions but was %s"
          % resource_type)
    if version_id is None:
      return "%s/%s/%s/%s/%s" % (self.auth.ui, user_id, app_id, resource_type, resource_id)
    return "%s/%s/%s/%s/%s/versions/%s" % (self.auth.ui, user_id, app_id, resource_type,
                                           resource_id, version_id)

  @classmethod
  def split_clarifai_url(cls, url):
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
  def split_module_ui_url(cls, install):
    """Takes in a path like https://clarifai.com/zeiler/app/modules/module1/versions/2
        to split it apart into it's IDs.

        Returns:
          user_id: the author of the module.
          app_id: the author's app the module was created in.
          module_id: the module ID
          module_version_id: the version of the module.
        """
    user_id, app_id, resource_type, resource_id, resource_version_id = cls.split_clarifai_url(
        install)

    if resource_type != "modules" or resource_version_id is None:
      raise ValueError(
          "Provided install url must have 6 parts after the domain name. These are {user_id}/{app_id}/modules/{module_id}/versions/{module_version_id}"
      )
    return user_id, app_id, resource_id, resource_version_id
