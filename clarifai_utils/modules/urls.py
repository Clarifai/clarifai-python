from urllib.parse import urlparse


class ClarifaiModuleUrlHelper(object):
  """ Lots of helper functionality for dealing with urls around modules.
  """

  def __init__(self, auth):
    """
    Args:
      auth: a ClarifaiAuthHelper object.
    """
    self._auth = auth

  @property
  def auth(self):
    return self._auth

  def module_ui_url(self, user_id, app_id, module_id, module_version_id):
    """ This is the path to the module in community. """
    return "%s/%s/%s/modules/%s/module_versions/%s" % (self.auth.ui, user_id, app_id, module_id,
                                                       module_version_id)

  def module_install_ui_url(self, dest_user_id, dest_app_id, module_url):
    """ This is a url that allows for installation of the module from the community at 'module_url'
    into the destination app_id of the destination user_id. """
    return "%s/%s/%s/installed_module_versions/module_manager_install?page=install&install=%s" % (
        self.auth.ui, dest_user_id, dest_app_id, module_url)

  @classmethod
  def split_module_ui_url(cls, install):
    """ Takes in a path like https://clarifai.com/zeiler/app/modules/module1/module_versions/2
    to split it apart into it's IDs.

    Returns:
      user_id: the author of the module.
      app_id: the author's app the module was created in.
      module_id: the module ID
      module_version_id: the version of the module.
    """
    path = install
    if install.startswith("https://") or install.startswith("http://"):  # strip off the domain
      o = urlparse(install)
      path = o.path
    path = path.lstrip('/')
    parts = path.split('/')
    if len(parts) != 6:
      st.error(
          "Provided install url must have 6 parts after the domain name. These are {user_id}/{app_id}/modules/{module_id}/module_versions/{module_version_id}"
      )
      st.stop()
    user_id = parts[0]
    app_id = parts[1]
    module_id = parts[3]
    module_version_id = parts[5]
    return user_id, app_id, module_id, module_version_id
