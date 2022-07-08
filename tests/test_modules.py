from collections import namedtuple

import pytest
from clarifai_utils.modules.urls import ClarifaiModuleUrlHelper


class TestModules:

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def test_module_urls(self):

    authObj = namedtuple("auth", "ui")

    auth = authObj(ui="https://clarifai.com")

    helper = ClarifaiModuleUrlHelper(auth)

    url = helper.module_ui_url("clarifai", "main", "module_1", "version_1")
    assert (url == "https://clarifai.com/clarifai/main/modules/module_1/module_versions/version_1")

    install_url = helper.module_install_ui_url("person", "app", url)
    assert (
        install_url ==
        "https://clarifai.com/person/app/installed_module_versions/module_manager_install?page=install&install=%s"
        % url)

    custom_imv_id = "some_imv_id"
    helper = ClarifaiModuleUrlHelper(auth, custom_imv_id)
    install_url = helper.module_install_ui_url("person", "app", url)
    assert (
        install_url ==
        "https://clarifai.com/person/app/installed_module_versions/%s?page=install&install=%s" %
        (custom_imv_id, url))

  def test_slitting(self):

    (
        user_id,
        app_id,
        module_id,
        module_version_id,
    ) = ClarifaiModuleUrlHelper.split_module_ui_url(
        "clarifai.com/clarifai/main/modules/module_1/module_versions/2")
    assert user_id == "clarifai"
    assert app_id == "main"
    assert module_id == "module_1"
    assert module_version_id == "2"

    # Should work with https:// prefix.
    (
        user_id,
        app_id,
        module_id,
        module_version_id,
    ) = ClarifaiModuleUrlHelper.split_module_ui_url(
        "https://clarifai.com/clarifai/main/modules/module_1/module_versions/2")
    assert user_id == "clarifai"
    assert app_id == "main"
    assert module_id == "module_1"
    assert module_version_id == "2"

    # Too many things.
    with pytest.raises(ValueError):
      (
          user_id,
          app_id,
          module_id,
          module_version_id,
      ) = ClarifaiModuleUrlHelper.split_module_ui_url(
          "https://clarifai.com/clarifai/main/modules/module_1/module_versions/2/something")

    # Not enough things.
    with pytest.raises(ValueError):
      (
          user_id,
          app_id,
          module_id,
          module_version_id,
      ) = ClarifaiModuleUrlHelper.split_module_ui_url(
          "clarifai/main/modules/module_1/module_versions/2")

    # Not not right "modules" in path.
    with pytest.raises(ValueError):
      (
          user_id,
          app_id,
          module_id,
          module_version_id,
      ) = ClarifaiModuleUrlHelper.split_module_ui_url(
          "clarifai/main/modules_abc/module_1/module_versions/2")

    # Not not right "module_versions" in path.
    with pytest.raises(ValueError):
      (
          user_id,
          app_id,
          module_id,
          module_version_id,
      ) = ClarifaiModuleUrlHelper.split_module_ui_url(
          "clarifai/main/modules/module_1/module_versions_abc/2")
