from collections import namedtuple

import pytest

from clarifai.urls.helper import ClarifaiUrlHelper


@pytest.fixture()
def helper():
  authObj = namedtuple("auth", "ui")
  auth = authObj(ui="http://fake")
  return ClarifaiUrlHelper(auth)


def test_module_Ui_url(helper):
  url = helper.module_ui_url("clarifai", "main", "module_1", "version_1")
  assert (url == "http://fake/clarifai/main/modules/module_1/module_versions/version_1")


def test_module_install_url(helper):
  install_url = helper.module_install_ui_url("person", "app", "XXX")
  assert (
      install_url ==
      "http://fake/person/app/installed_module_versions/module_manager_install/install?install=XXX"
  )


def test_install_with_custom_imv_id():
  auth_obj = namedtuple("auth", "ui")
  auth = auth_obj(ui="http://fake")

  custom_imv_id = "some_imv_id"
  helper = ClarifaiUrlHelper(auth, custom_imv_id)
  install_url = helper.module_install_ui_url("person", "app", "XXX")
  assert (install_url ==
          "http://fake/person/app/installed_module_versions/some_imv_id/install?install=XXX")


def test_split_of_ui_clarifai_url():
  (
      user_id,
      app_id,
      module_id,
      module_version_id,
  ) = ClarifaiUrlHelper.split_module_ui_url(
      "clarifai.com/clarifai/main/modules/module_1/module_versions/2")
  assert user_id == "clarifai"
  assert app_id == "main"
  assert module_id == "module_1"
  assert module_version_id == "2"


def test_split_with_https_prefix():
  (
      user_id,
      app_id,
      module_id,
      module_version_id,
  ) = ClarifaiUrlHelper.split_module_ui_url(
      "https://clarifai.com/clarifai/main/modules/module_1/module_versions/2")
  assert user_id == "clarifai"
  assert app_id == "main"
  assert module_id == "module_1"
  assert module_version_id == "2"


def test_too_many_things_happen():
  with pytest.raises(
      ValueError,
      match="Provided url must have 4 or 6 parts after the domain name",
  ):
    ClarifaiUrlHelper.split_module_ui_url(
        "https://clarifai.com/clarifai/main/modules/module_1/module_versions/2/something")


def test_not_enougth_items():
  with pytest.raises(
      ValueError, match="Provided url must have 4 or 6 parts after the domain name"):
    ClarifaiUrlHelper.split_module_ui_url("clarifai/main/modules/module_1/module_versions/2")


def test_no_right_modules_in_path():
  with pytest.raises(
      ValueError, match="Provided url must have 4 or 6 parts after the domain name"):
    ClarifaiUrlHelper.split_module_ui_url("clarifai/main/modules_abc/module_1/module_versions/2")


def test_no_right_module_version_in_path():
  with pytest.raises(
      ValueError, match="Provided url must have 4 or 6 parts after the domain name"):
    ClarifaiUrlHelper.split_module_ui_url("clarifai/main/modules/module_1/module_versions_abc/2")


def test_no_right_model_version_in_path():
  with pytest.raises(
      ValueError, match="Provided url must have 4 or 6 parts after the domain name"):
    ClarifaiUrlHelper.split_clarifai_url("clarifai/main/models/model_1/model_versions_abc/2")


def test_split_model_with_https_prefix():
  (
      user_id,
      app_id,
      resource_type,
      model_id,
      model_version_id,
  ) = ClarifaiUrlHelper.split_clarifai_url(
      "https://clarifai.com/clarifai/main/models/model_1/versions/2")
  assert user_id == "clarifai"
  assert app_id == "main"
  assert resource_type == "models"
  assert model_id == "model_1"
  assert model_version_id == "2"


def test_split_model_without_version_with_https_prefix():
  (
      user_id,
      app_id,
      resource_type,
      model_id,
      model_version_id,
  ) = ClarifaiUrlHelper.split_clarifai_url("https://clarifai.com/clarifai/main/models/model_1")
  assert user_id == "clarifai"
  assert app_id == "main"
  assert resource_type == "models"
  assert model_id == "model_1"
  assert model_version_id == None
