from unittest import mock
from unittest.mock import Mock

import pytest as pytest

from clarifai.client.auth.helper import ClarifaiAuthHelper, clear_cache


@pytest.fixture(autouse=True)
def clear_caches():
  clear_cache()


def test_ui_default_url():
  default = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  assert default.ui == "https://clarifai.com"
  assert default.pat == "fake_pat"


@pytest.mark.parametrize(("input_url", "expected_url"), (
    ("http://localhost:3002", "http://localhost:3002"),
    ("https://localhost:3002", "https://localhost:3002"),
    ("https://clarifai.com", "https://clarifai.com"),
))
def test_ui_urls(input_url, expected_url):
  helper = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui=input_url)
  assert helper.ui == expected_url


def test_passing_no_schema_url_use_https_when_server_is_running():

  def raise_exception():
    return Mock()

  with mock.patch('urllib.request.urlopen', new_callable=raise_exception):
    helper = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="server")
    assert helper.ui == "https://server"


def test_passing_no_schema_url_show_error_when_not_server_running():

  def raise_exception():
    return Mock(side_effect=Exception("http_exception"))

  with mock.patch('urllib.request.urlopen', new_callable=raise_exception):
    with pytest.raises(
        Exception,
        match="Could not get a valid response from url: localhost:3002, is the API running there?"
    ):
      ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="localhost:3002")


def test_passing_no_schema_url_detect_http_when_SSL_in_error():

  def raise_exception():
    return Mock(side_effect=Exception("Has SSL in error"))

  with mock.patch('urllib.request.urlopen', new_callable=raise_exception):
    helper = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="localhost:3002")
    assert helper.ui == "http://localhost:3002"


def test_passing_no_schema_url_require_port():

  def raise_exception():
    return Mock(side_effect=Exception("Has SSL in error"))

  with mock.patch('urllib.request.urlopen', new_callable=raise_exception):
    with pytest.raises(
        Exception, match="When providing an insecure url it must have both host:port format"):
      ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="localhost")


def test_exception_empty_user():
  ClarifaiAuthHelper("", "main", "fake_pat", validate=False)
  with pytest.raises(
      Exception,
      match="Need 'user_id' to not be empty in the query params or user CLARIFAI_USER_ID env var"):
    ClarifaiAuthHelper("", "main", "fake_pat")


def test_exception_empty_app():
  ClarifaiAuthHelper("clarifai", "", "fake_pat", validate=False)
  with pytest.raises(
      Exception,
      match="Need 'app_id' to not be empty in the query params or user CLARIFAI_APP_ID env var"):
    ClarifaiAuthHelper("clarifai", "", "fake_pat")


def test_exception_empty_pat():
  ClarifaiAuthHelper("clarifai", "main", "", validate=False)
  with pytest.raises(
      Exception,
      match=
      "Need 'pat' or 'token' in the query params or use one of the CLARIFAI_PAT or CLARIFAI_SESSION_TOKEN env vars"
  ):
    ClarifaiAuthHelper("clarifai", "main", "")
