from clarifai_utils.auth.helper import ClarifaiAuthHelper


class TestAuth:

  @classmethod
  def setUpClass(cls):
    pass

  @classmethod
  def tearDownClass(cls):
    pass

  def test_ui_urls(self):
    default = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
    assert default.ui == "https://clarifai.com"

    default = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="localhost:3002")
    assert default.ui == "http://localhost:3002"

    default = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="http://localhost:3002")
    assert default.ui == "http://localhost:3002"

    default = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="https://localhost:3002")
    assert default.ui == "https://localhost:3002"

    default = ClarifaiAuthHelper("clarifai", "main", "fake_pat", ui="https://clarifai.com")
    assert default.ui == "https://clarifai.com"
