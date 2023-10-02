import os

import pytest

from clarifai.client.search import Search
from clarifai.client.user import User
from clarifai.errors import UserError

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = "ci_search_app"
DOG_IMG_URL = "https://samples.clarifai.com/dog.tiff"


@pytest.fixture
def client():
  return User(user_id=CREATE_APP_USER_ID)


@pytest.fixture
def search():
  return Search(user_id=CREATE_APP_USER_ID, app_id=CREATE_APP_ID, top_k=1)


test_filters = [{
    "geo_point": {
        "longitude": -29.0,
        "latitude": 40.0,
        "geo_limit": 10
    }
}, {
    "concepts": [{
        "name": "dog",
        "value": 1
    }]
}, {
    "text_raw": "a dog"
}, {
    "image_url": [DOG_IMG_URL]
}]


class TestAnnotationSearch:
  __test__ = False

  def __init__(self, client, search) -> None:
    self.client = client
    self.search = search
    self.app_id = CREATE_APP_ID
    self.dog_image_url = DOG_IMG_URL
    self.geo_info = [-30.0, 40.0]  # longitude, latitude
    self.concepts = ["dog"]

    self.upload_input()

  def upload_input(self):
    inp_obj = self.client.create_app(app_id=self.app_id, base_workflow="Universal").inputs()
    input_proto = inp_obj.get_input_from_url(
        input_id="dog-tiff",
        image_url=self.dog_image_url,
        labels=self.concepts,
        geo_info=self.geo_info)
    inp_obj.upload_inputs([input_proto])

  @pytest.mark.parametrize("filter", test_filters)
  def test_filter_search(self, filter: dict):
    query = self.search.query(filters=[filter])
    for q in query:
      assert q.hits.input.id == "dog-tiff"

  def test_rank_search(self):
    query = self.search.query(ranks=[{"image_url": ["https://samples.clarifai.com/dog.tiff"]}])
    for q in query:
      assert q.hits.input.id == "dog-tiff"

  def test_schema_error(self):
    with pytest.raises(UserError):
      _ = self.search.query(filters=[{
          "geo_point": {
              "longitude": -29.0,
              "latitude": 40.0,
              "geo_limit": 10,
              "extra": 1
          }
      }])

    # Incorrect Concept Keys
    with pytest.raises(UserError):
      _ = self.search.query(filters=[{
          "concepts": [{
              "value": 1,
              "concept_id": "deer"
          }, {
              "name": "dog",
              "value": 1
          }]
      }])

    # Incorrect Concept Values
    with pytest.raises(UserError):
      _ = self.search.query(filters=[{
          "concepts": [{
              "name": "deer",
              "value": 2
          }, {
              "name": "dog",
              "value": 1
          }]
      }])

  def teardown_class(self):
    self.client.delete_app(app_id=CREATE_APP_ID)


def test_search(client, search):
  tests = TestAnnotationSearch(client, search)

  # Run tests
  tests.test_filter_search()
  tests.test_schema_error()

  # Teardown
  tests.teardown_class()
