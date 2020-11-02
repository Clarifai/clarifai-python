import unittest.mock as mock

from clarifai.rest import ClarifaiApp

from .mock_extensions import assert_request, mock_request


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_model_version(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "model_version": {
    "id": "@modelVersionID",
    "created_at": "2017-10-31T16:30:31.226185Z",
    "status": {
      "code": 21100,
      "description": "Model trained successfully"
    },
    "active_concept_count": 5,
    "train_stats": {}
  }
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.get_version('@modelVersionID')

  assert response['model_version']['id'] == '@modelVersionID'
  assert response['model_version']['status']['code'] == 21100

  assert_request(mock_execute_request, 'GET', '/v2/models/@modelID/versions/@modelVersionID')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_model_versions(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "model_versions": [
    {
      "id": "@modelVersionID1",
      "created_at": "2017-10-31T16:30:31.226185Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      },
      "active_concept_count": 5,
      "train_stats": {}
    },
    {
      "id": "@modelVersionID2",
      "created_at": "2017-05-16T19:20:38.733764Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      },
      "active_concept_count": 5,
      "train_stats": {}
    }
  ]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.list_versions()

  assert response['model_versions'][0]['id'] == '@modelVersionID1'
  assert response['model_versions'][0]['status']['code'] == 21100
  assert response['model_versions'][0]['active_concept_count'] == 5

  assert response['model_versions'][1]['id'] == '@modelVersionID2'
  assert response['model_versions'][1]['status']['code'] == 21100
  assert response['model_versions'][1]['active_concept_count'] == 5

  assert_request(mock_execute_request, 'GET', '/v2/models/@modelID/versions', """
{"page": 1, "per_page": 20}
""")


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_delete_model_version(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """{
  "status": {
    "code": 10000,
    "description": "Ok"
  }
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.delete_version('@modelVersionID')

  assert response['status']['code'] == 10000

  assert_request(mock_execute_request, 'DELETE', '/v2/models/@modelID/versions/@modelVersionID')
