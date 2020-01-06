import mock

from clarifai.rest import ClarifaiApp

from .mock_extensions import assert_request, mock_request

TINY_IMAGE_BASE64 = b'R0lGODlhAQABAIABAP///wAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_models_by_name(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "models": [{
    "id": "@modelID",
    "name": "celeb-v1.3",
    "created_at": "2016-10-25T19:30:38.541073Z",
    "app_id": "main",
    "output_info": {
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "concept",
      "type_ext": "facedetect-identity"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2016-10-25T19:30:38.541073Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      },
      "active_concept_count": 10554
    },
    "display_name": "Celebrity"
  }]
}
""")

  app = ClarifaiApp()

  models = app.models.search("celeb*")

  assert models[0].model_id == '@modelID'
  assert models[0].model_version == '@modelVersionID'

  assert_request(mock_execute_request, 'POST', '/v2/models/searches', """
{
  "model_query": {
    "name": "celeb*"
  }
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_models_by_name_and_type(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "models": [{
    "id": "@modelID",
    "name": "focus",
    "created_at": "2017-03-06T22:57:00.660603Z",
    "app_id": "main",
    "output_info": {
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "blur",
      "type_ext": "focus"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2017-03-06T22:57:00.684652Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      }
    },
    "display_name": "Focus"
  }]
}
""")

  app = ClarifaiApp()

  models = app.models.search("*", 'focus')

  assert models[0].model_id == '@modelID'
  assert models[0].model_version == '@modelVersionID'

  assert_request(mock_execute_request, 'POST', '/v2/models/searches', """
{
  "model_query": {
    "name": "*",
    "type": "focus"
  }
}
  """)
