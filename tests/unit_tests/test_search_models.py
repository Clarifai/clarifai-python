import unittest.mock as mock

from clarifai.rest import ClarifaiApp

from .mock_extensions import assert_request, mock_request

TINY_IMAGE_BASE64 = b'R0lGODlhAQABAIABAP///wAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_models_by_name(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """

{
    "status": {
        "code": 10000,
        "description": "Ok",
        "req_id": "08e649a6116f4f56992e1676b25dcde6"
    },
    "models": [
        {
            "id": "@modelID",
            "name": "moderation",
            "created_at": "2017-05-12T21:28:00.471607Z",
            "app_id": "main",
            "output_info": {
                "message": "Show output_info with: GET /models/{model_id}/output_info",
                "type": "concept",
                "type_ext": "concept"
            },
            "model_version": {
                "id": "@modelVersionID",
                "created_at": "2017-10-26T20:29:09.263232Z",
                "status": {
                    "code": 21100,
                    "description": "Model is trained and ready"
                },
                "active_concept_count": 5,
                "worker_id": "8b7c05a25ce04d0490367390665f1526"
            },
            "display_name": "Moderation"
        }
    ]
}
""")

  app = ClarifaiApp()

  models = app.models.search("moderation*")

  assert models[0].model_id == '@modelID'
  assert models[0].model_version == '@modelVersionID'

  assert_request(mock_execute_request, 'POST', '/v2/models/searches', """
{
  "model_query": {
    "name": "moderation*"
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
    "name": "color",
    "created_at": "2016-05-11T18:05:45.924367Z",
    "app_id": "main",
    "output_info": {
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "color",
      "type_ext": "color"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2016-07-13T01:19:12.147644Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      }
    },
    "display_name": "Color"
  }]
}
""")

  app = ClarifaiApp()

  models = app.models.search("*", 'color')

  assert models[0].model_id == '@modelID'
  assert models[0].model_version == '@modelVersionID'

  assert_request(mock_execute_request, 'POST', '/v2/models/searches', """
{
  "model_query": {
    "name": "*",
    "type": "color"
  }
}
  """)
