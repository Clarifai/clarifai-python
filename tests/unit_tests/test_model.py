import unittest.mock as mock
import pytest
import requests

from clarifai.errors import ApiError, UserError
from clarifai.rest import ClarifaiApp

from .mock_extensions import (assert_request, assert_requests, mock_request,
                              mock_request_with_failed_response)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_create_model(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "model": {
    "id": "@modelID",
    "name": "@modelName",
    "created_at": "2019-01-22T11:54:12.375436048Z",
    "app_id": "@appID",
    "output_info": {
      "output_config": {
        "concepts_mutually_exclusive": false,
        "closed_environment": false,
        "max_concepts": 0,
        "min_value": 0
      },
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "concept",
      "type_ext": "concept"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2019-01-22T11:54:12.406406642Z",
      "status": {
        "code": 21102,
        "description": "Model not yet trained"
      },
      "active_concept_count": 2,
      "train_stats": {}
    }
  }
}
"""
      ])

  app = ClarifaiApp()
  model = app.models.create(
      '@modelID',
      '@modelName',
      concepts=['dog', 'cat'],
      concepts_mutually_exclusive=False,
      closed_environment=False)

  assert model.model_status_code == 21102
  assert model.model_id == '@modelID'
  assert model.model_version == '@modelVersionID'

  assert_request(mock_execute_request, 'POST', '/v2/models', """
{
  "model": {
    "id": "@modelID",
    "name": "@modelName",
    "output_info": {
      "data": {
        "concepts": [
          {
            "id": "dog",
            "value": 1.0
          },
          {
            "id": "cat",
            "value": 1.0
          }
        ]
      },
      "output_config": {}
    }
  }
}
      """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_model(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "model": {
    "id": "@modelID",
    "name": "some-model-name",
    "created_at": "2017-05-16T19:20:38.733764Z",
    "app_id": "main",
    "output_info": {
      "data": {
        "concepts": [{
          "id": "@conceptID",
          "name": "safe",
          "created_at": "2017-05-16T19:20:38.450157Z",
          "language": "en",
          "app_id": "main"
        }]
      },
      "type": "concept",
      "type_ext": "concept"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2017-05-16T19:20:38.733764Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      },
      "active_concept_count": 5
    },
    "display_name": "Moderation"
  }
}
""", """
{
  "status": {
    "code": 10000,
      "description": "Ok"
    },
  "models": []
}
      """
      ])

  app = ClarifaiApp()
  model = app.models.get('@modelID')

  assert model.model_status_code == 21100
  assert model.model_id == '@modelID'
  assert model.model_version == '@modelVersionID'
  assert model.concepts[0].concept_id == '@conceptID'

  assert_request(mock_execute_request, 'GET', '/v2/models/@modelID', '{}')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_non_existing_model(mock_http_client):  # type: (mock.Mock) -> None
  response = requests.Response()
  response.status_code = 401
  mock_execute_request = mock_request_with_failed_response(
      mock_http_client,
      json_responses=[ApiError("/v2/models/@nonexistentID", {}, "GET", response)])

  app = ClarifaiApp()
  with pytest.raises(ApiError):
    app.models.get("@nonexistentID")

  assert_request(mock_execute_request, 'GET', '/v2/models/@nonexistentID')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_all_models(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
      "description": "Ok"
    },
  "models": [
    {
      "id": "@modelID1",
      "name": "@modelName1",
      "created_at": "2019-01-16T23:33:46.605294Z",
      "app_id": "main",
      "output_info": {
        "message": "Show output_info with: GET /models/{model_id}/output_info",
        "type": "detect",
        "type_ext": "detect"
      },
      "model_version": {
        "id": "28b2ff6148684aa2b18a34cd004b4fac",
        "created_at": "2019-01-16T23:33:46.605294Z",
        "status": {
          "code": 21100,
          "description": "Model trained successfully"
        },
        "train_stats": {}
      },
      "display_name": "Face Detection"
    },
    {
      "id": "@modelID2",
      "name": "@modelName2",
      "created_at": "2019-01-16T23:33:46.605294Z",
      "app_id": "main",
      "output_info": {
        "message": "Show output_info with: GET /models/{model_id}/output_info",
        "type": "embed",
        "type_ext": "detect-embed"
      },
      "model_version": {
        "id": "fc6999e5eb274dfdba826f6b1c7ffdab",
        "created_at": "2019-01-16T23:33:46.605294Z",
        "status": {
          "code": 21100,
          "description": "Model trained successfully"
        },
        "train_stats": {}
      },
      "display_name": "Face Embedding"
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
      "description": "Ok"
    },
  "models": []
}
      """
      ])

  app = ClarifaiApp()
  models = list(app.models.get_all())

  assert models[0].model_id == '@modelID1'
  assert models[0].model_name == '@modelName1'
  assert models[0].output_info['type_ext'] == 'detect'

  assert models[1].model_id == '@modelID2'
  assert models[1].model_name == '@modelName2'
  assert models[1].output_info['type_ext'] == 'detect-embed'

  assert_requests(mock_execute_request, [
      ('GET', '/v2/models', '{"per_page": 20, "page": 1}'),
      ('GET', '/v2/models', '{"per_page": 20, "page": 2}'),
  ])

  # An empty list should be returned in the case of an emtpy JSON
  mock_request(
      mock_http_client,
      json_responses=[
          """
         {
           "status": {
             "code": 10000,
               "description": "Ok"
             },
           "models": []
         }
         """
      ])
  models = list(app.inputs.get_all())
  assert not models


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_model_inputs(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
    "status": {
        "code": 10000,
        "description": "Ok"
    },
    "inputs": [{
        "id": "@inputID",
        "data": {
            "image": {
                "url": "@imageURL"
            },
            "concepts": [{
                "id": "@conceptID",
                "name": "@conceptName",
                "value": 1,
                "app_id": "@conceptAppID"
            }]
        },
        "created_at": "2017-10-15T16:30:52.964888Z",
        "status": {
            "code": 30000,
            "description": "Download complete"
        }
    }]
}
  """)

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  inputs = model.get_inputs()

  assert inputs['inputs'][0]['id'] == '@inputID'
  assert inputs['inputs'][0]['data']['image']['url'] == '@imageURL'

  assert_request(mock_execute_request, 'GET', '/v2/models/@modelID/inputs', """
{"page": 1, "per_page": 20}
""")


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_delete_model(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  }
}
  """)

  app = ClarifaiApp()
  response = app.models.delete('@modelID')

  assert response['status']['code'] == 10000

  assert_request(mock_execute_request, 'DELETE', '/v2/models/@modelID', '{}')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_delete_all_models(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  }
}
  """)

  app = ClarifaiApp()
  response = app.models.delete_all()

  assert response['status']['code'] == 10000

  assert_request(mock_execute_request, 'DELETE', '/v2/models', '{"delete_all": true}')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_update_model(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "models": [{
    "id": "@modelID",
    "name": "@newModelName",
    "created_at": "2017-11-27T08:35:13.911899Z",
    "app_id": "@appID",
    "output_info": {
      "output_config": {
        "concepts_mutually_exclusive": true,
        "closed_environment": true
      },
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "concept",
      "type_ext": "concept"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2017-11-27T08:35:14.298376733Z",
      "status": {
        "code": 21102,
        "description": "Model not yet trained"
      }
    }
  }]
}
  """)

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')

  updated_model = model.update(
      model_name='@newModelName',
      concept_ids=['@conceptID1'],
      concepts_mutually_exclusive=True,
      closed_environment=True)

  assert updated_model.model_name == '@newModelName'

  assert_request(mock_execute_request, 'PATCH', '/v2/models', """
 {
  "models": [
    {
      "id": "@modelID",
      "name": "@newModelName",
      "output_info": {
        "data": {
          "concepts": [
            {
              "id": "@conceptID1",
              "value": 1.0
            }
          ]
        },
        "output_config": {
          "concepts_mutually_exclusive": true,
          "closed_environment": true
        }
      }
    }
  ],
  "action": "merge"
}
""")

  # User Error should be thrown if unsupported action is given
  with pytest.raises(UserError):
    model.update(model_name='@newModelName', action='splice')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_train_model(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "model": {
    "id": "@modelID",
    "name": "@modelName",
    "created_at": "2019-01-20T15:51:21.641006Z",
    "app_id": "@appID",
    "output_info": {
      "output_config": {
        "concepts_mutually_exclusive": false,
        "closed_environment": false,
        "max_concepts": 0,
        "min_value": 0
      },
      "message": "Show output_info with: GET /models/{model_id}/output_info",
      "type": "concept",
      "type_ext": "concept"
    },
    "model_version": {
      "id": "@modelVersionID",
      "created_at": "2019-01-20T15:51:25.093744401Z",
      "status": {
        "code": 21100,
        "description": "Model trained successfully"
      },
      "active_concept_count": 2,
      "train_stats": {}
    }
  }
}
  """)

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  updated_model = model.train(timeout=120, raise_on_timeout=True)

  assert updated_model.model_id == '@modelID'
  assert updated_model.model_name == '@modelName'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/versions', '{}')
