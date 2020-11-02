import unittest.mock as mock

from clarifai.rest import ClarifaiApp, Image

from .mock_extensions import assert_request, mock_request


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_predict_with_invalid_url(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10020,
    "description": "Failure"
  },
  "outputs": [
  {
    "id": "@outputID",
    "status": {
      "code": 30002,
      "description": "Download failed; check URL",
      "details": "404 Client Error: Not Found for url: @invalidURL"
    },
    "created_at": "2019-01-20T19:39:15.460417224Z",
    "model": {
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
        },
        "train_stats": {}
      },
      "display_name": "Color"
    },
    "input": {
      "id": "@inputID",
      "data": {
        "image": {
          "url": "@invalidURL"
        }
      }
    },
    "data": {}
  }
  ]
}
""")

  app = ClarifaiApp()
  model = app.public_models.general_model
  response = model.predict_by_url('@invalidURL')

  assert response['status']['code'] == 10020

  assert_request(mock_execute_request, 'POST', '/v2/models/' + model.model_id + '/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@invalidURL"
        }
      }
    }
  ]
}
        """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_predict_multiple_with_one_invalid_url(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10010,
    "description": "Mixed Success"
  },
  "outputs": [
    {
      "id": "@outputID1",
      "status": {
        "code": 10000,
        "description": "Ok"
      },
      "created_at": "2019-01-20T20:25:20.302505245Z",
      "model": {
        "id": "@modelID",
        "name": "general",
        "created_at": "2016-03-09T17:11:39.608845Z",
        "app_id": "main",
        "output_info": {
          "message": "Show output_info with: GET /models/{model_id}/output_info",
          "type": "concept",
          "type_ext": "concept"
        },
        "model_version": {
          "id": "@modelVersionID",
          "created_at": "2016-07-13T01:19:12.147644Z",
          "status": {
            "code": 21100,
            "description": "Model trained successfully"
          },
          "train_stats": {}
        },
        "display_name": "General"
      },
      "input": {
        "id": "@inputID1",
        "data": {
          "image": {
            "url": "@validURL"
          }
        }
      },
      "data": {
        "concepts": [
          {
            "id": "@concept1",
            "name": "people",
            "value": 0.9963381,
            "app_id": "main"
          },
          {
            "id": "@concept2",
            "name": "one",
            "value": 0.9879057,
            "app_id": "main"
          }
        ]
      }
    },
    {
      "id": "@outputID2",
      "status": {
        "code": 30002,
        "description": "Download failed; check URL",
        "details": "404 Client Error: Not Found for url: @invalidURL"
      },
      "created_at": "2019-01-20T20:25:20.302505245Z",
      "model": {
        "id": "@modelID",
        "name": "general",
        "created_at": "2016-03-09T17:11:39.608845Z",
        "app_id": "main",
        "output_info": {
          "message": "Show output_info with: GET /models/{model_id}/output_info",
          "type": "concept",
          "type_ext": "concept"
        },
        "model_version": {
          "id": "@modelVersionID",
          "created_at": "2016-07-13T01:19:12.147644Z",
          "status": {
            "code": 21100,
            "description": "Model trained successfully"
          },
          "train_stats": {}
        },
        "display_name": "General"
      },
      "input": {
        "id": "@inputID2",
        "data": {
          "image": {
            "url": "@invalidURL"
          }
        }
      },
      "data": {}
    }
  ]
}
""")

  app = ClarifaiApp()
  model = app.public_models.general_model
  response = model.predict([Image(url='@validURL'), Image(url='@invalidURL')])

  assert response['status']['code'] == 10010
  assert response['status']['description'] == 'Mixed Success'

  assert_request(mock_execute_request, 'POST', '/v2/models/' + model.model_id + '/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@validURL"
        }
      }
    },
    {
      "data": {
        "image": {
          "url": "@invalidURL"
        }
      }
    }
  ]
}
        """)
