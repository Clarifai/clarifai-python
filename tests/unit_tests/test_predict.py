# -*- coding: utf-8 -*-

import unittest.mock as mock

from clarifai.rest import ClarifaiApp, Image, ModelOutputConfig, ModelOutputInfo

from .mock_extensions import assert_request, mock_request


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_concept_predict(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "outputs": [{
    "id": "@outputID",
    "status": {
      "code": 10000,
      "description": "Ok"
    },
    "created_at": "2017-11-17T19:32:58.760477937Z",
    "model": {
      "id": "@modelID",
      "name": "@modelName",
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
        }
      },
      "display_name": "@modelDisplayName"
    },
    "input": {
      "id": "@inputID",
      "data": {
        "image": {
          "url": "@imageUrl"
      }
    }
    },
    "data": {
       "concepts": [{
         "id": "@conceptID1",
         "name": "@conceptName1",
         "value": 0.99,
         "app_id": "main"
       }, {
         "id": "@conceptID2",
         "name": "@conceptName2",
         "value": 0.98,
         "app_id": "main"
       }]
    }
  }]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict_by_url('@url')

  output = response['outputs'][0]
  assert output['id'] == '@outputID'
  assert output['input']['id'] == '@inputID'
  assert output['data']['concepts'][0]['id'] == '@conceptID1'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@url"
        }
      }
    }
  ]
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_concept_predict_with_arguments(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "outputs": [
    {
      "id": "@outputID",
      "status": {
        "code": 10000,
        "description": "Ok"
      },
      "created_at": "2019-01-29T17:15:32.450063489Z",
      "model": {
        "id": "@modelID",
        "name": "@modelName",
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
        "id": "@inputID",
        "data": {
          "image": {
            "url": "@url"
          }
        }
      },
      "data": {
        "concepts": [
          {
            "id": "@conceptID1",
            "name": "menschen",
            "value": 0.9963381,
            "app_id": "main"
          },
          {
            "id": "@conceptID2",
            "name": "ein",
            "value": 0.9879057,
            "app_id": "main"
          },
          {
            "id": "@conceptID3",
            "name": "Porträt",
            "value": 0.98490834,
            "app_id": "main"
          }
        ]
      }
    }
  ]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict_by_url('@url', lang='de', max_concepts=3, min_value=0.5)

  output = response['outputs'][0]
  assert output['input']['id'] == '@inputID'
  assert output['id'] == '@outputID'
  assert output['data']['concepts'][0]['id'] == '@conceptID1'

  assert output['model']['id'] == '@modelID'
  assert output['model']['name'] == '@modelName'
  assert output['model']['model_version']['id'] == '@modelVersionID'
  assert output['model']['output_info']['type_ext'] == 'concept'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@url"
        }
      }
    }
  ],
  "model": {
    "output_info": {
      "output_config": {
        "language": "de",
        "max_concepts": 3,
        "min_value": 0.5
      }
    }
  }
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_concept_bulk_predict(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "outputs": [{
    "id": "@outputID1",
    "status": {
      "code": 10000,
      "description": "Ok"
    },
    "created_at": "2017-11-17T19:32:58.760477937Z",
    "model": {
      "id": "@modelID1",
      "name": "@modelName1",
      "created_at": "2016-03-09T17:11:39.608845Z",
      "app_id": "main",
      "output_info": {
        "message": "Show output_info with: GET /models/{model_id}/output_info",
        "type": "concept",
        "type_ext": "concept"
      },
      "model_version": {
        "id": "@modelVersionID1",
        "created_at": "2016-07-13T01:19:12.147644Z",
        "status": {
          "code": 21100,
          "description": "Model trained successfully"
        }
      },
      "display_name": "@modelDisplayName1"
    },
    "input": {
      "id": "@inputID1",
      "data": {
        "image": {
          "url": "@imageUrl1"
        }
      }
    },
    "data": {
      "concepts": [{
        "id": "@conceptID11",
        "name": "@conceptName11",
        "value": 0.99,
        "app_id": "main"
      }, {
        "id": "@conceptID12",
        "name": "@conceptName12",
        "value": 0.98,
        "app_id": "main"
      }]
    }
  },
  {
    "id": "@outputID2",
    "status": {
      "code": 10000,
      "description": "Ok"
    },
    "created_at": "2017-11-17T19:32:58.760477937Z",
    "model": {
      "id": "@modelID2",
      "name": "@modelName2",
      "created_at": "2016-03-09T17:11:39.608845Z",
      "app_id": "main",
      "output_info": {
        "message": "Show output_info with: GET /models/{model_id}/output_info",
        "type": "concept",
        "type_ext": "concept"
      },
      "model_version": {
        "id": "@modelVersionID2",
        "created_at": "2016-07-13T01:19:12.147644Z",
        "status": {
          "code": 21100,
          "description": "Model trained successfully"
        }
      },
      "display_name": "@modelDisplayName2"
    },
    "input": {
      "id": "@inputID2",
      "data": {
        "image": {
          "url": "@imageUrl2"
        }
      }
    },
    "data": {
      "concepts": [{
        "id": "@conceptID21",
        "name": "@conceptName21",
        "value": 0.99,
        "app_id": "main"
      }, {
        "id": "@conceptID22",
        "name": "@conceptName22",
        "value": 0.98,
        "app_id": "main"
      }]
    }
  }]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict([Image(url='@url1'), Image(url='@url2')])

  output1 = response['outputs'][0]
  assert output1['input']['id'] == '@inputID1'
  assert output1['id'] == '@outputID1'
  assert output1['data']['concepts'][0]['id'] == '@conceptID11'
  assert output1['data']['concepts'][1]['id'] == '@conceptID12'

  output2 = response['outputs'][1]
  assert output2['input']['id'] == '@inputID2'
  assert output2['id'] == '@outputID2'
  assert output2['data']['concepts'][0]['id'] == '@conceptID21'
  assert output2['data']['concepts'][1]['id'] == '@conceptID22'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@url1"
        }
      }
    },
    {
      "data": {
        "image": {
          "url": "@url2"
        }
      }
    }
  ]
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_concept_bulk_predict_with_arguments(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "outputs": [
    {
      "id": "@outputID1",
      "status": {
        "code": 10000,
        "description": "Ok"
      },
      "created_at": "2019-01-29T16:45:43.793810775Z",
      "model": {
        "id": "aaa03c23b3724a16a56b629203edc62c",
        "name": "general",
        "created_at": "2016-03-09T17:11:39.608845Z",
        "app_id": "main",
        "output_info": {
          "message": "Show output_info with: GET /models/{model_id}/output_info",
          "type": "concept",
          "type_ext": "concept"
        },
        "model_version": {
          "id": "aa9ca48295b37401f8af92ad1af0d91d",
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
            "url": "https://clarifai.com/developer/static/images/model-samples/celeb-001.jpg"
          }
        }
      },
      "data": {
        "concepts": [
          {
            "id": "@conceptID11",
            "name": "menschen",
            "value": 0.9963381,
            "app_id": "main"
          },
          {
            "id": "@conceptID12",
            "name": "ein",
            "value": 0.9879057,
            "app_id": "main"
          }
        ]
      }
    },
    {
      "id": "@outputID2",
      "status": {
        "code": 10000,
        "description": "Ok"
      },
      "created_at": "2019-01-29T16:45:43.793810775Z",
      "model": {
        "id": "aaa03c23b3724a16a56b629203edc62c",
        "name": "general",
        "created_at": "2016-03-09T17:11:39.608845Z",
        "app_id": "main",
        "output_info": {
          "message": "Show output_info with: GET /models/{model_id}/output_info",
          "type": "concept",
          "type_ext": "concept"
        },
        "model_version": {
          "id": "aa9ca48295b37401f8af92ad1af0d91d",
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
            "url": "https://clarifai.com/developer/static/images/model-samples/apparel-001.jpg"
          }
        }
      },
      "data": {
        "concepts": [
          {
            "id": "@conceptID21",
            "name": "brillen und kontaktlinsen",
            "value": 0.99984586,
            "app_id": "main"
          },
          {
            "id": "@conceptID22",
            "name": "linse",
            "value": 0.999823,
            "app_id": "main"
          }
        ]
      }
    }
  ]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict(
      [Image(url='@url1'), Image(url='@url2')],
      ModelOutputInfo(
          output_config=ModelOutputConfig(language='de', max_concepts=2, min_value=0.5)))

  output1 = response['outputs'][0]
  assert output1['input']['id'] == '@inputID1'
  assert output1['id'] == '@outputID1'
  assert output1['data']['concepts'][0]['id'] == '@conceptID11'
  assert output1['data']['concepts'][1]['id'] == '@conceptID12'

  output2 = response['outputs'][1]
  assert output2['input']['id'] == '@inputID2'
  assert output2['id'] == '@outputID2'
  assert output2['data']['concepts'][0]['id'] == '@conceptID21'
  assert output2['data']['concepts'][1]['id'] == '@conceptID22'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@url1"
        }
      }
    },
    {
      "data": {
        "image": {
          "url": "@url2"
        }
      }
    }
  ],
  "model": {
    "output_info": {
      "output_config": {
        "language": "de",
        "max_concepts": 2,
        "min_value": 0.5
      }
    }
  }
}
  """)


# To be future-proof against expansion, response objects with unknown fields should be
# parsed correctly and unknown fields ignored.
@mock.patch('clarifai.rest.http_client.HttpClient')
def test_predict_with_unknown_response_fields(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok",
    "unknown_field": "val"
  },
  "outputs": [{
    "id": "@outputID",
    "status": {
      "code": 10000,
      "description": "Ok"
    },
    "created_at": "2017-11-17T19:32:58.760477937Z",
    "model": {
      "id": "@modelID",
      "name": "@modelName",
      "created_at": "2016-03-09T17:11:39.608845Z",
      "app_id": "main",
      "output_info": {
        "message": "Show output_info with: GET /models/{model_id}/output_info",
        "type": "concept",
        "type_ext": "concept",
        "unknown_field": "val"
      },
      "model_version": {
        "id": "@modelVersionID",
        "created_at": "2016-07-13T01:19:12.147644Z",
        "status": {
          "code": 21100,
          "description": "Model trained successfully"
        },
        "unknown_field": "val"
      },
      "display_name": "@modelDisplayName",
      "unknown_field": "val"
    },
    "input": {
      "id": "@inputID",
      "data": {
        "image": {
          "url": "@imageUrl",
          "unknown_field": "val"
        },
        "unknown_field": "val"
      },
      "unknown_field": "val"
    },
    "data": {
      "concepts": [{
        "id": "@conceptID1",
        "name": "@conceptName1",
        "value": 0.99,
        "app_id": "main",
        "unknown_field": "val"
      }, {
        "id": "@conceptID2",
        "name": "@conceptName2",
        "value": 0.98,
        "app_id": "main",
        "unknown_field": "val"
      }],
      "unknown_field": "val"
    },
    "unknown_field": "val"
  }]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict_by_url('@url')

  output = response['outputs'][0]
  assert output['id'] == '@outputID'
  assert output['input']['id'] == '@inputID'
  assert output['data']['concepts'][0]['id'] == '@conceptID1'

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@url"
        }
      }
    }
  ]
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_predict_with_input_id(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000, 
    "description": "Ok"
  }, 
  "outputs": [
    {
      "status": {
        "code": 10000, 
        "description": "Ok"
      }, 
      "created_at": "2019-02-27T16:36:12.896173378Z", 
      "input": {
        "data": {
          "image": {
            "url": "@imageURL"
          }
        }, 
        "id": "@inputID"
      }, 
      "model": {
        "display_name": "NSFW", 
        "name": "nsfw-v1.0", 
        "output_info": {
          "type_ext": "concept", 
          "message": "Show output_info with: GET /models/{model_id}/output_info", 
          "type": "concept"
        }, 
        "created_at": "2016-09-17T22:18:59.955626Z", 
        "app_id": "main", 
        "model_version": {
          "status": {
            "code": 21100, 
            "description": "Model trained successfully"
          }, 
          "created_at": "2018-01-23T19:25:09.618692Z", 
          "id": "aa47919c9a8d4d94bfa283121281bcc4", 
          "train_stats": {}
        }, 
        "id": "e9576d86d2004ed1a38ba0cf39ecb4b1"
      }, 
      "data": {
        "concepts": [
          {
            "app_id": "main", 
            "id": "@conceptID1", 
            "value": 0.87529075, 
            "name": "sfw"
          }, 
          {
            "app_id": "main", 
            "id": "@conceptID2", 
            "value": 0.124709226, 
            "name": "nsfw"
          }
        ]
      }, 
      "id": "59fa1efec39244c98b6827694db555e3"
    }
  ]
}
""")

  app = ClarifaiApp()
  model = app.models.get(model_id='@modelID')
  response = model.predict([Image(url='@imageURL', image_id='@inputID')])

  assert_request(mock_execute_request, 'POST', '/v2/models/@modelID/outputs', """
{
  "inputs": [
    {
      "data": {
        "image": {
          "url": "@imageURL"
        }
      }, 
      "id": "@inputID"
    }
  ]
}
  """)

  output = response['outputs'][0]
  assert output['input']['id'] == '@inputID'
  assert output['data']['concepts'][0]['id'] == '@conceptID1'
