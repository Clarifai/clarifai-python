import unittest.mock as mock

from clarifai.rest import (ClarifaiApp, Geo, GeoLimit, GeoPoint, InputSearchTerm, OutputSearchTerm,
                           SearchQueryBuilder)

from .mock_extensions import assert_request, mock_request

TINY_IMAGE_BASE64 = b'R0lGODlhAQABAIABAP///wAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_inputs_by_concept_id(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": [
    {
      "score": 0.5,
      "input": {
        "id": "@inputID",
        "created_at": "2016-11-22T17:06:02Z",
        "data": {
          "image": {
            "url": "@inputURL"
          }
        },
        "status": {
          "code": 30000,
          "description": "Download complete"
        }
      }
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": []
}
"""
      ])

  app = ClarifaiApp()

  qb = SearchQueryBuilder()
  qb.add_term(OutputSearchTerm(concept_id='@conceptID'))
  inputs = app.inputs.search(qb)

  assert inputs[0].input_id == '@inputID'
  assert inputs[0].score == 0.5

  assert_request(mock_execute_request, 'POST', '/v2/searches', """
{
  "query": {
    "ands": [
      {
        "output": {
          "data": {
            "concepts": [
              {
                "id": "@conceptID",
                "value": 1.0
              }
            ]
          }
        }
      }
    ]
  },
  "pagination": {
    "per_page": 20, 
    "page": 1
  }
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_inputs_by_concept_name(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": [
    {
      "score": 0.5,
      "input": {
        "id": "@inputID",
        "created_at": "2016-11-22T17:06:02Z",
        "data": {
          "image": {
            "url": "@inputURL"
          }
        },
        "status": {
          "code": 30000,
          "description": "Download complete"
        }
      }
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": []
}
"""
      ])

  app = ClarifaiApp()

  qb = SearchQueryBuilder()
  qb.add_term(OutputSearchTerm(concept='@conceptName'))
  inputs = app.inputs.search(qb)

  assert inputs[0].input_id == '@inputID'
  assert inputs[0].score == 0.5

  assert_request(mock_execute_request, 'POST', '/v2/searches', """
{
  "query": {
    "ands": [
      {
        "output": {
          "data": {
            "concepts": [
              {
                "name": "@conceptName",
                "value": 1.0
              }
            ]
          }
        }
      }
    ]
  },
  "pagination": {
    "per_page": 20, 
    "page": 1
  }
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_inputs_by_geo_location(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": [
    {
      "score": 0.5,
      "input": {
        "id": "@inputID",
        "created_at": "2016-11-22T17:06:02Z",
        "data": {
          "image": {
            "url": "@inputURL"
          }
        },
        "status": {
          "code": 30000,
          "description": "Download complete"
        }
      }
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": []
}
"""
      ])

  app = ClarifaiApp()

  qb = SearchQueryBuilder()
  qb.add_term(
      InputSearchTerm(geo=Geo(
          geo_point=GeoPoint(longitude=1.5, latitude=-1),
          geo_limit=GeoLimit(limit_type='kilometer', limit_range=1))))
  inputs = app.inputs.search(qb)

  assert inputs[0].input_id == '@inputID'
  assert inputs[0].score == 0.5

  assert_request(mock_execute_request, 'POST', '/v2/searches', """
{
  "query": {
    "ands": [
      {
        "input": {
          "data": {
            "geo": {
              "geo_point": {
                "longitude": 1.5,
                "latitude": -1
              },
              "geo_limit": {
                "type": "withinKilometers",
                "value": 1
              }
            }
          }
        }
      }
    ]
  },
  "pagination": {
    "per_page": 20, 
    "page": 1
  }
}
  """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_inputs_by_url(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": [
    {
      "score": 0.5,
      "input": {
        "id": "@inputID",
        "created_at": "2016-11-22T17:06:02Z",
        "data": {
          "image": {
            "url": "@inputURL"
          }
        },
        "status": {
          "code": 30000,
          "description": "Download complete"
        }
      }
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": []
}
"""
      ])

  app = ClarifaiApp()

  qb = SearchQueryBuilder()
  qb.add_term(InputSearchTerm(url="@inputURL"))

  inputs = app.inputs.search(qb)

  assert inputs[0].input_id == '@inputID'
  assert inputs[0].score == 0.5

  assert_request(mock_execute_request, 'POST', '/v2/searches', """
  {
    "query": {
      "ands": [
        {
          "input": {
            "data": {
              "image": {
                 "url": "@inputURL"
               }
            }
          }
        }  
      ]
    },
    "pagination": {
      "per_page": 20, 
      "page": 1
    }
  }
    """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_inputs_by_metadata(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "hits": [
    {
      "score": 0.5,
      "input": {
        "id": "@inputID",
        "created_at": "2016-11-22T17:06:02Z",
        "data": {
          "image": {
            "url": "@inputURL"
          }
        },
        "status": {
          "code": 30000,
          "description": "Download complete"
        }
      }
    }
  ]
}
""", """
{
"status": {
  "code": 10000,
  "description": "Ok"
},
"hits": []
}
"""
      ])

  app = ClarifaiApp()

  qb = SearchQueryBuilder()
  qb.add_term(InputSearchTerm(metadata={"@key1": "@value1", "@key2": "@value2"}))

  inputs = app.inputs.search(qb)

  assert inputs[0].input_id == '@inputID'
  assert inputs[0].score == 0.5

  assert_request(mock_execute_request, 'POST', '/v2/searches', """
  {
    "query": {
      "ands": [
        {
          "input": {
            "data": {
              "metadata": {
                "@key1": "@value1",
                "@key2": "@value2"
              }
            }
          }
        }  
      ]
    },
    "pagination": {
      "per_page": 20, 
      "page": 1
    }
  }
    """)
