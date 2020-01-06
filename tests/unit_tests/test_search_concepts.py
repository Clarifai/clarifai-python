# -*- coding: utf-8 -*-

import mock

from clarifai.rest import ClarifaiApp

from .mock_extensions import assert_requests, mock_request


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_concepts(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [
    {
      "id": "@conceptID1",
      "name": "concealer",
      "value": 1,
      "created_at": "2016-03-17T11:43:01.223962Z",
      "language": "en",
      "app_id": "main",
      "definition": "concealer"
    },
    {
      "id": "@conceptID2",
      "name": "concentrate",
      "value": 1,
      "created_at": "2016-03-17T11:43:01.223962Z",
      "language": "en",
      "app_id": "main",
      "definition": "direct one's attention on something"
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": []
}
"""
      ])

  app = ClarifaiApp()
  concepts = list(app.concepts.search('conc*'))

  assert concepts[0].concept_id == '@conceptID1'
  assert concepts[1].concept_id == '@conceptID2'

  assert_requests(
      mock_execute_request,
      requests=[('POST', '/v2/concepts/searches', """
{
  "concept_query": {
    "name": "conc*"
  },
  "pagination": {
    "per_page": 20,
    "page": 1
  }
}
  """), ('POST', '/v2/concepts/searches', """
{
  "concept_query": {
    "name": "conc*"
  },
  "pagination": {
    "per_page": 20,
    "page": 2
  }
}
  """)])


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_search_concepts_with_language(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [
    {
      "id": "@conceptID1",
      "name": "狗",
      "value": 1,
      "created_at": "2016-03-17T11:43:01.223962Z",
      "language": "zh",
      "app_id": "main"
    },
    {
      "id": "@conceptID2",
      "name": "狗仔队",
      "value": 1,
      "created_at": "2016-03-17T11:43:01.223962Z",
      "language": "zh",
      "app_id": "main"
    }
  ]
}
""", """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": []
}
"""
      ])

  app = ClarifaiApp()
  concepts = list(app.concepts.search('狗*', lang='zh'))

  assert concepts[0].concept_id == '@conceptID1'
  assert concepts[0].concept_name == u'狗'

  assert concepts[1].concept_id == '@conceptID2'
  assert concepts[1].concept_name == u'狗仔队'

  assert_requests(
      mock_execute_request,
      requests=[('POST', '/v2/concepts/searches', """
{
  "concept_query": {
    "name": "狗*",
    "language": "zh"
  },
  "pagination": {
    "per_page": 20,
    "page": 1
  }
}
  """), ('POST', '/v2/concepts/searches', """
{
  "concept_query": {
    "name": "狗*",
    "language": "zh"
  },
  "pagination": {
    "per_page": 20,
    "page": 2
  }
}
  """)])
