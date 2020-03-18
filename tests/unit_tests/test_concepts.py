import unittest.mock as mock

import pytest

from clarifai.errors import ApiError, UserError
from clarifai.rest import ClarifaiApp

from .mock_extensions import (assert_request, assert_requests, mock_request,
                              mock_request_with_failed_response)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_bulk_create_concepts(mock_http_client):
  mock_execute_request = mock_request(mock_http_client, """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [{
    "id": "@id1",
    "name": "@name1",
    "created_at": "2018-03-06T20:24:55.407961035Z",
    "language": "en",
    "app_id": "@appID"
  }, {
    "id": "@id2",
    "name": "@name2",
    "created_at": "2018-03-06T20:24:55.407961035Z",
    "language": "en",
    "app_id": "@appID"
  }]
}
""")

  app = ClarifaiApp()

  # User Error should be generated when amount of ids and concepts are unequal
  with pytest.raises(UserError):
    concepts = app.concepts.bulk_create(['@id1'], ['@new-name1', '@new-name2'])

  concepts = app.concepts.bulk_create(['@id1', '@id2'], ['@name1', '@name2'])
  assert len(concepts) == 2

  assert concepts[0].concept_id == '@id1'
  assert concepts[0].concept_name == '@name1'

  assert concepts[1].concept_id == '@id2'
  assert concepts[1].concept_name == '@name2'

  assert_request(mock_execute_request, 'POST', '/v2/concepts', """
{
  "concepts": [
    {
      "id": "@id1",
      "name": "@name1",
      "value": 1.0
    },
    {
      "id": "@id2",
      "name": "@name2",
      "value": 1.0
    }
  ]
}
        """)


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_concept(mock_http_client):
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concept": {
    "id": "@conceptID1",
    "name": "@conceptName1",
    "created_at": "2018-03-06T20:24:55.407961035Z",
    "language": "en",
    "app_id": "@appID"
  }
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

  concept = app.concepts.get('@conceptID1')
  assert concept.concept_id == '@conceptID1'
  assert concept.concept_name == '@conceptName1'

  assert_request(mock_execute_request, 'GET', '/v2/concepts/@conceptID1')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_non_existing_concept(mock_http_client):
  mock_execute_request = mock_request_with_failed_response(
      mock_http_client, json_responses=[ApiError("/v2/concepts/@nonexistentID", {}, "GET")])

  app = ClarifaiApp()
  with pytest.raises(ApiError):
    app.concepts.get("@nonexistentID")

  assert_request(mock_execute_request, 'GET', '/v2/concepts/@nonexistentID')


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_get_all_concepts(mock_http_client):
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
  "status": {
    "code": 10000,
    "description": "Ok"
  },
  "concepts": [{
    "id": "@conceptID1",
    "name": "@conceptName1",
    "created_at": "2017-10-15T16:28:28.901994Z",
    "language": "en",
    "app_id": "@appID"
  }, {
    "id": "@conceptID2",
    "name": "@conceptName2",
    "created_at": "2017-10-15T16:26:46.667104Z",
    "language": "en",
    "app_id": "@appID"
  }]
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
  concepts = list(app.concepts.get_all())

  assert concepts[0].concept_id == '@conceptID1'
  assert concepts[0].concept_name == '@conceptName1'

  assert concepts[1].concept_id == '@conceptID2'
  assert concepts[1].concept_name == '@conceptName2'

  assert len(concepts) == 2

  assert_requests(mock_execute_request, [
      ('GET', '/v2/concepts', '{"per_page": 20, "page": 1}'),
      ('GET', '/v2/concepts', '{"per_page": 20, "page": 2}'),
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
  "concepts": []
}
"""
      ])
  concepts = list(app.concepts.get_all())
  assert not concepts


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_bulk_update_concepts(mock_http_client):
  mock_execute_request = mock_request(
      mock_http_client,
      json_responses=[
          """
{
    "status": {
        "code": 10000,
        "description": "Ok"
    },
    "concepts": [{
        "id": "@id1",
        "name": "@new-name1",
        "created_at": "2018-03-06T20:24:55.407961035Z",
        "language": "en",
        "app_id": "@appID"
    }, {
        "id": "@id2",
        "name": "@new-name2",
        "created_at": "2018-03-06T20:24:55.407961035Z",
        "language": "en",
        "app_id": "@appID"
    }]
}
"""
      ])

  app = ClarifaiApp()
  concepts = app.concepts.bulk_update(['@id1', '@id2'], ['@new-name1', '@new-name2'])

  assert len(concepts) == 2

  assert concepts[0].concept_id == '@id1'
  assert concepts[0].concept_name == '@new-name1'

  assert concepts[1].concept_id == '@id2'
  assert concepts[1].concept_name == '@new-name2'

  # User Error should be thrown if unsupported action is given
  with pytest.raises(UserError):
    app.concepts.bulk_update(['@id'], ['@name'], 'splice')

  assert_request(mock_execute_request, 'PATCH', '/v2/concepts', """
{
  "action": "overwrite",
  "concepts": [
    {
      "id": "@id1",
      "name": "@new-name1",
      "value": 1.0
    },
    {
      "id": "@id2",
      "name": "@new-name2",
      "value": 1.0
    }
  ]
}
        """)
