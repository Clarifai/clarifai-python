import unittest.mock as mock

from clarifai.rest import ClarifaiApp, Model

from .mock_extensions import assert_request, mock_request


@mock.patch('clarifai.rest.http_client.HttpClient')
def test_model_evaluate(mock_http_client):  # type: (mock.Mock) -> None
  mock_execute_request = mock_request(mock_http_client, """
{
    "status": {
        "code": 10000,
        "description": "Ok"
    },
    "model_version": {
        "id": "@modelVersionID",
        "created_at": "2017-01-01T00:00:00.000000Z",
        "status": {
            "code": 21100,
            "description": "Model trained successfully"
        },
        "active_concept_count": 2,
        "metrics": {
            "status": {
                "code": 21303,
                "description": "Model is queued for evaluation."
            }
        },
        "total_input_count": 30
    }
}
""")

  app = ClarifaiApp()
  model = Model(app.api, model_id='@modelID')
  model.model_version = '@modelVersionID'
  response = model.evaluate()

  assert response['status']['code'] == 10000

  assert_request(mock_execute_request, 'POST',
                 '/v2/models/@modelID/versions/@modelVersionID/metrics', '{}')
