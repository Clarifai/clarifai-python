from unittest import mock

import grpc
import pytest as pytest
from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper, clear_cache
from clarifai.client.auth.stub import AuthorizedStub, RetryStub


class MockRpcError(grpc.RpcError):
  pass


@pytest.fixture(autouse=True)
def clear_caches():
  clear_cache()


def test_auth_unary_unary():
  auth = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  stub = AuthorizedStub(auth)
  with mock.patch.object(stub.stub, 'ListInputs', spec=stub.stub.ListInputs) as mock_f:
    req = service_pb2.ListInputsRequest()
    req.user_app_id.app_id = 'test_auth_unary_unary'
    stub.ListInputs(req)
    mock_f.assert_called_with(req, metadata=auth.metadata)


def test_auth_unary_unary_future():
  auth = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  stub = AuthorizedStub(auth)
  with mock.patch.object(stub.stub, 'ListInputs', spec=stub.stub.ListInputs) as mock_f:
    req = service_pb2.ListInputsRequest()
    req.user_app_id.app_id = 'test_auth_unary_unary_future'
    stub.ListInputs.future(req)
    mock_f.future.assert_called_with(req, metadata=auth.metadata)


def test_auth_unary_stream():
  auth = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  stub = AuthorizedStub(auth)
  with mock.patch.object(stub.stub, 'StreamInputs', spec=stub.stub.StreamInputs) as mock_f:
    req = service_pb2.StreamInputsRequest()
    req.user_app_id.app_id = 'test_auth_unary_stream'
    stub.StreamInputs(req)
    mock_f.assert_called_with(req, metadata=auth.metadata)


def test_retry_unary_unary():
  max_attempts = 5
  auth = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  stub = RetryStub(AuthorizedStub(auth), max_attempts=max_attempts, backoff_time=0.0001)
  retry_response = service_pb2.MultiInputResponse()
  retry_response.status.code = status_code_pb2.CONN_THROTTLED
  success_response = service_pb2.MultiInputResponse()
  success_response.status.code = status_code_pb2.SUCCESS
  for nfailures in range(0, max_attempts + 1):
    mock_resps = [retry_response] * nfailures + [success_response]
    with mock.patch.object(
        stub.stub, 'ListInputs', spec=stub.stub.stub.ListInputs, side_effect=mock_resps) as mock_f:
      req = service_pb2.ListInputsRequest()
      req.user_app_id.app_id = 'test_retry_unary_unary'
      res = stub.ListInputs(req)
      assert mock_f.call_count == min(max_attempts, len(mock_resps))
      if nfailures < max_attempts:
        assert res is success_response
      else:
        assert res is retry_response


def test_retry_grpcconn_unary_unary():
  max_attempts = 5
  auth = ClarifaiAuthHelper("clarifai", "main", "fake_pat")
  stub = RetryStub(AuthorizedStub(auth), max_attempts=max_attempts, backoff_time=0.0001)
  retry_response = service_pb2.MultiInputResponse()
  retry_response.status.code = status_code_pb2.CONN_THROTTLED
  success_response = service_pb2.MultiInputResponse()
  success_response.status.code = status_code_pb2.SUCCESS
  error = MockRpcError()
  error.code = lambda: grpc.StatusCode.UNAVAILABLE
  for nfailures in range(0, max_attempts + 1):
    mock_resps = [error] * nfailures + [success_response]
    with mock.patch.object(
        stub.stub, 'ListInputs', spec=stub.stub.stub.ListInputs, side_effect=mock_resps) as mock_f:
      req = service_pb2.ListInputsRequest()
      req.user_app_id.app_id = 'test_retry_unary_unary'
      try:
        res = stub.ListInputs(req)
      except Exception as e:
        res = e
      assert mock_f.call_count == min(max_attempts, len(mock_resps))
      if nfailures < max_attempts:
        assert res is success_response
      else:
        assert res is error
