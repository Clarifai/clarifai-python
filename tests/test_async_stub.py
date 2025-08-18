from unittest.mock import AsyncMock, patch

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper


@pytest.fixture
def mock_auth_helper():
    """Create mock auth helper"""
    return ClarifaiAuthHelper(user_id='openai', app_id='chat-completion', pat='test_pat')


@pytest.mark.asyncio
async def test_async_stub_creation(mock_auth_helper):
    """Test async stub is created properly"""
    from clarifai.client.auth import create_stub

    stub = create_stub(mock_auth_helper, is_async=True)

    assert stub is not None
    assert stub.is_async is True
    assert stub.metadata == mock_auth_helper.metadata


@pytest.mark.asyncio
async def test_async_get_model_version(mock_auth_helper):
    """Test basic async get request works"""
    from clarifai.client.auth import create_stub

    mock_response = service_pb2.SingleModelResponse()
    mock_response.status.code = status_code_pb2.SUCCESS
    stub = create_stub(mock_auth_helper, is_async=True)

    with patch.object(stub.stub, 'GetModelVersion', AsyncMock(return_value=mock_response)):
        request = service_pb2.GetModelVersionRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id="openai", app_id="chat-completion"),
            model_id="o4-mini",
        )

        response = await stub.GetModelVersion(request)
        assert response.status.code == status_code_pb2.SUCCESS
