from unittest.mock import AsyncMock

import pytest

from clarifai.client.model import Model


@pytest.mark.asyncio
async def test_model_async_predict(monkeypatch):
  #Dummy model obj
  model = Model(model_id='dummy', user_id='user', app_id='app')
  monkeypatch.setattr(model, 'async_predict', AsyncMock(return_value="mocked response"))
  result = await model.async_predict("any_input_data")
  assert result == "mocked response"


@pytest.mark.asyncio
async def test_model_async_generate(monkeypatch):
  model = Model(model_id='dummy', user_id='user', app_id='app')

  async def mock_generator():
    yield "mocked chunk 1"
    yield "mocked chunk 2"

  monkeypatch.setattr(model, 'async_generate', AsyncMock(return_value=mock_generator()))
  generator = await model.async_generate("any_input_data")
  responses = []
  async for response in generator:
    responses.append(response)
  assert responses == ["mocked chunk 1", "mocked chunk 2"]
