import os

import pytest

from clarifai.client.model import Model
from clarifai.client.user import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]


def test_list_models():
    user = User()

    # list all public models
    public_models = user.list_models(user_id="all", show=False)
    assert len(public_models) > 0

    # Commenented out this for now, as it takes long time to run
    # list models of current user
    # models = user.list_models(show=False)
    # assert len(models) > 0

    # list models of an app
    models = user.list_models(user_id="clarifai", app_id="main", show=False)
    assert len(models) > 0

    # list public models of an account
    models = user.list_models(user_id="openai", show=True)
    assert len(models) > 0

    # list public models with valid params
    fmodels = user.list_models(featured_only=True, show=False)
    assert len(fmodels) < len(public_models)

    # list public models with invalid params
    with pytest.raises(Exception):
        user.list_models(user_id="openai", a=True, c=5)

    # return Clarifai Model instance
    models = user.list_models(
        user_id="openai", sort_by_name=True, show=False, return_clarifai_model=True
    )
    assert isinstance(models[0], Model)
