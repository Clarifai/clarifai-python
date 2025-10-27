"""Tests for Model initialization validation."""
import os

import pytest

from clarifai.client.model import Model
from clarifai.errors import UserError

CLARIFAI_PAT = os.environ.get("CLARIFAI_PAT", "test_pat")
CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")

# Valid model for comparison
MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
GENERAL_MODEL_ID = "aaa03c23b3724a16a56b629203edc62c"


class TestModelInitValidation:
    """Tests for Model constructor validation."""

    @pytest.mark.requires_secrets
    def test_valid_model_url(self):
        """Test that a valid model URL initializes successfully."""
        url = f"https://clarifai.com/{MAIN_APP_USER_ID}/{MAIN_APP_ID}/models/{GENERAL_MODEL_ID}"
        model = Model(url=url, pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE)
        assert model.id == GENERAL_MODEL_ID
        assert model.user_id == MAIN_APP_USER_ID
        assert model.app_id == MAIN_APP_ID

    @pytest.mark.requires_secrets
    def test_valid_model_id(self):
        """Test that a valid model_id initializes successfully."""
        model = Model(
            user_id=MAIN_APP_USER_ID,
            app_id=MAIN_APP_ID,
            model_id=GENERAL_MODEL_ID,
            pat=CLARIFAI_PAT,
            base_url=CLARIFAI_API_BASE,
        )
        assert model.id == GENERAL_MODEL_ID
        assert model.user_id == MAIN_APP_USER_ID
        assert model.app_id == MAIN_APP_ID

    @pytest.mark.requires_secrets
    def test_nonexistent_model_url(self):
        """Test that a non-existent model URL raises UserError with clear message."""
        url = f"https://clarifai.com/{MAIN_APP_USER_ID}/{MAIN_APP_ID}/models/non-existent-model-xyz-123"
        with pytest.raises(UserError) as exc_info:
            Model(url=url, pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE)

        error_msg = str(exc_info.value)
        # Check that the error message contains helpful information
        assert "does not exist" in error_msg or "cannot be accessed" in error_msg
        assert url in error_msg

    @pytest.mark.requires_secrets
    def test_nonexistent_model_id(self):
        """Test that a non-existent model_id raises UserError with clear message."""
        with pytest.raises(UserError) as exc_info:
            Model(
                user_id=MAIN_APP_USER_ID,
                app_id=MAIN_APP_ID,
                model_id="non-existent-model-xyz-123",
                pat=CLARIFAI_PAT,
                base_url=CLARIFAI_API_BASE,
            )

        error_msg = str(exc_info.value)
        # Check that the error message contains helpful information
        assert "does not exist" in error_msg or "cannot be accessed" in error_msg
        assert "non-existent-model-xyz-123" in error_msg

    def test_missing_url_and_model_id(self):
        """Test that missing both url and model_id raises UserError."""
        with pytest.raises(UserError) as exc_info:
            Model(pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE)

        error_msg = str(exc_info.value)
        assert "must specify one of url or model_id" in error_msg

    def test_both_url_and_model_id(self):
        """Test that providing both url and model_id raises UserError."""
        url = f"https://clarifai.com/{MAIN_APP_USER_ID}/{MAIN_APP_ID}/models/{GENERAL_MODEL_ID}"
        with pytest.raises(UserError) as exc_info:
            Model(
                url=url, model_id=GENERAL_MODEL_ID, pat=CLARIFAI_PAT, base_url=CLARIFAI_API_BASE
            )

        error_msg = str(exc_info.value)
        assert "only specify one of url or model_id" in error_msg
