"""Test for _update_old_fields functionality in OpenAIModelClass."""

from clarifai.runners.models.dummy_openai_model import DummyOpenAIModel


class TestUpdateOldFields:
    """Tests for _update_old_fields method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = DummyOpenAIModel()

    def test_max_tokens_only(self):
        """Test that max_tokens is promoted to max_completion_tokens and removed."""
        request_data = {"max_tokens": 100}
        result = self.model._update_old_fields(request_data)

        assert result.get("max_completion_tokens") == 100
        assert "max_tokens" not in result

    def test_max_completion_tokens_only(self):
        """Test that max_completion_tokens is kept as-is, no max_tokens added."""
        request_data = {"max_completion_tokens": 200}
        result = self.model._update_old_fields(request_data)

        assert result.get("max_completion_tokens") == 200
        assert "max_tokens" not in result

    def test_both_fields_present(self):
        """Test that max_completion_tokens is preferred when both fields are present."""
        request_data = {"max_tokens": 100, "max_completion_tokens": 200}
        result = self.model._update_old_fields(request_data)

        assert result.get("max_completion_tokens") == 200
        assert "max_tokens" not in result

    def test_neither_field_present(self):
        """Test that no changes are made when neither field is present."""
        request_data = {"temperature": 0.7}
        result = self.model._update_old_fields(request_data)

        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result
        assert result.get("temperature") == 0.7

    def test_zero_values(self):
        """Test that zero values are handled correctly."""
        request_data = {"max_tokens": 0}
        result = self.model._update_old_fields(request_data)

        assert result.get("max_completion_tokens") == 0
        assert "max_tokens" not in result

    def test_other_fields_unchanged(self):
        """Test that other fields in request_data are not affected."""
        request_data = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "model": "gpt-4",
        }
        result = self.model._update_old_fields(request_data)

        assert result.get("temperature") == 0.7
        assert result.get("top_p") == 0.9
        assert result.get("model") == "gpt-4"
        assert result.get("max_completion_tokens") == 100
        assert "max_tokens" not in result
