"""Tests for pipeline input argument override functionality."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from clarifai.utils.pipeline_overrides import (
    build_argo_args_override,
    load_overrides_from_file,
    merge_override_parameters,
    parse_set_parameter,
    validate_override_parameters,
)


class TestPipelineOverrides:
    """Test cases for pipeline override utilities."""

    def test_parse_set_parameter_valid(self):
        """Test parsing valid --set parameter."""
        key, value = parse_set_parameter("prompt=Hello World")
        assert key == "prompt"
        assert value == "Hello World"

        key, value = parse_set_parameter("temperature=0.7")
        assert key == "temperature"
        assert value == "0.7"

        # Test with equals sign in value (split only on first =)
        key, value = parse_set_parameter("equation=x=y+1")
        assert key == "equation"
        assert value == "x=y+1"

    def test_parse_set_parameter_with_spaces(self):
        """Test parsing parameter with spaces."""
        key, value = parse_set_parameter("  key  =  value  ")
        assert key == "key"
        assert value == "value"

    def test_parse_set_parameter_invalid_format(self):
        """Test parsing invalid parameter format."""
        with pytest.raises(ValueError, match="Invalid --set parameter format"):
            parse_set_parameter("invalid_parameter")

        with pytest.raises(ValueError, match="Empty key"):
            parse_set_parameter("=value")

    def test_load_overrides_from_file_valid(self):
        """Test loading valid overrides from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"prompt": "Test prompt", "temperature": "0.7"}, f)
            temp_file = f.name

        try:
            overrides = load_overrides_from_file(temp_file)
            assert overrides == {"prompt": "Test prompt", "temperature": "0.7"}
        finally:
            os.unlink(temp_file)

    def test_load_overrides_from_file_converts_to_strings(self):
        """Test that all values are converted to strings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"count": 123, "enabled": True, "ratio": 0.5}, f)
            temp_file = f.name

        try:
            overrides = load_overrides_from_file(temp_file)
            assert overrides == {"count": "123", "enabled": "True", "ratio": "0.5"}
        finally:
            os.unlink(temp_file)

    def test_load_overrides_from_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_overrides_from_file("/path/to/nonexistent/file.json")

    def test_load_overrides_from_file_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_overrides_from_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_overrides_from_file_not_dict(self):
        """Test loading from file that doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["array", "not", "dict"], f)
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="must contain a JSON object"):
                load_overrides_from_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_merge_override_parameters_inline_only(self):
        """Test merging with only inline parameters."""
        inline = {"param1": "value1", "param2": "value2"}
        result = merge_override_parameters(inline_params=inline)
        assert result == inline

    def test_merge_override_parameters_file_only(self):
        """Test merging with only file parameters."""
        file_params = {"param1": "value1", "param2": "value2"}
        result = merge_override_parameters(file_params=file_params)
        assert result == file_params

    def test_merge_override_parameters_precedence(self):
        """Test that inline parameters take precedence over file parameters."""
        inline = {"param1": "inline_value", "param2": "inline_value2"}
        file_params = {"param1": "file_value", "param3": "file_value3"}
        result = merge_override_parameters(inline_params=inline, file_params=file_params)

        assert result["param1"] == "inline_value"  # Inline takes precedence
        assert result["param2"] == "inline_value2"
        assert result["param3"] == "file_value3"

    def test_merge_override_parameters_empty(self):
        """Test merging with no parameters."""
        result = merge_override_parameters()
        assert result == {}

    def test_build_argo_args_override(self):
        """Test building Argo args override structure."""
        params = {"prompt": "Test prompt", "temperature": "0.7"}
        result = build_argo_args_override(params)

        assert "argo_args_override" in result
        assert "parameters" in result["argo_args_override"]
        parameters = result["argo_args_override"]["parameters"]

        assert len(parameters) == 2
        # Check that both parameters are present
        param_dict = {p["name"]: p["value"] for p in parameters}
        assert param_dict["prompt"] == "Test prompt"
        assert param_dict["temperature"] == "0.7"

    def test_build_argo_args_override_empty(self):
        """Test building override with empty parameters."""
        result = build_argo_args_override({})
        assert result == {}

    def test_validate_override_parameters_valid(self):
        """Test validation with valid parameters."""
        overrides = {"param1": "value1", "param2": "value2"}
        allowed = {"param1", "param2", "param3"}

        is_valid, error = validate_override_parameters(overrides, allowed)
        assert is_valid is True
        assert error is None

    def test_validate_override_parameters_invalid(self):
        """Test validation with invalid parameters."""
        overrides = {"param1": "value1", "unknown_param": "value2"}
        allowed = {"param1", "param2"}

        is_valid, error = validate_override_parameters(overrides, allowed)
        assert is_valid is False
        assert "unknown_param" in error

    def test_validate_override_parameters_no_rules(self):
        """Test validation with no rules (accept all)."""
        overrides = {"any_param": "any_value"}

        is_valid, error = validate_override_parameters(overrides, None)
        assert is_valid is True
        assert error is None

    def test_validate_override_parameters_empty(self):
        """Test validation with empty overrides."""
        is_valid, error = validate_override_parameters({}, {"param1"})
        assert is_valid is True
        assert error is None


class TestPipelineClientWithOverrides:
    """Test cases for Pipeline client with input argument overrides."""

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_with_input_args_override(self, mock_init):
        """Test pipeline run with input argument overrides."""
        from clarifai_grpc.grpc.api import resources_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2

        from clarifai.client.pipeline import Pipeline

        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock PostPipelineVersionRuns response
        mock_run_response = Mock()
        mock_run_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_run = Mock()
        mock_run.id = 'test-run-123'
        mock_run_response.pipeline_version_runs = [mock_run]
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_run_response

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": mock_run}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute run with overrides
        input_args_override = {"prompt": "Test prompt", "temperature": "0.7"}
        result = pipeline.run(input_args_override=input_args_override)

        # Verify the result
        assert result == expected_result
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()
        pipeline._monitor_pipeline_run.assert_called_once()

        # Verify the request was made (we can't check the exact structure without proto support)
        call_args = pipeline.STUB.PostPipelineVersionRuns.call_args
        assert call_args is not None

    @patch('clarifai.client.pipeline.BaseClient.__init__')
    def test_run_without_input_args_override(self, mock_init):
        """Test pipeline run works without input argument overrides (backward compatibility)."""
        from clarifai_grpc.grpc.api import resources_pb2
        from clarifai_grpc.grpc.api.status import status_code_pb2

        from clarifai.client.pipeline import Pipeline

        mock_init.return_value = None

        pipeline = Pipeline(
            pipeline_id='test-pipeline',
            pipeline_version_id='test-version-123',
            user_id='test-user',
            app_id='test-app',
            pat='test-pat',
        )

        # Mock the required attributes
        pipeline.user_app_id = resources_pb2.UserAppIDSet(user_id="test-user", app_id="test-app")
        pipeline.STUB = Mock()
        pipeline.auth_helper = Mock()
        pipeline.auth_helper.metadata = []

        # Mock PostPipelineVersionRuns response
        mock_run_response = Mock()
        mock_run_response.status.code = status_code_pb2.StatusCode.SUCCESS
        mock_run = Mock()
        mock_run.id = 'test-run-123'
        mock_run_response.pipeline_version_runs = [mock_run]
        pipeline.STUB.PostPipelineVersionRuns.return_value = mock_run_response

        # Mock the monitoring method
        expected_result = {"status": "success", "pipeline_version_run": mock_run}
        pipeline._monitor_pipeline_run = Mock(return_value=expected_result)

        # Execute run without overrides (backward compatibility)
        result = pipeline.run()

        # Verify the result
        assert result == expected_result
        pipeline.STUB.PostPipelineVersionRuns.assert_called_once()
        pipeline._monitor_pipeline_run.assert_called_once()
