"""Utilities for handling pipeline input argument overrides."""

import json
from typing import Any, Dict, Optional


def parse_set_parameter(param_str: str) -> tuple[str, str]:
    """Parse a --set parameter string into key-value pair.

    Args:
        param_str: Parameter string in format "key=value"

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If parameter string is not in correct format
    """
    if '=' not in param_str:
        raise ValueError(
            f"Invalid --set parameter format: '{param_str}'. Expected format: key=value"
        )

    key, value = param_str.split('=', 1)
    key = key.strip()
    value = value.strip()

    if not key:
        raise ValueError(f"Empty key in --set parameter: '{param_str}'")

    return key, value


def load_overrides_from_file(file_path: str) -> Dict[str, str]:
    """Load parameter overrides from a JSON file.

    Args:
        file_path: Path to JSON file containing overrides

    Returns:
        Dictionary of parameter name to value mappings

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSON or doesn't contain a dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in overrides file '{file_path}': {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Overrides file '{file_path}' must contain a JSON object (dictionary), got {type(data).__name__}"
        )

    # Convert all values to strings (Argo convention)
    return {str(k): str(v) for k, v in data.items()}


def merge_override_parameters(
    inline_params: Optional[Dict[str, str]] = None, file_params: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Merge inline and file-based parameter overrides.

    Inline parameters take precedence over file parameters.

    Args:
        inline_params: Parameters from --set flags
        file_params: Parameters from --overrides-file

    Returns:
        Merged dictionary of parameters
    """
    result = {}

    if file_params:
        result.update(file_params)

    if inline_params:
        result.update(inline_params)

    return result


def build_argo_args_override(parameters: Dict[str, str]) -> Dict[str, Any]:
    """Build an ArgoArgsOverride structure from parameter dictionary.

    This creates a dictionary structure compatible with the proto message
    format that will be used when the proto is available.

    Args:
        parameters: Dictionary of parameter name to value mappings

    Returns:
        Dictionary structure compatible with OrchestrationArgsOverride proto
    """
    if not parameters:
        return {}

    # Build structure compatible with proto message format
    # This will be serialized to proto when clarifai-grpc is updated
    return {
        'argo_args_override': {
            'parameters': [{'name': name, 'value': value} for name, value in parameters.items()]
        }
    }


def validate_override_parameters(
    override_params: Dict[str, str], allowed_params: Optional[set] = None
) -> tuple[bool, Optional[str]]:
    """Validate that override parameters are allowed.

    Args:
        override_params: Parameters to validate
        allowed_params: Set of allowed parameter names. If None, validation is skipped.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not override_params:
        return True, None

    if allowed_params is None:
        # No validation rules provided, accept all parameters
        return True, None

    unknown_params = set(override_params.keys()) - allowed_params
    if unknown_params:
        return False, f"Unknown parameters: {', '.join(sorted(unknown_params))}"

    return True, None
