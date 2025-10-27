# Pipeline Step Secrets Feature - Implementation Summary

## Overview
This implementation adds support for **Pipeline Step Secrets** to the Clarifai Python SDK and CLI, enabling different pipeline steps to access distinct sets of secrets with step-level isolation.

## Changes Made

### 1. Pipeline Client SDK (`clarifai/client/pipeline.py`)

Added four new methods to the `Pipeline` class for managing step secrets:

#### `add_step_secret(step_ref, secret_name, secret_ref)`
- Adds a secret to a specific pipeline step
- Stores secrets in internal `_step_secrets` dictionary
- Returns updated step secrets configuration
- Thread-safe for single-process usage

#### `remove_step_secret(step_ref, secret_name)`
- Removes a secret from a specific pipeline step
- Automatically cleans up empty step entries
- Raises KeyError if step or secret doesn't exist
- Returns updated step secrets configuration

#### `list_step_secrets(step_ref)`
- Lists all secrets for a specific step
- Returns empty dict if step has no secrets
- Read-only operation

#### `get_step_secrets()`
- Returns all step secrets for the pipeline
- Returns complete mapping of step_ref -> secrets
- Read-only operation

**Implementation Details:**
- Uses lazy initialization of `_step_secrets` attribute
- Comprehensive docstrings with examples
- Type hints for all parameters and return values
- Logging of secret operations

### 2. Pipeline Builder (`clarifai/runners/pipelines/pipeline_builder.py`)

#### New Method: `_add_step_secrets_to_pipeline_version(pipeline_version, step_secrets_config)`
- Populates `step_version_secrets` field in PipelineVersionConfig proto
- Creates StepSecretConfig proto for each step
- Handles AttributeError gracefully if proto doesn't support the field
- Logs information about added secrets

#### Modified: `create_pipeline()`
- Reads `step_secrets` from pipeline config
- Calls `_add_step_secrets_to_pipeline_version()` if secrets are present
- Maintains backward compatibility (no secrets = no changes)

#### Modified: `upload_pipeline(path, no_lockfile, step_secrets)`
- New parameter: `step_secrets` (optional)
- Merges CLI-provided secrets with config file secrets
- CLI secrets take precedence over config file secrets
- Logs merge operation

**Proto Field Structure:**
```python
pipeline_version.config.step_version_secrets[step_ref].secrets[secret_name] = secret_ref
```

### 3. CLI (`clarifai/cli/pipeline.py`)

#### Modified: `upload` command
- Added `--step-secret` flag (multiple=True)
- Format: `step_ref:SECRET_NAME=secret_ref_path`
- Supports multiple `--step-secret` flags
- Validation and error handling for malformed input

**Parsing Logic:**
1. Split on `:` to get step_ref and secret_part
2. Split secret_part on `=` to get secret_name and secret_ref
3. Strip whitespace from all components
4. Validate non-empty values
5. Build step_secrets dictionary
6. Pass to upload_pipeline()

**Example Usage:**
```bash
clarifai pipeline upload \
  --step-secret step1:API_KEY=users/user123/secrets/my-key \
  --step-secret step1:DB_PASSWORD=users/user123/secrets/db-pass \
  --step-secret step2:EMAIL_TOKEN=users/user123/secrets/email-token
```

### 4. Tests

#### `tests/test_pipeline_client.py` (extended)
Added 5 new test methods:
- `test_add_step_secret`: Tests adding secrets to steps
- `test_remove_step_secret`: Tests removing secrets and error cases
- `test_list_step_secrets`: Tests listing secrets for a step
- `test_get_step_secrets`: Tests retrieving all step secrets

#### `tests/test_pipeline_step_secrets.py` (new file)
Created with 4 test classes/methods:
- `test_add_step_secrets_to_pipeline_version`: Tests proto population
- `test_add_empty_step_secrets`: Tests empty secrets don't cause errors
- `test_create_pipeline_with_step_secrets`: Tests end-to-end pipeline creation
- `test_create_pipeline_without_step_secrets`: Tests backward compatibility

#### `tests/cli/test_pipeline.py` (extended)
Added new test class `TestPipelineStepSecretsCliFlag` with 5 methods:
- `test_upload_with_single_step_secret`: Single secret parsing
- `test_upload_with_multiple_step_secrets`: Multiple secrets parsing
- `test_upload_with_invalid_step_secret_format`: Error handling
- `test_upload_without_step_secrets`: Backward compatibility

**Test Coverage:**
- Unit tests for all new methods
- Integration tests for CLI flag parsing
- Error handling and edge cases
- Backward compatibility scenarios
- Proto version compatibility checks

## Backward Compatibility

✅ **Fully Backward Compatible:**
- All new fields and parameters are optional
- Existing pipelines without secrets work unchanged
- No modifications to existing API calls required
- Graceful degradation if proto doesn't support new fields
- No breaking changes to method signatures

## Security Considerations

1. **Reference-Only Storage**: Only secret references (paths) are stored, never actual values
2. **Runtime Injection**: Secret values are injected by backend at runtime
3. **Step Isolation**: Each step only receives its explicitly configured secrets
4. **No Value Leakage**: Secret values never appear in configs, logs, or responses

## Dependencies

**Proto Requirements:**
- Requires `clarifai-grpc >= 11.9.8` for full functionality
- `PipelineVersionConfig.step_version_secrets` field
- `StepSecretConfig` message type with `secrets` map field

**Current Behavior if Proto Doesn't Support:**
- Logs warning message
- Continues without error
- Allows gradual rollout of proto updates

## API Field Structure

Based on the issue requirements, the expected proto structure is:

```protobuf
message PipelineVersionConfig {
  map<string, StepSecretConfig> step_version_secrets = N;
}

message StepSecretConfig {
  map<string, string> secrets = 1;
}
```

**Populated Example:**
```json
{
  "pipeline_version": {
    "config": {
      "step_version_secrets": {
        "step1": {
          "secrets": {
            "API_KEY": "users/user123/secrets/my-api-key",
            "DB_PASSWORD": "users/user123/secrets/db-secret"
          }
        },
        "step2": {
          "secrets": {
            "EMAIL_TOKEN": "users/user123/secrets/email-token"
          }
        }
      }
    }
  }
}
```

## Configuration File Format

**config.yaml with step secrets:**
```yaml
pipeline:
  id: my-pipeline
  user_id: user123
  app_id: app456
  orchestration_spec:
    argo_orchestration_spec: |
      # Argo workflow definition
  step_directories:
    - step1
    - step2
  step_secrets:
    step1:
      API_KEY: users/user123/secrets/my-api-key
      DB_PASSWORD: users/user123/secrets/db-secret
    step2:
      EMAIL_TOKEN: users/user123/secrets/email-token
```

## Usage Examples

See `PIPELINE_STEP_SECRETS_EXAMPLES.md` for comprehensive usage examples including:
- Python SDK usage
- CLI usage
- Config file format
- Secret reference format
- Best practices
- Security recommendations

## Testing Recommendations

1. **Unit Tests** ✅ - Completed
2. **Integration Tests** - Requires API access:
   - Create pipeline with secrets via SDK
   - Create pipeline with secrets via CLI
   - Verify secrets in GetPipelineVersion response
   - Update pipeline secrets via PATCH
   - Run pipeline and verify secrets are injected

3. **Manual Testing** - Should be performed:
   - Upload pipeline with CLI --step-secret flag
   - Upload pipeline with config.yaml step_secrets
   - Combine CLI and config file secrets
   - Verify invalid formats are rejected
   - Test with clarifai-grpc >= 11.9.8

## Known Limitations

1. **Proto Version Dependency**: Full functionality requires clarifai-grpc >= 11.9.8
2. **No GET/PATCH Methods Yet**: Pipeline class doesn't have methods to fetch or update existing pipeline versions (only handles runs)
3. **CLI Display**: List/get commands don't display step secrets yet (can be added in future)

## Future Enhancements

Potential future improvements (not in scope of current implementation):
- Add `get_pipeline_version()` method to Pipeline class
- Add `patch_pipeline_version()` method to Pipeline class
- Display step secrets in `clarifai pipeline ls` command
- Add `clarifai pipeline version get` command to show secrets
- Add `clarifai pipeline version patch --step-secret` command
- Support for secrets in pipeline version PATCH operations

## Files Modified

1. `clarifai/client/pipeline.py` - Added helper methods
2. `clarifai/runners/pipelines/pipeline_builder.py` - Added proto population
3. `clarifai/cli/pipeline.py` - Added CLI flag
4. `tests/test_pipeline_client.py` - Added tests
5. `tests/test_pipeline_step_secrets.py` - New test file
6. `tests/cli/test_pipeline.py` - Added CLI tests

## Files Created

1. `tests/test_pipeline_step_secrets.py` - New test file
2. `PIPELINE_STEP_SECRETS_EXAMPLES.md` - Usage examples
3. `IMPLEMENTATION_SUMMARY.md` - This file
