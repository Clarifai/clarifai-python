# Input Arguments Override Examples

This document demonstrates how to use input argument overrides when running pipelines.

## Overview

Input argument overrides allow you to dynamically override orchestration-specific parameters (e.g., Argo Workflow arguments) for each pipeline run without requiring new PipelineVersions. This is particularly useful for:

- Prompt injection for Agentic AI
- Dynamically adjusting model parameters
- Testing different configurations
- Supporting multi-tenant use cases

## CLI Usage

### 1. Inline Parameter Overrides

Use the `--set` flag to provide inline parameter overrides:

```bash
clarifai pipeline run \
  --pipeline_id=my-pipeline \
  --pipeline_version_id=v1 \
  --user_id=user123 \
  --app_id=app456 \
  --compute_cluster_id=cc1 \
  --nodepool_id=np1 \
  --set prompt="Summarize this research paper" \
  --set temperature="0.7" \
  --set max_tokens="500"
```

### 2. File-Based Overrides

Create a JSON file with your overrides:

**overrides.json:**
```json
{
  "prompt": "Summarize this research paper",
  "temperature": "0.7",
  "max_tokens": "500"
}
```

Then use the `--overrides-file` flag:

```bash
clarifai pipeline run \
  --pipeline_id=my-pipeline \
  --pipeline_version_id=v1 \
  --user_id=user123 \
  --app_id=app456 \
  --compute_cluster_id=cc1 \
  --nodepool_id=np1 \
  --overrides-file overrides.json
```

### 3. Combining Both Methods

Inline parameters take precedence over file parameters:

```bash
clarifai pipeline run \
  --pipeline_id=my-pipeline \
  --pipeline_version_id=v1 \
  --user_id=user123 \
  --app_id=app456 \
  --compute_cluster_id=cc1 \
  --nodepool_id=np1 \
  --overrides-file overrides.json \
  --set prompt="Override the file prompt"
```

## SDK Usage

### Basic Example

```python
from clarifai.client.pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    pipeline_id='my-pipeline',
    pipeline_version_id='v1',
    user_id='user123',
    app_id='app456',
    nodepool_id='nodepool1',
    compute_cluster_id='cluster1',
    pat='your-personal-access-token'
)

# Run with input argument overrides
result = pipeline.run(
    input_args_override={
        "prompt": "Summarize this research paper",
        "temperature": "0.7",
        "max_tokens": "500"
    }
)

print(f"Pipeline run status: {result['status']}")
```

### Loading Overrides from File

```python
import json
from clarifai.client.pipeline import Pipeline

# Load overrides from file (values are automatically converted to strings)
with open('overrides.json', 'r') as f:
    overrides = json.load(f)

# Note: load_overrides_from_file helper can be used instead
# from clarifai.utils.pipeline_overrides import load_overrides_from_file
# overrides = load_overrides_from_file('overrides.json')

# Run pipeline with overrides
pipeline = Pipeline(
    pipeline_id='my-pipeline',
    pipeline_version_id='v1',
    user_id='user123',
    app_id='app456',
    nodepool_id='nodepool1',
    compute_cluster_id='cluster1',
    pat='your-personal-access-token'
)

result = pipeline.run(input_args_override=overrides)
```

### Dynamic Overrides in a Loop

```python
from clarifai.client.pipeline import Pipeline

pipeline = Pipeline(
    pipeline_id='my-pipeline',
    pipeline_version_id='v1',
    user_id='user123',
    app_id='app456',
    nodepool_id='nodepool1',
    compute_cluster_id='cluster1',
    pat='your-personal-access-token'
)

# Run pipeline with different prompts
prompts = [
    "Summarize this document",
    "Extract key findings",
    "Generate an abstract"
]

results = []
for prompt in prompts:
    result = pipeline.run(
        input_args_override={"prompt": prompt, "temperature": "0.7"}
    )
    results.append(result)
    print(f"Completed run with prompt: {prompt}")
```

## Helper Utilities

The SDK provides utility functions for working with overrides:

```python
from clarifai.utils.pipeline_overrides import (
    parse_set_parameter,
    load_overrides_from_file,
    merge_override_parameters,
    build_argo_args_override,
    validate_override_parameters
)

# Parse CLI-style parameter
key, value = parse_set_parameter("temperature=0.7")

# Load from file
file_overrides = load_overrides_from_file("overrides.json")

# Merge inline and file parameters (inline takes precedence)
inline_overrides = {"prompt": "Custom prompt"}
final_overrides = merge_override_parameters(inline_overrides, file_overrides)

# Validate parameters against allowed set
is_valid, error = validate_override_parameters(
    final_overrides,
    allowed_params={"prompt", "temperature", "max_tokens"}
)

if not is_valid:
    print(f"Validation error: {error}")
    # Handle error appropriately
else:
    print("Parameters are valid")
    # Proceed with pipeline run

# Build Argo-compatible structure
argo_override = build_argo_args_override(final_overrides)
```

## Important Notes

1. **String Values**: All parameter values are treated as strings, following Argo Workflow conventions.

2. **Parameter Validation**: Unknown parameters will be rejected by the backend with clear error messages.

3. **Backward Compatibility**: Running pipelines without overrides continues to work as before. The `input_args_override` parameter is optional.

4. **Proto Compatibility**: The implementation is forward-compatible with future proto updates. When the `input_args_override` field becomes available in the proto, it will be used automatically.

5. **Security**: Only parameters defined in the PipelineVersion's orchestration spec can be overridden. This prevents accidental misconfiguration.

## Troubleshooting

### Error: "Invalid --set parameter format"

Make sure your inline parameters follow the `key=value` format:
```bash
--set temperature=0.7  # Correct
--set temperature 0.7  # Wrong - missing equals sign
```

### Error: "Invalid JSON in overrides file"

Ensure your JSON file is properly formatted:
```json
{
  "prompt": "value",
  "temperature": "0.7"
}
```

### Values Not Being Applied

- Check that parameter names match exactly (case-sensitive)
- Verify all values are strings or will be converted to strings
- Ensure the parameters exist in your pipeline's orchestration spec
