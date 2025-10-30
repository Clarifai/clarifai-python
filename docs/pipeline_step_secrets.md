# Pipeline Step Secrets Usage Guide

This guide explains how to use pipeline step secrets in the Clarifai Python SDK.

## Overview

Pipeline step secrets allow different pipeline steps to access distinct sets of secrets with step-level isolation. Each step can have its own set of secret environment variables that are mounted securely at runtime.

## Configuration Format

Define step secrets in your pipeline `config.yaml` file within the `orchestration_spec` section:

```yaml
pipeline:
  id: "my-pipeline"
  user_id: "user123"
  app_id: "app456"
  step_directories:
    - step1
    - step2
  orchestration_spec:
    argo_orchestration_spec: |
      apiVersion: argoproj.io/v1alpha1
      kind: Workflow
      spec:
        entrypoint: sequence
        templates:
        - name: sequence
          steps:
            - - name: step-0
                templateRef:
                  name: users/user123/apps/app456/pipeline_steps/step1
                  template: users/user123/apps/app456/pipeline_steps/step1
            - - name: step-1
                templateRef:
                  name: users/user123/apps/app456/pipeline_steps/step2
                  template: users/user123/apps/app456/pipeline_steps/step2
    
    # Define secrets for each step
    step_version_secrets:
      step-0:
        secrets:
          API_KEY: users/user123/secrets/my-api-key
          DB_PASSWORD: users/user123/secrets/db-secret
      step-1:
        secrets:
          EMAIL_TOKEN: users/user123/secrets/email-token
```

## Secret Reference Format

Secret references follow the pattern: `users/{user_id}/secrets/{secret_name}`

- The secrets must already exist in your Clarifai account
- Only references are stored in the config; actual values are injected at runtime
- Each step can only access its explicitly configured secrets

## CLI Usage

### 1. Initialize a Pipeline with Secrets

```bash
clarifai pipeline init my-pipeline
```

Then edit the generated `config.yaml` to add the `step_version_secrets` section as shown above.

### 2. Upload Pipeline with Secrets

```bash
clarifai pipeline upload config.yaml
```

This will:
1. Upload all pipeline steps from `step_directories`
2. Create the pipeline with step secrets configuration
3. Generate a `config-lock.yaml` file that includes the secrets config

### 3. Run Pipeline

```bash
clarifai pipeline run \
  --pipeline_id my-pipeline \
  --user_id user123 \
  --app_id app456 \
  --compute_cluster_id cluster-id \
  --nodepool_id nodepool-id
```

## Python SDK Usage

**Note on Format**: The Python SDK uses a simplified format for step secrets compared to the YAML configuration. In YAML, secrets are nested under a `secrets` key, but in the Python API you provide them directly as a dictionary.

### Get Pipeline Version with Secrets

```python
from clarifai.client.pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    pipeline_id="my-pipeline",
    pipeline_version_id="version-123",
    user_id="user123",
    app_id="app456",
    pat="your-pat"
)

# Get pipeline version details including secrets
version = pipeline.get_pipeline_version()
print(version['config']['step_version_secrets'])
```

### List Step Secrets

```python
# List all step secrets
all_secrets = pipeline.list_step_secrets()
print(all_secrets)
# Output: {
#   'step-0': {'API_KEY': 'users/user123/secrets/my-api-key', ...},
#   'step-1': {'EMAIL_TOKEN': 'users/user123/secrets/email-token'}
# }

# List secrets for specific step
step0_secrets = pipeline.list_step_secrets(step_ref='step-0')
print(step0_secrets)
# Output: {
#   'step-0': {'API_KEY': 'users/user123/secrets/my-api-key', ...}
# }
```

### Create Pipeline Version with Secrets

```python
from clarifai.client.pipeline import Pipeline

pipeline = Pipeline(
    pipeline_id="my-pipeline",
    user_id="user123",
    app_id="app456",
    pat="your-pat"
)

# Define orchestration spec
orchestration_spec = {
    "argo_orchestration_spec": """
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  entrypoint: sequence
  templates:
  - name: sequence
    steps:
      - - name: step-0
          templateRef:
            name: users/user123/apps/app456/pipeline_steps/step1/versions/v1
            template: users/user123/apps/app456/pipeline_steps/step1/versions/v1
"""
}

# Define step secrets (simplified format for Python API)
# Note: The Python API accepts a simplified format where you provide
# the secrets directly without the 'secrets' wrapper key
step_version_secrets = {
    "step-0": {
        "API_KEY": "users/user123/secrets/my-api-key",
        "DB_PASSWORD": "users/user123/secrets/db-secret"
    }
}

# Create new version
version_id = pipeline.create_pipeline_version(
    orchestration_spec=orchestration_spec,
    step_version_secrets=step_version_secrets,
    description="Pipeline version with step secrets"
)
print(f"Created version: {version_id}")
```

## Security Considerations

- **Step-Level Isolation**: Each step only accesses explicitly configured secrets
- **Reference-Only Storage**: Only secret references are stored in config files
- **Runtime Injection**: Actual secret values are injected securely at runtime
- **Kubernetes Integration**: Backend uses Kubernetes SecretKeyRef for secure mounting
- **No Value Leakage**: Secret values are never logged or exposed in API responses

## Best Practices

1. **Manage Secrets via Config File**: Always define secrets in `config.yaml` rather than trying to add them programmatically
2. **Use Descriptive Names**: Give secrets clear, descriptive names like `API_KEY`, `DB_PASSWORD`
3. **Minimize Access**: Only give each step the secrets it needs
4. **Version Control**: Use `config-lock.yaml` to track which secrets are configured for each pipeline version
5. **Regular Rotation**: Rotate secrets regularly and update references in config

## Troubleshooting

### Secret Not Found Error

If you get an error about a secret not existing:
- Verify the secret exists in your Clarifai account
- Check that the secret reference path is correct: `users/{user_id}/secrets/{secret_name}`
- Ensure you have permission to access the secret

### Secret Not Available in Step

If a step can't access a secret:
- Verify the step reference matches the one in your Argo orchestration spec
- Check that the secret is defined for that specific step in `step_version_secrets`
- Ensure the pipeline was uploaded after adding the secrets configuration

## Example Project Structure

```
my-pipeline/
├── config.yaml              # Pipeline config with step_version_secrets
├── config-lock.yaml         # Generated lockfile with versions
├── step1/
│   ├── config.yaml
│   ├── requirements.txt
│   └── 1/
│       └── pipeline_step.py
└── step2/
    ├── config.yaml
    ├── requirements.txt
    └── 1/
        └── pipeline_step.py
```

## Related Documentation

- [Clarifai Secrets Management](https://docs.clarifai.com/api-guide/secrets)
- [Pipeline Orchestration](https://docs.clarifai.com/api-guide/pipelines)
- [Pipeline Steps](https://docs.clarifai.com/api-guide/pipeline-steps)
