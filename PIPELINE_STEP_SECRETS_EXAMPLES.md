# Pipeline Step Secrets Usage Examples

## Using the Python SDK

### Example 1: Adding step secrets to a Pipeline object

```python
from clarifai.client.pipeline import Pipeline

# Initialize a pipeline
pipeline = Pipeline(
    pipeline_id='my-pipeline',
    user_id='user123',
    app_id='app456',
    pat='your-pat-token'
)

# Add secrets to specific steps
pipeline.add_step_secret('step1', 'API_KEY', 'users/user123/secrets/my-api-key')
pipeline.add_step_secret('step1', 'DB_PASSWORD', 'users/user123/secrets/db-secret')
pipeline.add_step_secret('step2', 'EMAIL_TOKEN', 'users/user123/secrets/email-token')

# List secrets for a specific step
step1_secrets = pipeline.list_step_secrets('step1')
print(f"Step1 secrets: {step1_secrets}")
# Output: {'API_KEY': 'users/user123/secrets/my-api-key', 'DB_PASSWORD': 'users/user123/secrets/db-secret'}

# Get all step secrets
all_secrets = pipeline.get_step_secrets()
print(f"All step secrets: {all_secrets}")
# Output: {
#     'step1': {'API_KEY': '...', 'DB_PASSWORD': '...'},
#     'step2': {'EMAIL_TOKEN': '...'}
# }

# Remove a secret
pipeline.remove_step_secret('step1', 'API_KEY')
```

### Example 2: Using step secrets in config.yaml

```yaml
pipeline:
  id: my-pipeline
  user_id: user123
  app_id: app456
  orchestration_spec:
    argo_orchestration_spec: |
      apiVersion: argoproj.io/v1alpha1
      kind: Workflow
      spec:
        templates:
          - name: main
            steps:
              - - name: process-step
                  templateRef:
                    name: users/user123/apps/app456/pipeline-steps/step1
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

## Using the CLI

### Example 1: Upload pipeline with step secrets via CLI flags

```bash
# Upload pipeline with a single step secret
clarifai pipeline upload --step-secret step1:API_KEY=users/user123/secrets/my-key

# Upload with multiple secrets for the same step
clarifai pipeline upload \
  --step-secret step1:API_KEY=users/user123/secrets/my-key \
  --step-secret step1:DB_PASSWORD=users/user123/secrets/db-pass

# Upload with secrets for multiple steps
clarifai pipeline upload \
  --step-secret step1:API_KEY=users/user123/secrets/my-key \
  --step-secret step1:DB_PASSWORD=users/user123/secrets/db-pass \
  --step-secret step2:EMAIL_TOKEN=users/user123/secrets/email-token

# Upload from specific config path with secrets
clarifai pipeline upload /path/to/config.yaml \
  --step-secret step1:API_KEY=users/user123/secrets/my-key
```

### Example 2: Combining config file and CLI secrets

If your config.yaml already has some step secrets defined:

```yaml
pipeline:
  # ... other config ...
  step_secrets:
    step1:
      API_KEY: users/user123/secrets/default-key
```

And you run:

```bash
clarifai pipeline upload --step-secret step1:API_KEY=users/user123/secrets/override-key
```

The CLI-provided secret will override the config file value. This allows for:
- Default secrets in config.yaml for development
- Override secrets via CLI for different environments (staging, production)

## Secret Reference Format

Secret references follow the Clarifai resource path format:
```
users/{user_id}/secrets/{secret_id}
```

Examples:
- `users/alice/secrets/openai-api-key`
- `users/company-org/secrets/database-password`
- `users/bob/secrets/email-service-token`

## Security Best Practices

1. **Never commit actual secret values** - Only commit secret references
2. **Use descriptive secret IDs** - Makes it easier to manage in Clarifai console
3. **Principle of least privilege** - Each step should only have access to secrets it needs
4. **Rotate secrets regularly** - Update secret values in Clarifai console without changing pipeline config

## Backend Behavior

When a pipeline runs:
1. The backend validates all secret references exist and user has access
2. For each step, only its configured secrets are mounted as environment variables
3. Secret values are injected at runtime - never stored in pipeline config
4. Each step runs in isolation with its own set of secrets

## Backward Compatibility

The step_version_secrets field is **completely optional**:
- Existing pipelines without secrets continue to work unchanged
- You can gradually add secrets to existing pipelines
- Mixing steps with and without secrets in the same pipeline is supported
