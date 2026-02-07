# SDK Actions

The chat agent can execute SDK operations via structured JSON actions. Use a ```json code block with the action format.

## Read-only Actions (auto-executed)

These run automatically without confirmation:

### List Apps
```json
{"action": "list_apps"}
```

### List Models
```json
{"action": "list_models"}
```

### List Models in Specific App
```json
{"action": "list_models", "app_id": "my-app"}
```

### List Datasets
```json
{"action": "list_datasets", "app_id": "my-app"}
```

### List Workflows
```json
{"action": "list_workflows", "app_id": "my-app"}
```

### List Concepts
```json
{"action": "list_concepts", "app_id": "my-app"}
```

### List Pipelines
```json
{"action": "list_pipelines"}
```

### List Compute Clusters
```json
{"action": "list_compute_clusters"}
```

### List Runners
```json
{"action": "list_runners"}
```

### Get User Info
```json
{"action": "get_user_info"}
```

## Actions Requiring Confirmation

These ask for user confirmation before executing:

### Delete an App
```json
{"action": "delete_app", "app_id": "app-to-delete"}
```

### Create an App
```json
{"action": "create_app", "app_id": "new-app"}
```

With optional parameters:
```json
{"action": "create_app", "app_id": "new-app", "base_workflow": "Universal", "description": "My new app"}
```

### Delete a Model
```json
{"action": "delete_model", "app_id": "my-app", "model_id": "model-to-delete"}
```

### Delete a Dataset
```json
{"action": "delete_dataset", "app_id": "my-app", "dataset_id": "dataset-to-delete"}
```

### Delete a Workflow
```json
{"action": "delete_workflow", "app_id": "my-app", "workflow_id": "workflow-to-delete"}
```

### Create a Dataset
```json
{"action": "create_dataset", "app_id": "my-app", "dataset_id": "new-dataset"}
```

## Example Conversations

**User:** "list my apps"
**Response:**
```json
{"action": "list_apps"}
```

**User:** "delete my app called goober"
**Response:**
```json
{"action": "delete_app", "app_id": "goober"}
```

**User:** "show me all models in my project-x app"
**Response:**
```json
{"action": "list_models", "app_id": "project-x"}
```

**User:** "create a new app called test-app"
**Response:**
```json
{"action": "create_app", "app_id": "test-app"}
```
