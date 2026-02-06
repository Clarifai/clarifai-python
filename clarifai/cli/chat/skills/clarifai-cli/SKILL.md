---
name: clarifai-cli
description: Execute Clarifai operations using the `clarifai` CLI. Use when the user needs to initialize models/pipelines, upload to Clarifai, run pipelines, manage compute clusters, or authenticate. Covers model init (toolkits, MCP, GitHub templates), pipeline templates, compute orchestration CLI, and all model/pipeline operations.
---

# Clarifai CLI

The `clarifai` CLI provides terminal access to all Clarifai platform operations including model initialization, upload, local testing, pipeline management, and compute orchestration.

## Quick Command Reference

| Task | Command |
|------|---------|
| Login | `clarifai login` |
| Initialize model | `clarifai model init` |
| Init from toolkit | `clarifai model init --toolkit vllm` |
| Init MCP server | `clarifai model init --model-type-id mcp` |
| Init from GitHub | `clarifai model init --github-url <url>` |
| Upload model | `clarifai model upload ./model` |
| Local runner | `clarifai model local-runner ./model` |
| Local gRPC | `clarifai model local-grpc ./model --mode container --port 8000` |
| Local test | `clarifai model local-test ./model --mode container` |
| Predict | `clarifai model predict --model_url <url> --inputs '{"prompt":"Hi"}'` |
| List models | `clarifai model list` |
| Init pipeline | `clarifai pipeline init` |
| Init from template | `clarifai pipeline init --template classifier-pipeline-resnet` |
| Upload pipeline | `clarifai pipeline upload` |
| Run pipeline | `clarifai pipeline run --compute_cluster_id <id> --nodepool_id <id>` |
| List pipelines | `clarifai pipeline list` |
| List templates | `clarifai pipelinetemplate list` |

## Use Case Routing

| User Request | Primary Command | Follow-up |
|--------------|-----------------|-----------|
| "Deploy vLLM model" | `clarifai model init --toolkit vllm` | `clarifai model upload` |
| "Deploy SGLang model" | `clarifai model init --toolkit sglang` | `clarifai model upload` |
| "Deploy HuggingFace model" | `clarifai model init --toolkit huggingface` | `clarifai model upload` |
| "Deploy Ollama model" | `clarifai model init --toolkit ollama` | `clarifai model local-runner` |
| "Create MCP server" | `clarifai model init --model-type-id mcp` | `clarifai model upload` |
| "Create OpenAI-compatible model" | `clarifai model init --model-type-id openai` | `clarifai model upload` |
| "Train image classifier" | `clarifai pipeline init --template classifier-pipeline-resnet` | `clarifai pipeline run` |
| "Train object detector" | `clarifai pipeline init --template detector-pipeline-yolof` | `clarifai pipeline run` |
| "Test model locally" | `clarifai model local-runner ./model` | **Always try first** |
| "Run model as local gRPC" | `clarifai model local-grpc ./model --mode container` | |
| "Run model test in container" | `clarifai model local-test ./model --mode container` | |

## Model Commands

### clarifai model init

Initialize a new model directory structure.

**Usage:**
```bash
clarifai model init [MODEL_PATH] [OPTIONS]
```

**Arguments:**
- `MODEL_PATH` - Path where to create the model directory. Default: current directory

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--toolkit` | `vllm`, `sglang`, `huggingface`, `ollama`, `lmstudio`, `python` | Initialize from toolkit template |
| `--model-type-id` | `mcp`, `openai` | Model type for MCPModelClass or OpenAIModelClass |
| `--model-name` | string | Model name for toolkit (e.g., "meta-llama/Llama-3.2-1B-Instruct") |
| `--github-url` | URL | Clone from GitHub repository |
| `--github-pat` | token | GitHub PAT for private repositories |
| `--port` | number | Port for Ollama/LM Studio server (default: 23333) |
| `--context-length` | number | Context length for Ollama/LM Studio (default: 8192) |

**Examples:**
```bash
# Initialize basic model
clarifai model init ./my-model

# Initialize vLLM model with specific HuggingFace model
clarifai model init ./my-vllm-model --toolkit vllm --model-name "Qwen/Qwen3-4B-Instruct-2507"

# Initialize SGLang model
clarifai model init ./my-sglang-model --toolkit sglang --model-name "unsloth/Llama-3.2-1B-Instruct"

# Initialize HuggingFace transformers model
clarifai model init ./my-hf-model --toolkit huggingface --model-name "meta-llama/Llama-3.2-1B-Instruct"

# Initialize Ollama model
clarifai model init ./my-ollama-model --toolkit ollama --model-name "llama3.1"

# Initialize MCP server
clarifai model init ./my-mcp-server --model-type-id mcp

# Initialize OpenAI-compatible wrapper
clarifai model init ./my-openai-wrapper --model-type-id openai

# Clone from GitHub example
clarifai model init ./my-model --github-url https://github.com/Clarifai/runners-examples/tree/main/llm/hf-llama-3_2-1b-instruct

# Initialize custom Python model
clarifai model init ./my-custom-model --toolkit python
```

**Created Structure:**
```
model/
├── 1/
│   └── model.py          # ModelClass implementation
├── requirements.txt      # Python dependencies
└── config.yaml          # Clarifai configuration
```

### clarifai model upload

Upload a model to Clarifai platform.

**Usage:**
```bash
clarifai model upload [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--stage` | `runtime`, `build`, `upload` | Checkpoint download stage (default: upload) |
| `--skip_dockerfile` | flag | Skip Dockerfile generation |
| `--platform` | string | Target Docker platform (e.g., "linux/amd64") |

**Examples:**
```bash
# Upload model from current directory
clarifai model upload

# Upload model from specific path
clarifai model upload ./my-model

# Upload with specific stage
clarifai model upload ./my-model --stage build

# Upload for specific platform
clarifai model upload ./my-model --platform "linux/amd64"
```

### clarifai model local-test

Test model locally by running its test() method.

**Usage:**
```bash
clarifai model local-test [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--mode` | `env`, `container` | Test in virtual env or Docker container (default: env) |
| `--keep_env` | flag | Keep virtual environment after testing (env mode) |
| `--keep_image` | flag | Keep Docker image after testing (container mode) |
| `--skip_dockerfile` | flag | Skip Dockerfile generation |

**Examples:**
```bash
# Quick test in virtual environment
clarifai model local-test ./my-model --mode env

# Production-like test in container (RECOMMENDED)
clarifai model local-test ./my-model --mode container

# Keep Docker image for debugging
clarifai model local-test ./my-model --mode container --keep_image
```

### clarifai model local-grpc

Run model locally as a gRPC server for testing.

**Usage:**
```bash
clarifai model local-grpc [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--port`, `-p` | number | gRPC server port (default: 8000) |
| `--mode` | `env`, `container` | Run in virtual env or Docker container (default: env) |
| `--keep_env` | flag | Keep virtual environment after running |
| `--keep_image` | flag | Keep Docker image after running |
| `--skip_dockerfile` | flag | Skip Dockerfile generation |

**Examples:**
```bash
# Start gRPC server in container
clarifai model local-grpc ./my-model --mode container --port 8000

# Test with SDK (in another terminal)
export CLARIFAI_API_BASE="localhost:8000"
python test_inference.py
```

### clarifai model local-runner

Run model as a local runner connected to Clarifai platform.

**Usage:**
```bash
clarifai model local-runner [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--pool_size` | number | Number of threads (default: 32) |
| `--mode` | `env`, `container`, `none` | Run environment (default: none) |
| `--suppress-toolkit-logs` | flag | Suppress toolkit logs (e.g., Ollama) |
| `--keep_image` | flag | Keep Docker image after running |

**Examples:**
```bash
# Run as local runner (fastest for Ollama)
clarifai model local-runner ./my-ollama-model

# Run in container mode
clarifai model local-runner ./my-model --mode container

# Clean up previous runner
clarifai computecluster delete local-runner-compute-cluster
clarifai model local-runner ./my-model
```

### clarifai model predict

Make predictions using a deployed model.

**Usage:**
```bash
clarifai model predict [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--model_url` | Model URL (e.g., https://clarifai.com/user/app/models/model) |
| `--model_id`, `--user_id`, `--app_id` | Alternative to model_url |
| `--compute_cluster_id`, `-cc_id` | Compute cluster ID |
| `--nodepool_id`, `-np_id` | Nodepool ID |
| `--deployment_id`, `-dpl_id` | Deployment ID |
| `--inputs` | JSON string of inputs |
| `--method` | Method to call (default: predict) |
| `--config` | Path to config file |

**Examples:**
```bash
# Predict with model URL
clarifai model predict --model_url "https://clarifai.com/anthropic/completion/models/claude-sonnet-4" \
  --inputs '{"prompt": "Hello, how are you?"}'

# Predict with compute resources
clarifai model predict --model_id my-model --user_id myuser --app_id myapp \
  --compute_cluster_id cc-123 --nodepool_id np-456 \
  --inputs '{"prompt": "Hello"}'

# Predict with deployment
clarifai model predict --model_url "https://clarifai.com/user/app/models/model" \
  --deployment_id dpl-789 \
  --inputs '{"prompt": "Hello"}'
```

### clarifai model list

List models.

**Usage:**
```bash
clarifai model list [USER_ID] [OPTIONS]
```

**Arguments:**
- `USER_ID` - User ID (default: current user, use "all" for all public models)

**Options:**
| Option | Description |
|--------|-------------|
| `--app_id`, `-a` | Filter by app ID |

**Examples:**
```bash
# List your models
clarifai model list

# List models in specific app
clarifai model list --app_id myapp

# List all public models
clarifai model list all
```

### clarifai model download-checkpoints

Download model checkpoints from external sources.

**Usage:**
```bash
clarifai model download-checkpoints [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `--out_path` | path | Output path for checkpoints |
| `--stage` | `runtime`, `build`, `upload` | Download stage (default: build) |

**Examples:**
```bash
# Download checkpoints for build
clarifai model download-checkpoints ./my-model --stage build

# Download to specific path
clarifai model download-checkpoints ./my-model --out_path ./checkpoints
```

### clarifai model signatures

Generate model method signatures.

**Usage:**
```bash
clarifai model signatures [MODEL_PATH] [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--out_path` | Output file path (default: stdout) |

**Examples:**
```bash
# Print signatures to console
clarifai model signatures ./my-model

# Save to file
clarifai model signatures ./my-model --out_path signatures.yaml
```

## Pipeline Commands

### clarifai pipeline init

Initialize a new pipeline project.

**Usage:**
```bash
clarifai pipeline init [PIPELINE_PATH] [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--template` | Initialize from template (e.g., classifier-pipeline-resnet, detector-pipeline-yolof) |

**Examples:**
```bash
# Interactive initialization
clarifai pipeline init ./my-pipeline

# Initialize from ResNet training template
clarifai pipeline init ./classifier --template classifier-pipeline-resnet

# Initialize from YOLOF detector template
clarifai pipeline init ./detector --template detector-pipeline-yolof
```

**Created Structure:**
```
pipeline/
├── config.yaml          # Pipeline configuration
├── stepA/               # First step
│   ├── config.yaml
│   ├── requirements.txt
│   └── 1/
│       └── pipeline_step.py
├── stepB/               # Second step
│   └── ...
└── README.md
```

### clarifai pipeline upload

Upload a pipeline to Clarifai.

**Usage:**
```bash
clarifai pipeline upload [PATH] [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--no-lockfile` | Skip creating config-lock.yaml |

**Examples:**
```bash
# Upload pipeline from current directory
clarifai pipeline upload

# Upload from specific path
clarifai pipeline upload ./my-pipeline
```

### clarifai pipeline run

Run a pipeline and monitor progress.

**Usage:**
```bash
clarifai pipeline run [OPTIONS]
```

**Required Options:**
| Option | Description |
|--------|-------------|
| `--compute_cluster_id` | Compute cluster ID (REQUIRED) |
| `--nodepool_id` | Nodepool ID (REQUIRED) |

**Pipeline Identification (one required):**
| Option | Description |
|--------|-------------|
| `--pipeline_url` | Full pipeline URL |
| `--pipeline_id` + `--user_id` + `--app_id` + `--pipeline_version_id` | Individual IDs |

**Additional Options:**
| Option | Description |
|--------|-------------|
| `--config` | Path to config file |
| `--timeout` | Max wait time in seconds (default: 3600) |
| `--monitor_interval` | Status check interval (default: 10) |
| `--log_file` | Path for log output |
| `--monitor` | Monitor existing run (requires pipeline_version_run_id) |
| `--set` | Override parameters inline (e.g., --set key=value) |
| `--overrides-file` | JSON/YAML file with parameter overrides |

**Examples:**
```bash
# Run pipeline with config-lock.yaml (auto-detected)
clarifai pipeline run --compute_cluster_id cc-123 --nodepool_id np-456

# Run with explicit pipeline URL
clarifai pipeline run \
  --pipeline_url "https://clarifai.com/user/app/pipelines/my-pipeline/versions/v1" \
  --compute_cluster_id cc-123 \
  --nodepool_id np-456

# Run with parameter overrides
clarifai pipeline run \
  --compute_cluster_id cc-123 \
  --nodepool_id np-456 \
  --set epochs=10 \
  --set batch_size=32

# Monitor existing run
clarifai pipeline run \
  --compute_cluster_id cc-123 \
  --nodepool_id np-456 \
  --pipeline_version_run_id run-789 \
  --monitor
```

### clarifai pipeline list

List pipelines.

**Usage:**
```bash
clarifai pipeline list [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--page_no` | Page number (default: 1) |
| `--per_page` | Items per page (default: 16) |
| `--app_id` | Filter by app ID |

**Examples:**
```bash
# List all pipelines
clarifai pipeline list

# List pipelines in specific app
clarifai pipeline list --app_id myapp
```

## Pipeline Template Commands

### clarifai pipelinetemplate list

List available pipeline templates.

**Usage:**
```bash
clarifai pipelinetemplate list
```

**Available Templates:**
| Template | Description |
|----------|-------------|
| `classifier-pipeline-resnet` | ResNet-50 image classification training |
| `detector-pipeline-yolof` | YOLOF object detection training |

### clarifai pipelinetemplate info

Get information about a specific template.

**Usage:**
```bash
clarifai pipelinetemplate info <template_name>
```

**Examples:**
```bash
clarifai pipelinetemplate info classifier-pipeline-resnet
```

## Pipeline Run Commands

### clarifai pipelinerun monitor

Monitor a pipeline run.

**Usage:**
```bash
clarifai pipelinerun monitor <run_id> [OPTIONS]
```

### clarifai pipelinerun pause / resume / cancel

Control pipeline execution.

**Usage:**
```bash
clarifai pipelinerun pause <run_id>
clarifai pipelinerun resume <run_id>
clarifai pipelinerun cancel <run_id>
```

## Pipeline Step Commands

### clarifai pipelinestep init

Initialize a new pipeline step.

**Usage:**
```bash
clarifai pipelinestep init [STEP_PATH] [OPTIONS]
```

## Authentication Commands

### clarifai login

Login to Clarifai CLI.

**Usage:**
```bash
clarifai login
```

This creates or updates your CLI context with your PAT (Personal Access Token).

### clarifai config

Manage CLI configuration contexts.

**Usage:**
```bash
clarifai config list
clarifai config use <context_name>
clarifai config delete <context_name>
```

## Compute Orchestration Commands

### clarifai computecluster

Manage compute clusters.

**Usage:**
```bash
clarifai computecluster list
clarifai computecluster create <cluster_id>
clarifai computecluster delete <cluster_id>
```

**Example (cleanup for local-runner):**
```bash
clarifai computecluster delete local-runner-compute-cluster
```

### clarifai nodepool

Manage nodepools within compute clusters.

**Usage:**
```bash
clarifai nodepool list --compute_cluster_id <id>
clarifai nodepool create <nodepool_id> --compute_cluster_id <id>
clarifai nodepool delete <nodepool_id> --compute_cluster_id <id>
```

### clarifai deployment

Manage model deployments.

**Usage:**
```bash
clarifai deployment list
clarifai deployment create
clarifai deployment delete <deployment_id>
```

## Artifact Commands

Manage pipeline artifacts for storing checkpoints, logs, models, and other files between pipeline steps.

### clarifai artifact list

List artifacts in an app or versions of an artifact.

```bash
# List all artifacts in an app
clarifai artifact list users/<user_id>/apps/<app_id>
clarifai af ls users/<user_id>/apps/<app_id>

# List versions of a specific artifact
clarifai artifact list users/<user_id>/apps/<app_id>/artifacts/<artifact_id> --versions
```

### clarifai artifact cp (upload/download)

Upload or download artifact files.

**Upload:**
```bash
# Basic upload
clarifai artifact cp ./model.pt users/<user_id>/apps/<app_id>/artifacts/<artifact_id>
clarifai af cp ./weights.safetensors users/u/apps/a/artifacts/model

# Upload with options
clarifai artifact cp ./model.pt users/u/apps/a/artifacts/model \
  --description="Production model v2.0" \
  --visibility=public
```

**Download:**
```bash
# Download latest version
clarifai artifact cp users/<user_id>/apps/<app_id>/artifacts/<artifact_id> ./downloads/

# Download specific version
clarifai artifact cp users/u/apps/a/artifacts/model/versions/v123 ./model.pt
```

### clarifai artifact get

Get artifact or version details.

```bash
clarifai artifact get users/<user_id>/apps/<app_id>/artifacts/<artifact_id>
clarifai af get users/u/apps/a/artifacts/model/versions/v123
```

### clarifai artifact delete

Delete artifacts or versions.

```bash
# Delete artifact
clarifai artifact delete users/<user_id>/apps/<app_id>/artifacts/<artifact_id>

# Delete specific version
clarifai artifact delete users/u/apps/a/artifacts/model/versions/v123

# Force delete (no confirmation)
clarifai af delete users/u/apps/a/artifacts/model --force
```

### Artifact CLI Options

| Option | Description |
|--------|-------------|
| `--visibility` | `private` (default), `public`, `org` |
| `--description` | Version description |
| `--expires-at` | RFC3339 expiration: `2024-12-31T23:59:59.999Z` |
| `--force`, `-f` | Skip confirmations/overwrite files |
| `--versions` | List artifact versions |

## Extended Search

For edge cases not covered here, search:
- **CLI source code**: https://github.com/Clarifai/clarifai-python/tree/main/clarifai/cli
- **Official documentation**: https://docs.clarifai.com/resources/api-overview/cli
- **Examples repository**: https://github.com/Clarifai/examples

## References

- Complete command reference: [references/commands.md](references/commands.md)
- Authentication guide: [references/authentication.md](references/authentication.md)