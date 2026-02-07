# Clarifai CLI Complete Command Reference

## Model Commands Detailed Reference

### clarifai model init

**All Options:**
```bash
clarifai model init [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to create model (default: current directory)

Options:
  --model-type-id TEXT    Model type: "mcp" or "openai"
  --toolkit TEXT          Toolkit: vllm, sglang, huggingface, ollama, lmstudio, python
  --model-name TEXT       Model name for toolkit (e.g., HuggingFace repo_id)
  --github-url TEXT       GitHub URL to clone from
  --github-pat TEXT       GitHub PAT for private repos
  --port TEXT             Port for Ollama/LM Studio (default: 23333)
  --context-length TEXT   Context length for Ollama/LM Studio (default: 8192)
```

**Toolkit Options:**
| Toolkit | Use Case | Model Name Format |
|---------|----------|-------------------|
| `vllm` | High-performance LLM serving | HuggingFace repo_id (e.g., "Qwen/Qwen3-4B-Instruct-2507") |
| `sglang` | Structured generation | HuggingFace repo_id |
| `huggingface` | Direct HF Transformers | HuggingFace repo_id |
| `ollama` | Ollama model wrapping | Ollama model name (e.g., "llama3.1") |
| `lmstudio` | LM Studio integration | LM Studio model name |
| `python` | Custom Python implementation | N/A |

### clarifai model upload

**All Options:**
```bash
clarifai model upload [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  --stage TEXT            Checkpoint download stage: runtime, build, upload (default: upload)
  --skip_dockerfile       Skip generating Dockerfile
  --platform TEXT         Target Docker platform (e.g., "linux/amd64")
```

### clarifai model local-test

**All Options:**
```bash
clarifai model local-test [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  --mode TEXT             Test mode: env, container (default: env)
  --keep_env              Keep virtual environment after testing
  --keep_image            Keep Docker image after testing
  --skip_dockerfile       Skip generating Dockerfile
```

### clarifai model local-grpc

**All Options:**
```bash
clarifai model local-grpc [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  -p, --port INTEGER      gRPC server port (default: 8000)
  --mode TEXT             Run mode: env, container (default: env)
  --keep_env              Keep virtual environment after running
  --keep_image            Keep Docker image after running
  --skip_dockerfile       Skip generating Dockerfile
```

### clarifai model local-runner

**All Options:**
```bash
clarifai model local-runner [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  --pool_size INTEGER     Number of threads (default: 32)
  --mode TEXT             Run mode: env, container, none (default: none)
  --suppress-toolkit-logs Suppress toolkit logs (e.g., Ollama)
  --keep_image            Keep Docker image after running
```

### clarifai model predict

**All Options:**
```bash
clarifai model predict [OPTIONS]

Options:
  --config PATH           Path to predict config file
  --model_id TEXT         Model ID
  --user_id TEXT          User ID
  --app_id TEXT           App ID
  --model_url TEXT        Full model URL
  -cc_id, --compute_cluster_id TEXT    Compute cluster ID
  -np_id, --nodepool_id TEXT           Nodepool ID
  -dpl_id, --deployment_id TEXT        Deployment ID
  -dpl_usr_id, --deployment_user_id TEXT   Deployment user ID
  --inputs TEXT           JSON string of inputs
  --method TEXT           Method to call (default: predict)
```

### clarifai model list

**All Options:**
```bash
clarifai model list [USER_ID] [OPTIONS]

Arguments:
  USER_ID                 User ID (default: current user, "all" for public models)

Options:
  -a, --app_id TEXT       Filter by app ID
```

### clarifai model download-checkpoints

**All Options:**
```bash
clarifai model download-checkpoints [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  --out_path PATH         Output path for checkpoints
  --stage TEXT            Download stage: runtime, build, upload (default: build)
```

### clarifai model signatures

**All Options:**
```bash
clarifai model signatures [MODEL_PATH] [OPTIONS]

Arguments:
  MODEL_PATH              Path to model directory (default: current directory)

Options:
  --out_path PATH         Output file path (default: stdout)
```

## Pipeline Commands Detailed Reference

### clarifai pipeline init

**All Options:**
```bash
clarifai pipeline init [PIPELINE_PATH] [OPTIONS]

Arguments:
  PIPELINE_PATH           Path to create pipeline (default: current directory)

Options:
  --template TEXT         Template name (e.g., classifier-pipeline-resnet)
```

### clarifai pipeline upload

**All Options:**
```bash
clarifai pipeline upload [PATH] [OPTIONS]

Arguments:
  PATH                    Path to pipeline directory (default: current directory)

Options:
  --no-lockfile           Skip creating config-lock.yaml
```

### clarifai pipeline run

**All Options:**
```bash
clarifai pipeline run [OPTIONS]

Options:
  --config PATH                     Config file path
  --pipeline_id TEXT                Pipeline ID
  --pipeline_version_id TEXT        Pipeline version ID
  --pipeline_version_run_id TEXT    Pipeline run ID (for monitoring)
  --user_id TEXT                    User ID
  --app_id TEXT                     App ID
  --nodepool_id TEXT                Nodepool ID (REQUIRED)
  --compute_cluster_id TEXT         Compute cluster ID (REQUIRED)
  --pipeline_url TEXT               Full pipeline URL
  --timeout INTEGER                 Max wait time in seconds (default: 3600)
  --monitor_interval INTEGER        Status check interval (default: 10)
  --log_file PATH                   Log file path
  --monitor                         Monitor existing run
  --set TEXT                        Override parameter (can repeat)
  --overrides-file PATH             Parameter overrides file
```

### clarifai pipeline list

**All Options:**
```bash
clarifai pipeline list [OPTIONS]

Options:
  --page_no INTEGER       Page number (default: 1)
  --per_page INTEGER      Items per page (default: 16)
  --app_id TEXT           Filter by app ID
```

### clarifai pipeline validate-lock

**All Options:**
```bash
clarifai pipeline validate-lock [LOCKFILE_PATH]

Arguments:
  LOCKFILE_PATH           Path to config-lock.yaml (default: config-lock.yaml)
```

## Pipeline Template Commands

### clarifai pipelinetemplate list

```bash
clarifai pipelinetemplate list
```

### clarifai pipelinetemplate info

```bash
clarifai pipelinetemplate info <TEMPLATE_NAME>
```

## Compute Commands

### clarifai computecluster

```bash
clarifai computecluster list
clarifai computecluster create <CLUSTER_ID> [OPTIONS]
clarifai computecluster delete <CLUSTER_ID>
clarifai computecluster info <CLUSTER_ID>
```

### clarifai nodepool

```bash
clarifai nodepool list --compute_cluster_id <ID>
clarifai nodepool create <NODEPOOL_ID> --compute_cluster_id <ID> [OPTIONS]
clarifai nodepool delete <NODEPOOL_ID> --compute_cluster_id <ID>
```

### clarifai deployment

```bash
clarifai deployment list
clarifai deployment create [OPTIONS]
clarifai deployment delete <DEPLOYMENT_ID>
```

## Authentication Commands

### clarifai login

```bash
clarifai login
```

Interactive login to create/update CLI context.

### clarifai config

```bash
clarifai config list
clarifai config use <CONTEXT_NAME>
clarifai config delete <CONTEXT_NAME>
clarifai config show
```

## Artifact Commands

### clarifai artifact

```bash
clarifai artifact list
clarifai artifact upload <PATH>
clarifai artifact download <ARTIFACT_ID>
```
