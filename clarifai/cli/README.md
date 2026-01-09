# Clarifai CLI

## Overview

The Clarifai CLI provides powerful commands for the complete machine learning lifecycle, from model development to production deployment. Key features include:

* **Model Development & Testing**: Initialize, develop, and test models locally
* **Pipeline Management**: Create and manage complex ML workflows  
* **Compute Orchestration**: Deploy and scale models on cloud infrastructure
* **Local Development**: Support for multiple AI toolkits (vLLM, Hugging Face, LMStudio, Ollama)
* **Context Management**: Manage multiple authentication profiles and environments
* **No MLOps Expertise Required**: Streamlined workflows for rapid development and deployment

## Quick Start

### Installation & Setup
```bash
# Install from PyPI
pip install -U clarifai

# Verify installation
clarifai --version

# Login and configure
clarifai login
```

## Model Operations

### Initialize New Model Project
```bash
# Create a new model project structure
clarifai model init my-model

# This creates:
# ├── config.yaml           # Model configuration
# ├── requirements.txt      # Dependencies  
# ├── 1/
# │   └── model.py         # Model implementation
# └── Dockerfile           # Container configuration
```

### Local Development & Testing
```bash
# Run model locally for development and debugging
clarifai model local-runner

# Run model with local gRPC server
clarifai model local-grpc

# Execute model unit tests
clarifai model local-test

# Generate model method signatures
clarifai model signatures
```

### Model Upload & Deployment
```bash
# Upload a trained model to Clarifai
clarifai model upload

# Download model checkpoints
clarifai model download-checkpoints

# List available models
clarifai model list

# Make predictions
clarifai model predict --model-url <model-url> --input <input-data>
```

## Pipeline Operations

### Initialize Pipeline Project
```bash
# Create new pipeline with interactive prompts
clarifai pipeline init my-pipeline

# This creates a complete pipeline structure:
# ├── config.yaml           # Pipeline configuration
# ├── stepA/               # First pipeline step
# │   ├── config.yaml     # Step A configuration
# │   ├── requirements.txt # Step A dependencies
# │   └── 1/
# │       └── pipeline_step.py  # Step A implementation
# ├── stepB/               # Second pipeline step
# └── README.md           # Documentation
```

### Pipeline Management
```bash
# Upload pipeline to Clarifai
clarifai pipeline upload

# List all pipelines
clarifai pipeline list

# Run pipeline and monitor progress
clarifai pipeline run

# Validate pipeline configuration
clarifai pipeline validate-lock
```

## Context Management

Manage CLI contexts for authentication and environment configuration:

### Basic Context Operations
```bash
# List all contexts
clarifai config get-contexts

# Switch context
clarifai config use-context production

# Show current context
clarifai config current-context

# Create new context
clarifai config create-context staging --user-id myuser --pat 678***
```

### Configuration Management
```bash
# View entire configuration
clarifai config view

# Delete a context
clarifai config delete-context old-context

# Edit configuration file
clarifai config edit

# Print environment variables for the active context
clarifai config env
```

## Compute Orchestration

Streamlined infrastructure management for training, deploying, and scaling ML models with automatic scaling and cross-provider support.

### Quick Deployment Example

Complete workflow for deploying a model:

```bash
# 1. Login to Clarifai
clarifai login

# 2. Create compute cluster
clarifai computecluster create --config cluster-config.yaml

# 3. Create nodepool
clarifai nodepool create --config nodepool-config.yaml

# 4. Deploy model
clarifai deployment create --config deployment-config.yaml
```

### Compute Cluster Management
```bash
# Create cluster
clarifai computecluster create --config <cluster-config.yaml>

# List clusters
clarifai computecluster list

# Delete cluster
clarifai computecluster delete --compute_cluster_id <cluster-id>
```

### Nodepool Management
```bash
# Create nodepool
clarifai nodepool create --config <nodepool-config.yaml>

# List nodepools
clarifai nodepool list --compute_cluster_id <cluster-id>

# Delete nodepool
clarifai nodepool delete --compute_cluster_id <cluster-id> --nodepool_id <nodepool-id>
```

### Deployment Management
```bash
# Create deployment
clarifai deployment create --config <deployment-config.yaml>

# List deployments
clarifai deployment list --nodepool_id <nodepool-id>

# Delete deployment
clarifai deployment delete --nodepool_id <nodepool-id> --deployment_id <deployment-id>
```

### Advanced Features
- **Health Probes**: Automatic liveness/readiness checks for deployed models
- **Secrets Management**: Secure handling of API keys and credentials
- **Git Integration**: Automatic metadata capture during deployments
- **Enhanced Monitoring**: Improved logging and diagnostics

## Pipeline Steps Management

### List Pipeline Steps
```bash
# List all pipeline steps
clarifai pipeline-step list

# List steps in specific app
clarifai pipeline-step list --app_id <app-id>

# List steps for specific pipeline
clarifai pipeline-step list --app_id <app-id> --pipeline_id <pipeline-id>

# Using alias
clarifai ps ls
```

### Pipeline Step Operations
```bash
# Upload pipeline step
clarifai pipeline-step upload

# Test pipeline step
clarifai pipeline-step test

# List with pagination
clarifai pipeline-step list --page_no 1 --per_page 10
```

## Local Development Toolkits

The CLI supports multiple AI toolkits for local model development:

### Supported Toolkits
- **vLLM**: High-performance LLM inference
- **Hugging Face**: Extensive model library integration  
- **LMStudio**: Local language model management
- **Ollama**: Local LLM deployment

### Local Development Workflow
```bash
# Initialize with specific toolkit
clarifai model init my-llm-model

# Develop and test locally
clarifai model local-runner

# Upload when ready
clarifai model upload
```

## Additional Commands

### Shell Completion
```bash
# Generate shell completion script
clarifai shell-completion

# For bash
eval "$(clarifai shell-completion bash)"

# For zsh  
eval "$(clarifai shell-completion zsh)"
```

### Context Execution
```bash
# Execute script with current context environment
clarifai run my-script.py
```

### Help & Information
```bash
# Get help for any command
clarifai --help
clarifai model --help
clarifai pipeline --help

# Check version
clarifai --version
```

## Examples & Resources

### Configuration Examples
- **[Compute Orchestration Configs](https://github.com/Clarifai/examples/tree/main/ComputeOrchestration/configs)**
- **[Model Examples Repository](https://github.com/Clarifai/runners-examples)**

### Documentation
- **[Official CLI Documentation](https://docs.clarifai.com/cli/)**
- **[Python SDK Documentation](https://docs.clarifai.com/resources/api-references/python)**
- **[Model Upload Guide](https://docs.clarifai.com/compute/models/upload)**
- **[Compute Orchestration Guide](https://docs.clarifai.com/compute/compute-orchestration/)**

### Community & Support
- **[Clarifai Community](https://clarifai.com/explore)**
- **[Discord Community](https://discord.gg/XAPE3Vtg)**
- **[GitHub Issues](https://github.com/Clarifai/clarifai-python/issues)**

## Getting Help

For command-specific help, use the `--help` flag:

```bash
clarifai --help                    # General help
clarifai model --help             # Model commands
clarifai pipeline --help          # Pipeline commands  
clarifai computecluster --help    # Compute cluster commands
clarifai config --help            # Configuration commands
```
