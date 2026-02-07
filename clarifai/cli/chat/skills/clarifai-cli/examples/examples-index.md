# Clarifai CLI Examples Index

## Command Examples

The SKILL.md contains comprehensive examples for each command. Here are quick references:

### Model Initialization Examples

```bash
# vLLM model
clarifai model init ./vllm-model --toolkit vllm --model-name "Qwen/Qwen3-4B-Instruct-2507"

# SGLang model
clarifai model init ./sglang-model --toolkit sglang --model-name "unsloth/Llama-3.2-1B-Instruct"

# HuggingFace model
clarifai model init ./hf-model --toolkit huggingface --model-name "meta-llama/Llama-3.2-1B-Instruct"

# MCP server
clarifai model init ./mcp-server --model-type-id mcp

# From GitHub
clarifai model init ./my-model --github-url https://github.com/Clarifai/runners-examples/tree/main/llm/hf-llama-3_2-1b-instruct
```

### Pipeline Training Examples

```bash
# Image classifier training
clarifai pipeline init ./classifier --template classifier-pipeline-resnet
clarifai pipeline upload ./classifier
clarifai pipeline run --compute_cluster_id cc-123 --nodepool_id np-456

# Object detector training
clarifai pipeline init ./detector --template detector-pipeline-yolof
clarifai pipeline upload ./detector
clarifai pipeline run --compute_cluster_id cc-123 --nodepool_id np-456
```

### Local Testing Examples

```bash
# Test in container (recommended)
clarifai model local-test ./my-model --mode container

# Run gRPC server
clarifai model local-grpc ./my-model --mode container --port 8000

# Run as local runner
clarifai model local-runner ./my-model
```

## GitHub Examples

For more complex examples, see:

- **Model examples**: https://github.com/Clarifai/runners-examples
- **Pipeline examples**: https://github.com/Clarifai/pipeline-examples
- **General examples**: https://github.com/Clarifai/examples
