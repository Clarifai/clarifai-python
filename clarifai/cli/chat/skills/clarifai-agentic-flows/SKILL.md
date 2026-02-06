---
name: clarifai-agentic-flows
description: Orchestrate multi-step Clarifai workflows. Use when the user needs to combine multiple Clarifai operations (e.g., "upload AND train", "train THEN deploy", "classify THEN describe"), build end-to-end ML pipelines, create orchestrator models for single-endpoint access, or design composite workflows. Triggers on sequential language ("then", "after", "followed by"), conjunctions combining operations ("and", "plus"), workflow/pipeline planning, multi-model chaining, or any request involving 2+ distinct Clarifai capabilities. Route here FIRST when detecting composite intent, then delegate to building-block skills.
---

# Agentic Flow Orchestration

Compose multiple Clarifai operations into cohesive workflows.

## Trigger Detection

**Route here FIRST when request contains ANY of:**
- **Composite operations**: "upload **and** train", "train **then** deploy", "detect **and** describe"
- **Sequential language**: "then", "after that", "followed by", "next", "finally"
- **Conjunction patterns**: "A **and** B", "A **plus** B", "A **with** B" (where A, B are distinct operations)
- **End-to-end requests**: "complete workflow", "full pipeline", "entire process", "start to finish"
- **Multi-model**: "chain models", "combine models", "orchestrate", "coordinate"
- **Single-endpoint hosting**: "single endpoint", "one API call", "unified interface"

**Key detection rule**: If the request mentions **2+ distinct Clarifai capabilities** (dataset, training, deployment, inference, MCP), route here to orchestrate—even if individual keywords would match other skills.

**Route to specific skill ONLY for clearly single operations:**
- Dataset upload only (no training/deployment mentioned) → `clarifai-datasets`
- Training only (no deployment/inference mentioned) → `clarifai-training-pipelines`
- Model deployment only (no chaining/orchestration) → `clarifai-model-upload`
- Inference only (single model, no chaining) → `clarifai-inference`
- MCP server only (no workflow context) → `clarifai-mcp`
- Custom pipeline steps only (no end-to-end context) → `clarifai-pipelines`

## Building Block Skills

| Skill | Use For |
|-------|---------|
| `clarifai-datasets` | Upload data with annotations (Classification, Detection, Segmentation) |
| `clarifai-training-pipelines` | Train classifiers (ResNet) or detectors (YOLOF) |
| `clarifai-model-upload` | Deploy custom models, orchestrators |
| `clarifai-mcp` | Tool-calling MCP servers |
| `clarifai-inference` | Model discovery, `predict()`, OpenAI API |
| `clarifai-pipelines` | Custom batch/ETL workflows |

## Decision Framework

### When to Use Each Pattern

| Factor | Client-Side Script | Server-Side Orchestrator | Platform Pipeline |
|--------|-------------------|-------------------------|-------------------|
| **Duration** | < 5 min | < 30 min | > 30 min |
| **Complexity** | Any | Complex logic | Heavy processing |
| **Location** | Local machine | Clarifai cloud | Clarifai cloud |
| **Scale** | Single user | Multiple users | Batch processing |
| **State** | Stateless | Can maintain state | Persistent state |

### Decision Tree

```
Does the workflow involve MODEL TRAINING?
├── YES → What kind of training?
│   ├── Image classifier with ResNet-50? → clarifai-training-pipelines (template)
│   ├── Object detector with YOLOF? → clarifai-training-pipelines (template)
│   └── ANY other training (LLM, segmentation, custom model)?
│       └── clarifai-pipelines (custom training pipeline - NOT vanilla Python)
│   └── THEN for deployment → clarifai-model-upload or clarifai-inference
└── NO → Is the workflow long-running (> 30 min)?
    ├── YES → Platform Pipeline (clarifai-pipelines)
    └── NO → Does it need to be a single endpoint?
        ├── YES → Server-Side Orchestrator Model
        └── NO → Client-Side Script
```

**CRITICAL: Training Template Limits**
- `clarifai-training-pipelines` has **ONLY 2 templates**: `classifier-pipeline-resnet` (ResNet-50) and `detector-pipeline-yolof` (YOLOF)
- **DO NOT suggest any other init --template options** - they don't exist
- For ANY other training (LLM fine-tuning, segmentation, custom models), use `clarifai-pipelines` to create custom training pipeline
- **NEVER generate vanilla Python scripts for training** - always wrap in pipeline format

**For "train then deploy" requests:**
1. First check if ResNet-50 classifier or YOLOF detector → `clarifai-training-pipelines`
2. Otherwise → `clarifai-pipelines` for custom training pipeline
3. Then route to `clarifai-model-upload` or `clarifai-inference` for deployment/inference

## Pattern 1: Client-Side Script

**Best for:** Quick integrations, local control, simple chains.

### Example: Image Analysis Pipeline

```python
import os
from clarifai.client import Model

PAT = os.environ["CLARIFAI_PAT"]

def analyze_and_describe_image(image_url: str) -> dict:
    """Chain object detection with LLM description."""

    # Step 1: Detect objects
    detector = Model(
        url="https://clarifai.com/clarifai/main/models/general-image-detection"
    )
    detections = detector.predict(image=image_url)

    # Step 2: Generate description with LLM
    llm = Model(
        url="https://clarifai.com/anthropic/completion/models/claude-sonnet-4"
    )

    prompt = f"""Based on these detected objects: {detections}
    Please provide a detailed description of the scene."""

    description = llm.predict(prompt=prompt)

    return {
        "detections": detections,
        "description": description
    }

# Usage
result = analyze_and_describe_image("https://example.com/image.jpg")
print(result["description"])
```

### Example: Multi-Model Comparison

```python
def compare_llm_responses(prompt: str) -> dict:
    """Compare responses from multiple LLMs."""

    models = {
        "claude": "https://clarifai.com/anthropic/completion/models/claude-sonnet-4",
        "gpt4": "https://clarifai.com/openai/chat-completion/models/gpt-4o",
        "llama": "https://clarifai.com/meta/Llama-3/models/llama-3_1-70b-instruct",
    }

    responses = {}
    for name, url in models.items():
        model = Model(url=url)
        responses[name] = model.predict(prompt=prompt)

    return responses
```

## Pattern 2: Server-Side Orchestrator Model

**Best for:** Single endpoint access, hosted logic, multiple consumers.

### Example: Orchestrator ModelClass

```python
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.client import Model
import os

class WorkflowOrchestrator(ModelClass):
    """Orchestrator that chains multiple Clarifai models."""

    def load_model(self):
        """Initialize client models."""
        self.classifier = Model(
            url="https://clarifai.com/clarifai/main/models/general-image-recognition"
        )
        self.llm = Model(
            url="https://clarifai.com/anthropic/completion/models/claude-sonnet-4"
        )

    @ModelClass.method
    def predict(
        self,
        image_url: str = "",
        task: str = Param(default="describe", description="Task: describe, analyze, caption")
    ) -> str:
        """Orchestrate multiple models based on task."""

        # Step 1: Classify image
        classifications = self.classifier.predict(image=image_url)

        # Step 2: Generate task-specific output
        if task == "describe":
            prompt = f"Describe this image with these classifications: {classifications}"
        elif task == "analyze":
            prompt = f"Provide a detailed analysis of an image with: {classifications}"
        elif task == "caption":
            prompt = f"Create a short caption for an image showing: {classifications}"
        else:
            prompt = f"Process this image data: {classifications}"

        return self.llm.predict(prompt=prompt)
```

### Deploy Orchestrator

1. Create directory structure: `orchestrator/1/model.py`
2. Add `config.yaml` and `requirements.txt` (see config example below)
3. **Load the `clarifai-cli` skill** for upload command (`clarifai model upload`)

### config.yaml for Orchestrator

```yaml
model:
  id: "workflow-orchestrator"
  user_id: "YOUR_USER_ID"
  app_id: "YOUR_APP_ID"
  model_type_id: "text-to-text"

build_info:
  python_version: "3.11"

inference_compute_info:
  cpu_limit: "2"
  cpu_memory: "4Gi"
  num_accelerators: 0
```

## Pattern 3: Hybrid Approach

**Best for:** Complex workflows with both quick and long-running parts.

### Example: Upload, Train, Deploy

```python
import os
from clarifai.client import App, Model
from clarifai.datasets.upload.base import ClarifaiDataset

def end_to_end_workflow(data_path: str, model_name: str):
    """Complete workflow: upload data, train, deploy."""

    app = App(
        user_id="YOUR_USER_ID",
        app_id="YOUR_APP_ID",
        pat=os.environ["CLARIFAI_PAT"]
    )

    # Step 1: Upload dataset (client-side, quick)
    print("Step 1: Uploading dataset...")
    dataset = app.create_dataset(dataset_id=f"{model_name}-data")
    # ... upload logic ...

    # Step 2: Trigger training pipeline (platform, long-running)
    print("Step 2: Starting training pipeline...")
    # Use clarifai pipeline run for training
    import subprocess
    subprocess.run([
        "clarifai", "pipeline", "run",
        "--pipeline_url", f"https://clarifai.com/{app.user_id}/{app.id}/pipelines/classifier-pipeline",
        "--compute_cluster_id", "YOUR_CLUSTER",
        "--nodepool_id", "YOUR_NODEPOOL",
        "--set", f"dataset_id={dataset.id}"
    ])

    # Step 3: Wait for training and deploy (client-side orchestration)
    print("Step 3: Monitoring training...")
    # ... monitor and deploy logic ...

    return f"Workflow complete for {model_name}"
```

## Quick Request Routing

| User Request | Pattern | Building Block Skills |
|--------------|---------|----------------------|
| "Chain Q&A and sentiment locally" | Client Script | `clarifai-inference` |
| "Single endpoint that chains models" | Orchestrator | `clarifai-model-upload` + `clarifai-inference` |
| "Upload 50k images and train ResNet classifier" | Pipeline | `clarifai-datasets` + `clarifai-training-pipelines` (template) |
| "Train ResNet-50 classifier and deploy" | Pipeline | `clarifai-training-pipelines` (template) + `clarifai-inference` |
| "Train YOLOF detector and deploy" | Pipeline | `clarifai-training-pipelines` (template) + `clarifai-inference` |
| "Fine-tune LLM and deploy" | Pipeline | `clarifai-pipelines` (custom) + `clarifai-model-upload` |
| "Train segmentation model" | Pipeline | `clarifai-pipelines` (custom training pipeline) |
| "Train custom model and deploy" | Pipeline | `clarifai-pipelines` (custom training pipeline) + `clarifai-model-upload` |
| "Deploy 2 models, run locally" | Client Script | `clarifai-inference` |
| "Deploy 2 models, give me single endpoint" | Orchestrator | `clarifai-model-upload` |
| "Create MCP server for workflow" | Orchestrator | `clarifai-mcp` |
| "Batch image processing pipeline" | Pipeline | `clarifai-pipelines` |

**Note:** Only ResNet-50 classifier and YOLOF detector have built-in templates. All other training uses custom pipelines.

## Common Use Cases

| User Request | Pattern | Implementation | Building Blocks |
|--------------|---------|----------------|-----------------|
| "Chain classifier with LLM" | Client Script | Sequential Model.predict() | `clarifai-inference` |
| "Single endpoint for my workflow" | Orchestrator | ModelClass calls other models | `clarifai-model-upload` |
| "Process 10K images with ML" | Pipeline | Platform pipeline batch steps | `clarifai-pipelines` |
| "Train and deploy automatically" | Hybrid | Script triggers pipeline | `clarifai-training-pipelines` + `clarifai-inference` |
| "Compare multiple models" | Client Script | Parallel Model.predict() | `clarifai-inference` |
| "RAG with embeddings + LLM" | Client/Orchestrator | Chain embedding + LLM | `clarifai-inference` or `clarifai-model-upload` |
| "Upload dataset, train, serve" | Hybrid | Dataset→Pipeline→Script | `clarifai-datasets` + `clarifai-training-pipelines` + `clarifai-inference` |

## RAG Pattern Example

```python
from clarifai.client import Model
import numpy as np

class RAGOrchestrator(ModelClass):
    """RAG pattern: retrieve then generate."""

    def load_model(self):
        self.embedder = Model(
            url="https://clarifai.com/openai/embed/models/text-embedding-3-large"
        )
        self.llm = Model(
            url="https://clarifai.com/anthropic/completion/models/claude-sonnet-4"
        )
        # Load your vector database / document store
        self.documents = self.load_documents()
        self.doc_embeddings = self.embed_documents()

    def load_documents(self):
        # Load your knowledge base
        return ["doc1 content", "doc2 content", ...]

    def embed_documents(self):
        return [self.embedder.predict(text=doc) for doc in self.documents]

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.predict(text=query)
        # Simple cosine similarity
        similarities = [
            np.dot(query_embedding, doc_emb)
            for doc_emb in self.doc_embeddings
        ]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    @ModelClass.method
    def predict(self, query: str = "") -> str:
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query)

        # Generate response with context
        prompt = f"""Based on these documents:
{chr(10).join(relevant_docs)}

Answer this question: {query}"""

        return self.llm.predict(prompt=prompt)
```

## Best Practices

### Client-Side Scripts
- Keep it simple and linear
- Handle errors at each step
- Log intermediate results
- Consider timeout handling

### Server-Side Orchestrators
- Initialize models in `load_model()`
- Keep `predict()` methods fast
- Handle model failures gracefully
- Consider caching for repeated calls

### Platform Pipelines
- Use for > 30 min operations
- Leverage artifacts for data passing
- Monitor with CLI commands
- Consider checkpointing

## Extended Search

For orchestration examples:
- **Agent examples**: https://github.com/Clarifai/examples/tree/main/agents
- **Workflow examples**: https://github.com/Clarifai/examples/tree/main/workflows
- **RAG examples**: https://github.com/Clarifai/examples/tree/main/RAG

## References

- Decision framework: [references/decision-framework.md](references/decision-framework.md)