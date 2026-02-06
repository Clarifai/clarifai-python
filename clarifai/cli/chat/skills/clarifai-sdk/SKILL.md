---
name: Clarifai Python SDK
description: Guide for using the Clarifai Python SDK for advanced operations not available in the CLI
commands: []
keywords:
  - sdk
  - python
  - dataset
  - inputs
  - upload
  - search
  - train
  - evaluate
  - workflow
  - predict
  - rag
  - concepts
  - similarity
---

# Clarifai Python SDK

The Clarifai Python SDK provides comprehensive programmatic access to the Clarifai platform with many features that go beyond what's available in the CLI.

## SDK vs CLI Comparison

| Feature | CLI | SDK |
|---------|-----|-----|
| List models/pipelines/deployments | ✅ | ✅ |
| Create/delete apps | ❌ | ✅ |
| Upload inputs (images, text, video) | ❌ | ✅ |
| Create/manage datasets | ❌ | ✅ |
| Train models | ❌ | ✅ |
| Evaluate models | ❌ | ✅ |
| Run predictions | Limited | ✅ Full |
| Semantic search | ❌ | ✅ |
| Manage concepts/annotations | ❌ | ✅ |
| Create workflows | ❌ | ✅ |
| RAG (Retrieval Augmented Generation) | ❌ | ✅ |
| Export models | ❌ | ✅ |

## Installation

```bash
pip install clarifai
```

## Authentication

Set your Personal Access Token (PAT):

```python
import os
os.environ["CLARIFAI_PAT"] = "your_pat_here"

# Or pass directly to constructors
from clarifai.client import User
user = User(user_id="your_user_id", pat="your_pat")
```

## Core Classes

### User - Entry Point

```python
from clarifai.client import User

# Initialize user
user = User(user_id="your_user_id")

# List your apps
apps = list(user.list_apps())

# Create a new app
app = user.create_app(app_id="my-app", base_workflow="Universal")

# List compute clusters
clusters = list(user.list_compute_clusters())

# List runners
runners = list(user.list_runners())

# List all pipelines across your apps
pipelines = list(user.list_pipelines())
```

### App - Application Management

```python
from clarifai.client import App

# Initialize from URL
app = App(url="https://clarifai.com/user_id/app_id")

# Or by IDs
app = App(user_id="user_id", app_id="app_id")

# List resources in the app
models = list(app.list_models())
datasets = list(app.list_datasets())
workflows = list(app.list_workflows())
concepts = list(app.list_concepts())

# Create resources
dataset = app.create_dataset(dataset_id="my-dataset")
model = app.create_model(model_id="my-model", model_type_id="text-classifier")
workflow = app.create_workflow(config_filepath="workflow.yaml")

# Get input count
count = app.get_input_count()
```

## Uploading Data (SDK Only)

### Upload Inputs

```python
from clarifai.client import Inputs

inputs = Inputs(user_id="user_id", app_id="app_id")

# Upload from URL
input_obj = inputs.get_input_from_url(
    input_id="unique-id",
    image_url="https://example.com/image.jpg"
)
inputs.upload_inputs([input_obj])

# Upload from file
input_obj = inputs.get_input_from_file(
    input_id="unique-id",
    image_file="path/to/image.jpg"
)
inputs.upload_inputs([input_obj])

# Upload text
input_obj = inputs.get_text_input(
    input_id="unique-id",
    raw_text="Your text content here"
)
inputs.upload_inputs([input_obj])

# Upload with metadata and concepts
input_obj = inputs.get_input_from_url(
    input_id="unique-id",
    image_url="https://example.com/image.jpg",
    labels=["cat", "animal"],  # Concepts
    metadata={"source": "web", "category": "pets"}
)
inputs.upload_inputs([input_obj])
```

### Upload to Dataset

```python
from clarifai.client import Dataset

dataset = Dataset(user_id="user_id", app_id="app_id", dataset_id="my-dataset")

# Upload from folder
dataset.upload_from_folder(
    folder_path="path/to/images/",
    input_type="image"
)

# Upload from CSV
dataset.upload_from_csv(
    csv_path="data.csv",
    input_type="text"
)

# Create dataset version
dataset.create_version(description="Initial version")
```

## Semantic Search (SDK Only)

```python
from clarifai.client import Search

search = Search(user_id="user_id", app_id="app_id", top_k=10)

# Search by text
results = search.query(query="cute puppies")

# Search by image URL
results = search.query(image_url="https://example.com/query-image.jpg")

# Search with filters
results = search.query(
    query="landscape",
    filters=[{"metadata": {"category": "nature"}}]
)

# Iterate through results
for hit in results:
    print(f"Score: {hit.score}, Input ID: {hit.input.id}")
```

## Model Training (SDK Only)

```python
from clarifai.client import Model

# Get a trainable model
model = Model(url="https://clarifai.com/user_id/app_id/models/my-model")

# Get available training templates
templates = model.list_training_templates()

# Get training parameters
params = model.get_params(template="my-template")

# Train the model
model_version = model.train(
    dataset=dataset,  # Your Dataset object
    template="my-template",
    concepts=["cat", "dog", "bird"]
)

# Check training status
status = model.training_status()
print(f"Status: {status.description}")
```

## Model Evaluation (SDK Only)

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/user_id/app_id/models/my-model")

# Evaluate model on a dataset
eval_result = model.evaluate(
    dataset=dataset,
    eval_id="my-evaluation"
)

# Get evaluation metrics
metrics = eval_result.get_metrics()
```

## Predictions

### Model Predictions

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/clarifai/main/models/general-image-recognition")

# Predict from URL
result = model.predict_by_url(
    url="https://example.com/image.jpg",
    input_type="image"
)

# Predict from file
result = model.predict_by_filepath(
    filepath="path/to/image.jpg",
    input_type="image"
)

# Predict from bytes
with open("image.jpg", "rb") as f:
    result = model.predict_by_bytes(
        input_bytes=f.read(),
        input_type="image"
    )

# Access predictions
for concept in result.outputs[0].data.concepts:
    print(f"{concept.name}: {concept.value:.2f}")
```

### Workflow Predictions

```python
from clarifai.client import Workflow

workflow = Workflow(url="https://clarifai.com/user_id/app_id/workflows/my-workflow")

# Predict
result = workflow.predict_by_url(
    url="https://example.com/image.jpg",
    input_type="image"
)

# Access results from each node
for node_result in result.results:
    print(f"Node: {node_result.id}")
```

## RAG - Retrieval Augmented Generation (SDK Only)

```python
from clarifai.rag import RAG

# Quick setup - creates app, workflow, and prompter
rag = RAG.setup(user_id="your_user_id")

# Or use existing workflow
rag = RAG(workflow_url="https://clarifai.com/user_id/app_id/workflows/rag-workflow")

# Upload documents for knowledge base
rag.upload(
    folder_path="path/to/documents/",
    chunk_size=1024,
    chunk_overlap=200
)

# Or upload from URL
rag.upload(url="https://example.com/document.pdf")

# Chat with your documents
response = rag.chat(messages=[
    {"role": "human", "content": "What is the main topic of the documents?"}
])
print(response)
```

## Export Model (SDK Only)

```python
from clarifai.client import Model

model = Model(url="https://clarifai.com/user_id/app_id/models/my-model")

# Export model
export_path = model.export(export_dir="./exported_model/")
```

## Pipeline Operations

```python
from clarifai.client import Pipeline, PipelineStep

# Get pipeline
pipeline = Pipeline(
    user_id="user_id",
    app_id="app_id",
    pipeline_id="my-pipeline"
)

# Run pipeline
result = pipeline.run(inputs={"text": "Hello world"})

# Create pipeline step
step = PipelineStep(
    pipeline_id="my-pipeline",
    step_id="step-1"
)
```

## Compute Orchestration

```python
from clarifai.client import ComputeCluster, Nodepool, Deployment

# Create compute cluster
user = User(user_id="your_user_id")
cluster = user.create_compute_cluster(
    compute_cluster_id="my-cluster",
    cloud_provider="aws",
    region="us-east-1"
)

# Get nodepool
nodepool = Nodepool(
    user_id="user_id",
    compute_cluster_id="cluster_id",
    nodepool_id="nodepool_id"
)

# Manage deployments
deployment = Deployment(
    user_id="user_id",
    deployment_id="deployment_id"
)
```

## Best Practices

1. **Always use PAT environment variable** for security:
   ```python
   os.environ["CLARIFAI_PAT"] = "your_pat"
   ```

2. **Batch uploads** for large datasets:
   ```python
   inputs.upload_inputs(input_list, batch_size=128)
   ```

3. **Use generators** for large lists:
   ```python
   for model in app.list_models():  # Yields, doesn't load all at once
       process(model)
   ```

4. **Handle rate limits** with retries:
   ```python
   from clarifai.utils.misc import BackoffIterator
   ```

## When to Use SDK vs CLI

**Use CLI for:**
- Quick listing of resources (models, pipelines, deployments)
- Configuration management
- Simple deployment operations
- Interactive exploration

**Use SDK for:**
- Uploading and managing data
- Training and evaluating models
- Running predictions programmatically
- Building RAG applications
- Search and similarity operations
- Automation and integration with other systems
- Complex workflows and pipelines

## More Information

- [Full SDK Documentation](https://docs.clarifai.com/python-sdk)
- [API Reference](https://docs.clarifai.com/api-guide)
- [Examples](https://github.com/Clarifai/clarifai-python/tree/main/examples)
