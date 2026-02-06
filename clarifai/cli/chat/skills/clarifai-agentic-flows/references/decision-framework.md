# Agentic Flow Decision Framework

## When This Skill Triggers

Route to this skill when request contains:
- Multiple operations: "create dataset **and** train model"
- Sequential language: "train **then** deploy **then** inference code"
- Multi-model requests: "deploy 3 models **and** generate script"
- Workflow mentions: "complete ML workflow", "orchestrate", "chain"
- Inference flows: "prediction flow", "chain predictions", "model A then model B"
- Hosted inference: "single endpoint", "host on Clarifai", "one API call"

**Does NOT trigger** for single operations → use specific building block skill.

---

## Three Decision Dimensions

| Dimension | Client Script | Orchestrator | Pipeline |
|-----------|--------------|--------------|----------|
| **Duration** | < 5 min | < 30 min | > 30 min |
| **Scale** | < 1k items | 1k-10k items | > 10k items |
| **Location** | Local machine | Clarifai cloud | Clarifai cloud |

---

## Quick Decision Tree

```
What's the longest step?
├── > 30 minutes (training, large batch)
│   └── Use Platform Pipeline
│       └── Are other steps quick?
│           ├── YES → Hybrid (Pipeline + Script/Orchestrator)
│           └── NO → Full Pipeline
│
└── < 30 minutes (inference, small batch)
    └── Where should it run?
        ├── User's machine → Client-Side Script
        └── Hosted endpoint → Server-Side Orchestrator
```

---

## Pattern Trade-offs

### Client-Side Script
- ✅ Full control and flexibility
- ✅ Easy to debug locally
- ✅ No deployment needed
- ❌ User must run and maintain
- ❌ No automatic scaling

### Server-Side Orchestrator
- ✅ Single API call from client
- ✅ Hosted on Clarifai
- ✅ No local maintenance
- ❌ Must be deployed as model
- ❌ Less flexible than client script

### Platform Pipeline
- ✅ Container isolation
- ✅ GPU support
- ✅ Long-running stability
- ❌ Argo complexity
- ❌ Only for batch/long operations

---

## Request → Pattern Routing

| User Request | Pattern | Building Block Skills |
|--------------|---------|----------------------|
| "Chain Q&A and sentiment locally" | Client Script | `clarifai-inference` |
| "Single endpoint that chains models" | Orchestrator | `clarifai-model-upload` + `clarifai-inference` |
| "Upload 50k images and train" | Pipeline | `clarifai-datasets` + `clarifai-training-pipelines` |
| "Train then deploy then inference code" | Hybrid | `clarifai-training-pipelines` + `clarifai-model-upload` + `clarifai-inference` |
| "Process 500 docs through 3 models" | Script/Orchestrator | `clarifai-inference` |
| "Deploy 2 models, run locally" | Client Script | `clarifai-inference` |
| "Deploy 2 models, give me single endpoint" | Orchestrator | `clarifai-model-upload` |
| "Q&A then sentiment, host on Clarifai" | Orchestrator | `clarifai-model-upload` |
| "Create MCP server for workflow" | Orchestrator | `clarifai-mcp` |
| "Batch image processing pipeline" | Pipeline | `clarifai-pipelines` |

---

## Building Block Skills Reference

| Building Block | Use For | Key Patterns |
|----------------|---------|--------------|
| `clarifai-datasets` | Upload data with annotations | `ClarifaiDataLoader`, Feature classes |
| `clarifai-training-pipelines` | Train classifiers/detectors | ResNet, YOLOF templates |
| `clarifai-model-upload` | Deploy custom models | `ModelClass`, config.yaml |
| `clarifai-mcp` | Tool-calling servers | `MCPModelClass`, FastMCP |
| `clarifai-inference` | Model discovery & predict | `Model.predict()`, OpenAI API |
| `clarifai-pipelines` | Custom batch workflows | Artifacts API, DAG steps |

---

## Decision Questions

1. **What's the longest operation?**
   - > 30 min → Must use pipeline for that step
   - < 5 min → Script or orchestrator

2. **Where should orchestration run?**
   - Local control → Client script
   - Hosted endpoint → Orchestrator model
   - No preference → Either works

3. **What's the scale?**
   - < 1k items → Any pattern
   - > 10k items → Pipeline (required)

4. **Do steps need GPU?**
   - Yes → Pipeline
   - No → Any pattern

5. **How often will this run?**
   - Once/rarely → Script (simpler)
   - Repeatedly → Orchestrator or pipeline

---

## Example Decisions

**"Chain Q&A and sentiment for 100 questions"**
- 100 items × 2 models = 200 API calls, < 5 min total
- Want local control
- → **Client Script** using `clarifai-inference`

**"Same flow but as production API"**
- Same operations but need hosted endpoint
- → **Server-Side Orchestrator** using `clarifai-model-upload`

**"Process 50k documents through 3 models"**
- 50k items = large scale
- Likely > 30 min total
- → **Platform Pipeline** using `clarifai-pipelines`

**"Train model then deploy and test"**
- Training > 30 min → Pipeline
- Deploy < 5 min → Script
- → **Hybrid** using `clarifai-training-pipelines` + `clarifai-inference`

**"Upload dataset, train classifier, generate inference code"**
- Upload: quick → script using `clarifai-datasets`
- Train: long → pipeline using `clarifai-training-pipelines`
- Inference code: quick → script using `clarifai-inference`
- → **Hybrid**