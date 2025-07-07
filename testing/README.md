# Clarifai Model Test Repository

This repository demonstrates how to create and set up Clarifai local runners for custom model development. It contains a sample text-to-text model that performs string manipulation operations.

## Overview

This repository contains a sample Clarifai model implementation that:
- Inherits from Clarifai's `ModelClass`
- Implements three different model methods: `predict`, `generate`, and `s` (stream processing)
- Demonstrates best practices for local model development and testing
- Shows how to configure and run models locally using Clarifai runners

## Prerequisites

Before setting up this repository, ensure you have:
- Python 3.8+ installed
- pip package manager
- Git
- A Clarifai account (for uploading models, optional for local development)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/srikanthbachala20/model-test-repo.git
cd model-test-repo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install the Clarifai SDK and all required dependencies.

### 3. Verify Installation
Test that the model works locally:
```bash
python 1/model.py
```

You should see output from both the `predict` and `generate` methods.

## Local Development and Testing

### Testing the Model Locally

The simplest way to test your model is to run it directly:
```bash
python 1/model.py
```

### Using Clarifai CLI for Local Testing

The Clarifai CLI provides several commands for local model development:

#### 1. Test Model Locally
```bash
clarifai model test-locally
```

#### 2. Run Model as Local Runner
```bash
clarifai model run-locally
```

This starts a gRPC server that serves your model locally, simulating how it would run in production.

#### 3. Local Development Mode
```bash
clarifai model local-dev
```

This connects your local model to the Clarifai platform for testing while keeping the model running locally.

### Testing Different Model Methods

The sample model provides three different methods:

#### Predict Method
Single input/output prediction:
```python
model = MyModel()
model.load_model()
result = model.predict("Hello", number_of_letters=5)
print(result)  # Output: Hello-AbCdE
```

#### Generate Method
Streaming generation of multiple outputs:
```python
model = MyModel()
model.load_model()
for result in model.generate("Test", number_of_letters=3):
    print(result)  # Generates 10 variations
```

#### Stream Processing Method
Process an iterator of inputs:
```python
model = MyModel()
model.load_model()
inputs = ["input1", "input2", "input3"]
for result in model.s(iter(inputs), number_of_letters=4):
    print(result)
```

## Configuration

### Model Configuration (config.yaml)
The `config.yaml` file contains essential model metadata and resource requirements:
```yaml
model:
  id: "my-model-id"           # Unique identifier for your model
  user_id: "my-user-id"       # Your Clarifai user ID
  app_id: "my-app-id"         # Your Clarifai app ID
  model_type_id: "text-to-text"  # Type of model (text-to-text, image-to-text, etc.)

build_info:
  python_version: "3.12"      # Python version to use

inference_compute_info:
  cpu_limit: "1"              # Maximum CPU cores
  cpu_memory: "1Gi"           # Maximum memory
  cpu_requests: "0.5"         # Minimum CPU cores
  cpu_memory_requests: "512Mi" # Minimum memory
  num_accelerators: 0         # Number of GPUs (0 for CPU-only)
  accelerator_type: ["NVIDIA-*"] # GPU type if using accelerators
  accelerator_memory: "1Gi"   # GPU memory if using accelerators
```

**Important**: Update the `model`, `user_id`, and `app_id` values according to your Clarifai setup. The `inference_compute_info` section is required for local testing and deployment.

### Model Parameters
The model accepts the following parameter:
- `number_of_letters` (int, default=3): Number of random letters to append to the input string

## Project Structure

```
model-test-repo/
├── 1/
│   └── model.py          # Main model implementation
├── config.yaml           # Model configuration
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```

### Model Implementation Details

The `MyModel` class in `1/model.py`:
- Inherits from `clarifai.runners.models.model_class.ModelClass`
- Implements required `load_model()` method
- Uses `@ModelClass.method` decorator for exposed methods
- Uses `Param` class for parameter definitions with defaults and descriptions

## Uploading to Clarifai Platform

Once you've tested your model locally and are satisfied with its performance:

### 1. Configure Authentication
```bash
clarifai login
```

### 2. Upload the Model
```bash
clarifai model upload
```

### 3. Deploy the Model
Follow the Clarifai documentation to deploy your model to a compute cluster.

## Advanced Usage

### Custom Model Development
To create your own model:
1. Modify the `MyModel` class in `1/model.py`
2. Implement your custom logic in the model methods
3. Update the `config.yaml` with your model details
4. Test locally using the provided commands
5. Upload to Clarifai platform

### Adding Dependencies
To add additional Python packages:
1. Add them to `requirements.txt`
2. Run `pip install -r requirements.txt`
3. Import and use them in your model code

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Not Loading**: Check that your model class inherits from `ModelClass` and implements `load_model()`
3. **Configuration Errors**: Verify your `config.yaml` has valid user_id, app_id, and model_id values
4. **"inference_compute_info not found" Error**: Ensure your `config.yaml` includes the complete `inference_compute_info` section as shown in the configuration example
5. **Network Timeouts**: If `clarifai model test-locally` fails due to network issues, try testing the model directly with `python 1/model.py` first

### Getting Help
- Check the [Clarifai Documentation](https://docs.clarifai.com/)
- Use `clarifai --help` for CLI command help
- Use `clarifai model --help` for model-specific commands

## Contributing

This repository serves as a template for Clarifai model development. Feel free to fork and modify it for your own model implementations.