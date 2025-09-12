# Model Resource Management

This document describes how the Clarifai Python SDK handles model resource management, particularly GPU memory cleanup and resource isolation during secrets changes.

## Problem

When secrets change in production deployments, the traditional approach of reloading models in-process can cause:

- **GPU Memory Duplication**: Creating a new model instance while the old one is still in memory
- **Out-of-Memory Errors**: Insufficient GPU memory for large models during reload
- **Resource Leaks**: Incomplete cleanup of GPU memory, file handles, and other resources
- **Framework Dependencies**: Complex cleanup code that depends on specific ML frameworks

## Solution: Subprocess-Based Model Isolation

The SDK now uses **subprocess-based model creation** to automatically handle GPU memory cleanup without requiring framework-specific code.

### How It Works

1. **Initial Model Loading**: Models are created in isolated subprocesses with clean GPU memory
2. **Secrets Change Detection**: The secrets watcher detects when secrets files change
3. **Subprocess Reload**: A new subprocess creates the updated model with fresh GPU memory
4. **Automatic Cleanup**: The subprocess approach ensures clean GPU memory without manual cleanup
5. **Model Replacement**: The new model replaces the old one after successful creation

### Benefits

- **Zero GPU Memory Duplication**: Each model is created in a fresh subprocess with clean GPU state
- **Framework Agnostic**: No need for PyTorch, TensorFlow, or other framework-specific cleanup code
- **Automatic Resource Management**: Process isolation handles all resource cleanup automatically
- **Production Ready**: Eliminates out-of-memory errors during secrets rotation

## Implementation

### Basic Usage

The subprocess-based model management is automatic and requires no changes to existing model code:

```python
from clarifai.runners.models.model_class import ModelClass

class MyLargeModel(ModelClass):
    def __init__(self):
        super().__init__()
        # Load your model (PyTorch, TensorFlow, etc.)
        self.model = load_my_large_model()

    @ModelClass.method
    def predict(self, text: str) -> str:
        return self.model.generate(text)
```

### Server Configuration

```python
from clarifai.runners.server import ModelServer

# Automatic subprocess-based model management
server = ModelServer("./model")

# To disable secrets reloading for very large models
server = ModelServer("./model", disable_secrets_reload=True)
```

### Command Line Options

```bash
# Default: subprocess-based model management enabled
python server.py --model_path ./model

# Disable for very large models or environments with memory constraints
python server.py --model_path ./model --disable_secrets_reload

# Environment variable
export CLARIFAI_DISABLE_SECRETS_RELOAD=true
```

## Optional: Model Cleanup Method

While the subprocess approach eliminates the need for manual cleanup, models can still implement a `cleanup()` method for backwards compatibility or additional resource management:

```python
class MyModel(ModelClass):
    def cleanup(self):
        """Optional cleanup method - subprocess approach makes this unnecessary."""
        # This method is now optional since subprocess isolation
        # automatically handles GPU memory and resource cleanup
        
        # Still useful for:
        # - Backwards compatibility
        # - Additional resource cleanup beyond GPU memory
        # - Custom cleanup logic
        pass
```

## Production Considerations

### When to Disable Secrets Reloading

Consider disabling automatic secrets reloading in these scenarios:

- **Very Large Models**: Models that take > 5 minutes to load
- **Memory-Constrained Environments**: Systems with limited RAM/GPU memory
- **Stable Secrets**: Environments where secrets rarely change
- **Custom Reload Logic**: Applications with their own reload mechanisms

### Monitoring

The subprocess approach provides clear logging for monitoring:

```
INFO - Detected secrets change, initiating subprocess-based model reload...
INFO - Creating new model instance in isolated subprocess...
INFO - Subprocess: Model created successfully
INFO - Successfully loaded model from subprocess
INFO - Subprocess-based model reload sequence completed successfully
```

## Migration from Framework-Specific Cleanup

If your existing code has framework-specific cleanup, you can simplify it:

### Before (Framework-Specific)
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
```

### After (Subprocess-Based)
```python
def cleanup(self):
    """Cleanup is now handled automatically by subprocess isolation."""
    # Optional: Keep for backwards compatibility or additional cleanup
    # The subprocess approach handles GPU memory automatically
    pass
```

## Troubleshooting

### Common Issues

**Subprocess Creation Fails**
- Check model path is correct
- Ensure sufficient system memory for subprocess
- Verify no import errors in model code

**Model Serialization Errors**
- Some models may not be serializable via pickle
- Consider simplifying model structure or using alternative IPC methods

**Performance Considerations**
- Initial model loading is slower due to subprocess overhead
- Reload time includes subprocess startup time
- Memory usage temporarily increases during reload (subprocess + main process)

### Debugging

Enable detailed logging to troubleshoot subprocess issues:

```python
import logging
logging.getLogger('clarifai').setLevel(logging.DEBUG)
```

## Backward Compatibility

This change is fully backward compatible:

- Existing models continue to work without modification
- The `cleanup()` method is still called but is now optional
- All existing APIs and configuration options remain the same
- Framework-specific cleanup code continues to work but is no longer necessary