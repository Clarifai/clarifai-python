# Model Resource Management Guide

## Secrets Watcher and GPU Memory Management

The Clarifai Python SDK automatically watches for secrets changes and reloads models when secrets are updated. However, this can cause GPU memory issues and subprocess problems for large models.

### The Problem

When secrets change, the server creates a new model instance without properly cleaning up the old one. This can lead to:

- **GPU Memory Duplication**: Multiple model instances loaded simultaneously
- **Out-of-Memory Errors**: Especially with large models (LLMs, vision models)
- **Zombie Subprocesses**: If models use subprocess-based implementations
- **Resource Leaks**: File handles, network connections, etc.

### Solution 1: Implement Model Cleanup

Override the `cleanup()` method in your model class to properly release resources:

```python
from clarifai.runners.models.model_class import ModelClass
import torch

class MyLargeModel(ModelClass):
    def load_model(self):
        # Load your model (e.g., transformer, vision model)
        self.model = torch.load('my_model.pt')
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def cleanup(self):
        """Clean up GPU memory and other resources."""
        if hasattr(self, 'model'):
            # Clear model from GPU memory
            del self.model
            
        # Clear CUDA cache if using PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Terminate any subprocesses
        if hasattr(self, 'worker_process'):
            self.worker_process.terminate()
            self.worker_process.join()
            
        # Close file handles, network connections, etc.
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

    @ModelClass.method
    def predict(self, text: str) -> str:
        return self.model.generate(text)
```

### Solution 2: Disable Automatic Secrets Reloading

If cleanup is not sufficient or your model is too large to reload safely, disable automatic reloading:

```bash
# Command line option
python clarifai/runners/server.py --model_path ./my_model --disable_secrets_reload

# Or set environment variable
export CLARIFAI_DISABLE_SECRETS_RELOAD=1
```

```python
# Programmatically
from clarifai.runners.server import ModelServer
server = ModelServer("./my_model", disable_secrets_reload=True)
```

### Solution 3: Manual Secrets Management

With automatic reloading disabled, you can manually reload secrets when needed:

```python
# Manually trigger reload (only when safe)
server.reload_model_on_secrets_change()
```

### Best Practices

1. **Always implement `cleanup()`** for models using significant resources
2. **Test memory usage** during development to identify leaks
3. **Monitor GPU memory** in production using `nvidia-smi` or similar tools
4. **Use disable_secrets_reload** for very large models or production environments
5. **Handle cleanup errors gracefully** - don't crash if cleanup fails

### Example Cleanup Patterns

#### PyTorch Models
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    torch.cuda.empty_cache()
```

#### TensorFlow Models
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    tf.keras.backend.clear_session()
```

#### Subprocess-based Models
```python
def cleanup(self):
    if hasattr(self, 'worker_processes'):
        for proc in self.worker_processes:
            proc.terminate()
        for proc in self.worker_processes:
            proc.join(timeout=5)  # Wait up to 5 seconds
```

#### File and Network Resources
```python
def cleanup(self):
    # Close file handles
    for attr_name in dir(self):
        attr = getattr(self, attr_name)
        if hasattr(attr, 'close'):
            try:
                attr.close()
            except Exception:
                pass  # Ignore errors during cleanup
```