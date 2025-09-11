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

class MyLargeModel(ModelClass):
    def load_model(self):
        # Load your model using any framework
        # This is just an example - use your preferred ML framework
        self.model = self._load_my_model()  # Your custom loading logic

    def cleanup(self):
        """Clean up GPU memory and other resources.
        
        This method should be customized based on your specific model and framework.
        """
        # Step 1: Clear model from memory
        if hasattr(self, 'model'):
            del self.model
            
        # Step 2: Framework-specific GPU memory cleanup (if applicable)
        self._cleanup_gpu_memory()
            
        # Step 3: Terminate any subprocesses
        if hasattr(self, 'worker_process'):
            self.worker_process.terminate()
            self.worker_process.join()
            
        # Step 4: Close file handles, network connections, etc.
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

    def _cleanup_gpu_memory(self):
        """Framework-specific GPU memory cleanup - override as needed."""
        # PyTorch example (only if torch is installed)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch not available
            
        # TensorFlow example (only if tensorflow is installed)
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass  # tensorflow not available

    @ModelClass.method
    def predict(self, text: str) -> str:
        return self.model.generate(text)  # Your prediction logic
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
2. **Use conditional imports** for framework-specific cleanup to maintain compatibility
3. **Test memory usage** during development to identify leaks
4. **Monitor GPU memory** in production using `nvidia-smi` or similar tools
5. **Use disable_secrets_reload** for very large models or production environments
6. **Handle cleanup errors gracefully** - don't crash if cleanup fails
7. **Keep the package framework-agnostic** - don't assume specific ML frameworks are installed

### Example Cleanup Patterns

#### Framework-Specific Examples

**PyTorch Models (if torch is available):**
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    
    # Only use torch if it's installed
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # torch not available, skip GPU cleanup
```

**TensorFlow Models (if tensorflow is available):**
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    
    # Only use tensorflow if it's installed
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass  # tensorflow not available, skip cleanup
```

**Generic Framework-Agnostic Cleanup:**
```python
def cleanup(self):
    # Always safe - works with any framework
    if hasattr(self, 'model'):
        del self.model
    
    # Clean up other resources without framework dependencies
    if hasattr(self, 'tokenizer'):
        del self.tokenizer
        
    # Force garbage collection
    import gc
    gc.collect()
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
    # Close file handles - framework agnostic
    for attr_name in dir(self):
        attr = getattr(self, attr_name)
        if hasattr(attr, 'close'):
            try:
                attr.close()
            except Exception:
                pass  # Ignore errors during cleanup
                
    # Clean up temporary files
    if hasattr(self, 'temp_files'):
        import os
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass
```

### Framework Compatibility

The cleanup mechanism is designed to be **framework-agnostic**. The clarifai-python package does not require any specific ML framework as a dependency. When implementing cleanup methods:

1. **Use conditional imports** for framework-specific cleanup (e.g., `try: import torch` with `except ImportError`)
2. **Provide fallback behavior** when frameworks aren't available
3. **Focus on resource cleanup** rather than framework-specific optimizations
4. **Test with and without** your chosen ML framework installed

This ensures your models work regardless of which ML frameworks users have installed.