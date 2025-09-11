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
        """Framework-agnostic GPU memory cleanup - customize based on your model's needs."""
        # Force garbage collection to release Python object references
        import gc
        gc.collect()
        
        # Framework-specific cleanup should be implemented in your model class
        # based on whichever ML framework you're actually using
        # Examples (only include what your model actually uses):
        
        # PyTorch (only if your model uses it)
        # try:
        #     import torch
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        # except ImportError:
        #     pass
        
        # TensorFlow (only if your model uses it)
        # try:
        #     import tensorflow as tf
        #     tf.keras.backend.clear_session()
        # except ImportError:
        #     pass

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

1. **Only use frameworks your model actually imports** - don't add new dependencies just for cleanup
2. **Use Python's garbage collector** (`gc.collect()`) as a framework-agnostic baseline
3. **Test memory usage** during development to identify leaks
4. **Monitor GPU memory** in production using `nvidia-smi` or similar tools
5. **Use disable_secrets_reload** for very large models or production environments
6. **Handle cleanup errors gracefully** - don't crash if cleanup fails
7. **Keep the package framework-agnostic** - don't assume specific ML frameworks are installed

### Example Cleanup Patterns

#### Framework-Specific Examples

**PyTorch Models:**
```python
def cleanup(self):
    # Clear model reference
    if hasattr(self, 'model'):
        del self.model
    
    # PyTorch-specific cleanup (only if your model actually uses PyTorch) 
    import torch  # Your model should already import this
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
```

**TensorFlow Models:**
```python  
def cleanup(self):
    # Clear model reference  
    if hasattr(self, 'model'):
        del self.model
    
    # TensorFlow-specific cleanup (only if your model actually uses TensorFlow)
    import tensorflow as tf  # Your model should already import this
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    import gc
    gc.collect()
```

**Generic/Framework-Agnostic Models:**
```python
def cleanup(self):
    # Clear model reference - works with any framework
    if hasattr(self, 'model'):
        del self.model
    
    # Clear other model components
    if hasattr(self, 'tokenizer'):
        del self.tokenizer
        
    # Terminate any child processes
    if hasattr(self, 'worker_processes'):
        for process in self.worker_processes:
            process.terminate()
            process.join()
    
    # Close file handles, network connections, etc.
    if hasattr(self, 'file_handles'):
        for handle in self.file_handles:
            handle.close()
    
    # Force Python garbage collection to release memory
    import gc
    gc.collect()
    
    # Note: For GPU memory cleanup, use the specific framework 
    # your model actually imports (PyTorch, TensorFlow, etc.)
    # Don't add framework dependencies just for cleanup
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

1. **Only use frameworks your model actually imports** - don't add new dependencies for cleanup
2. **Provide fallback behavior** when frameworks aren't available  
3. **Focus on resource cleanup** rather than framework-specific optimizations
4. **Test with and without** your chosen ML framework installed

This ensures your models work regardless of which ML frameworks users have installed.

### Framework-Agnostic GPU Memory Management

For GPU memory cleanup without depending on specific ML frameworks, use these approaches:

#### 1. Python Garbage Collection (Always Safe)
```python
def cleanup(self):
    # Clear object references
    if hasattr(self, 'model'):
        del self.model
    
    # Force garbage collection - works with any framework
    import gc
    gc.collect()
```

#### 2. Use Only Your Model's Existing Dependencies
```python
def cleanup(self):
    # Clear model reference
    if hasattr(self, 'model'):
        del self.model
    
    # Only use frameworks your model already imports
    # If your model.py imports torch, then use torch cleanup
    # If your model.py imports tensorflow, then use tensorflow cleanup
    # Don't add new framework dependencies just for cleanup
    
    # Your model should handle its own framework-specific cleanup
    # based on what it actually uses
```

#### 3. Optional GPU Memory Monitoring (No Framework Dependencies)
```python
def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Optional: Monitor GPU memory without ML framework dependencies
    # (only if nvidia-ml-py is available in your environment)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory after cleanup: {meminfo.used/1024**2:.1f}MB used")
    except (ImportError, Exception):
        pass  # nvidia-ml-py not available or no GPU
```

#### 4. Process-Level Isolation (Ultimate Fallback)
For very large models where memory cleanup is difficult, consider process-level isolation:
```python
# Run model in separate process
# When secrets change, restart the entire process
# This ensures complete memory cleanup without framework dependencies
```