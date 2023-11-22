## vLLM (text-to-text) Example

These can be used on the fly with minimal or no changes to test deploy vLLM models to the Clarifai platform. See the required files section for each model below.

### Prerequisites:
* weights: Input your local weights or download them from huggingface to `./example/1/weights`.
Example download from huggingface:
```
huggingface-cli download {MODEL_ID} --local-dir ./example/1/weights --local-dir-use-symlinks False --exclude {EXCLUDED FILE TYPES}
```
* requirements.txt: update your requirements.
* inference.py: update LLM() paramters. It is recommended to use `gpu_memory_utilization=0.7`.
