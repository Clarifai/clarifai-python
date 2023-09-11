## Custom Triton Configurations

The commandline triton model repository generation utils do work with default values for the various triton configurations but a few of these config values can be modified to suit different task specific needs.

* For vision models for instance, different input shapes for the `Height (H)` and `Width (W)` are supported and can be set via the commandline too.i.e.
```console
$ clarifai-model-upload-init --model_name <Your model name> \
		--model_type <select model type from available ones> \
		--image_shape "H, W"
		--repo_dir <directory in which to create your model repository>
```
`H` and `W` each have a maximum value of 1024.
`--image_shape` accepts both `"H, W"` and `"[H, W]"` format input.


## Generating the triton model repository without the commandline

The triton model repository can be generated via a python script specifying the same values as required in the commandline. Below is a sample of how the code would be structured with `visual_classifier`.

```python
from clarifai.models.model_serving.model_config import get_model_config, ModelTypes, TritonModelConfig
from clarifai.models.model_serving.pb_model_repository import TritonModelRepository

model_type = ModelTypes.visual_classifier
model_config: TritonModelConfig = get_model_config(model_type).make_triton_model_config(
	model_name="<model_name>",
	model_version="1",
	image_shape=<[H,W]>, # 0 < [H,W] <= 1024
)

triton_repo = TritonModelRepository(model_config)
triton_repo.build_repository("<dir>")
```
