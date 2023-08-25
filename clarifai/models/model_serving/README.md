## Clarifai Model Serving: Deploy Your Machine Learning Models to Clarifai.

Build and easily deploy machine learning models to Clarifai for inference using the [Nvidia Triton Inference Server](https://github.com/triton-inference-server/server).

## QuickStart Guide: Build a deployment ready model.

A step by step guide to building your own triton inference model and deploying it into a Clarifai app.

1. Generate a triton model repository via commandline.
```console
$ clarifai-model-upload-init --model_name <Your model name> \
		--model_type <select model type from available ones> \
		--repo_dir <directory in which to create your model repository>
```
2. Edit the `requirements.txt` file with dependencies needed to run inference on your model and the `labels.txt` (if available in dir) with the labels your model is to predict.
3. Add your model loading and inference code inside `inference.py` script of the generated model repository under the `setup()` and `predict()` functions respectively. Refer to  The [Inference Script section]() for a description of this file.
4. Generate a zip of your triton model for deployment via commandline.
```console
$ clarifai-triton-zip --triton_model_repository <path to triton model repository to be compressed> \
    --zipfile_name <name of the triton model zip> (Recommended to use 	  <model_name>_<model-type> convention for naming)
```
5. Upload the generated zip to a public file storage service to get a URL to the zip. This URL must be publicly accessible and downloadable as it's necessary for the last step: uploading the model to a Clarifai app.
6. Set your Clarifai auth credentials as environment variables.
```console
$ export CLARIFAI_USER_ID=<your clarifai user_id>
$ export CLARIFAI_APP_ID=<your clarifai app_id>
$ export CLARIFAI_PAT=<your clarifai PAT>
```
7. Upload your model to Clarifai. Please ensure that your configuration field maps adhere to [this](https://github.com/Clarifai/clarifai-python-utils/blob/main/clarifai/models/model_serving/model_config/deploy.py)
```console
$ clarifai-upload-model --url <URL to your model zip. Your zip file name is expected to have "zipfile_name" format (in clarifai-triton-zip), if not you need to specify your model_id and model_type> \
    --model_id <Your model ID on the platform> \
    --model_type <Clarifai model types> \
    --desc <A description of your model>
```

* Finally, navigate to your Clarifai app models and check that the deployed model appears. Click it on the model name to go the model versions table to track the status of the model deployment.

## Triton Model Repository

    <model_name>/
    ├── config.pbtx
    ├── requirements.txt
    ├── labels.txt (If applicable for given model-type)
    ├── triton_conda.yaml
    |
    └── 1/
        ├── __init__.py
        ├── inference.py
        └── model.py

A generated triton model repository looks as illustrated in the directory tree above. Any additional files such as model checkpoints and folders needed at inference time must all be placed under the `1/` directory.

- File Descriptions

| Filename | Description & Use |
| --- | --- |
| `config.pbtxt` | Contains the triton model configuration used by the triton inference server to guide inference requests processing. |
| `requirements.txt` | Contains dependencies needed by a user model to successfully make predictions.|
| `labels.txt` | Contains labels listed one per line, a model is trained to predict. The order of labels should match the model predicted class indexes. |
| `triton_conda.yaml` | Contains dependencies available in pre-configured execution environment. |
| `1/inference.py` | The inference script where users write their inference code. |
| `1/model.py` | The triton python backend model file run to serve inference requests. |

## Inference Script

An `inference.py` script with template code is generated during the triton model repository generation.
**This is the script where users write their inference code**.
This script is composed of a single class that contains a default init method and the `get_predictions()` method whose names mustn't be changed.

```python
"""User model inference script."""

import os
from pathlib import Path
from typing import Callable

class InferenceModel:
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    self.base_path: Path = os.path.dirname(__file__)
    ## sample model loading code:
    #self.checkpoint_path: Path = os.path.join(self.base_path, "your checkpoint filename/path")
    #self.model: Callable = <load_your_model_here from checkpoint or folder>

  #Add relevant model type decorator to the method below (see docs/model_types for ref.)
  def get_predictions(self, input_data):
    """
    Main model inference method.

    Args:
    -----
      input_data: A single input data item to predict on.
        Input data can be an image or text, etc depending on the model type.

    Returns:
    --------
      One of the clarifai.models.model_serving.models.output types. Refer to the README/docs
    """
    # Delete/Comment out line below and add your inference code
    raise NotImplementedError()
```

- `__init__()` used for one-time loading of inference time artifacts such as models, tokenizers, etc that are frequently called during inference to improve inference speed.

- `get_predictions()` takes an input data item whose type depends on the task the model solves, & returns predictions for an input data item.

`get_predictions()` should return any of the output types defined under [output](docs/output.md) and the predict function MUST be decorated with a task corresponding [model type decorator](docs/model_types.md). The model type decorators are responsible for passing input request batches for prediction and formatting the resultant predictions into triton inference responses.

Additional methods can be added to this script's `Infer` class by the user as deemed necessary for their model inference provided they are invoked inside `get_predictions()` if used at inference time.

## Next steps

- [Model types docs](docs/model_types.md)
- [Model Output types docs](docs/output.md)
- [Dependencies](docs/dependencies.md)
- [Examples](examples/)
- [Custom Configs](docs/custom_config.md/)

## Prerequisites

* To test infer with your built triton inference model on your computer/laptop, you need to have the [Triton Inference Server]((https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)) installed.

* For deployment to Clarifai, you need a [Clarifai account](https://clarifai.com/signup).

## Notes

* Ability to run inference tests locally with built triton models to be added in later release.
