# Clarifai Model Serving

## Overview

Model Serving is a part of user journey at Clarifai offers a user-friendly interface for deploying your local model into production with Clarifai, featuring:

* A convenient command-line interface (CLI)
* Easy implementation and testing in Python
* No need for MLops expertise.

## Quickstart Guide

Quick example for deploying a `text-to-text` model

### Initialize a Clarifai model repository

Suppose your working directory name is `your_model_dir`. Then run

```bash
$ clarifai create model --type text-to-text --working-dir your_model_dir
$ cd your_model_dir
```

In `your_model_dir` folder you will see essential files for deployment process

```bash
your_model_dir
├── clarifai_config.yaml
├── inference.py
├── test.py
└── requirements.txt
```

### Implementation

Write your code in class `InferenceModel` which is an interface between your model and Clarifai server in `inference.py`, there are 2 functions you must implement:

* `__init__`: load your model checkpoint once.
* `predict`: make prediction, called everytime when you make request from API.

For example, a complete implementation of a hf text-generation model

```python
import os
from typing import Dict, Union
from clarifai.models.model_serving.model_config import *

import torch
from transformers import AutoTokenizer
import transformers

class InferenceModel(TextToText):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path = os.path.dirname(__file__)
    # where you save hf checkpoint in your working dir e.i. `your_model_dir`
    model_path = os.path.join(self.base_path, "checkpoint")
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int]]) -> list:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of TextOutput

    """
    output_sequences = self.pipeline(
        input_data,
        eos_token_id=self.tokenizer.eos_token_id,
        **inference_parameters)

    # wrap outputs in Clarifai defined output
    return [TextOutput(each[0]) for each in output_sequences]
```

Update dependencies in `requirements.txt`

```
clarifai
torch=2.1.1
transformers==4.36.2
accelerate==0.26.1
```

### Test (optional)

> NOTE: Running `test` is also involved in `build` and `upload` command.

Test and play with your implementation by executing `test.py`.

Install pytest

```bash
$ pip install pytest
```

Execute test

```bash
$ pytest test.py
```

### Build

Prepare for deployment step. Run:

```bash
$ clarifai build model
```

You will obtain `*.clarifai` file, it's simply a zip having all nessecary files in it to get your model work on Clarifai platform.

`NOTE`: you need to upload your built file to cloud storage to get direct download `url` for next step

### Deployment

Login to Clarifai

```bash
$ clarifai login
Get your PAT from https://clarifai.com/settings/security and pass it here: <insert your pat here>
```

Upload

```bash
# upload built file directly
$ clarifai upload model <your-working-dir> --user-app <your_user_id>/<your_app_id> --id <your_model_id>
# or using direct download url of cloud storage
$ clarifai upload model --url <url> --user-app <your_user_id>/<your_app_id> --id <your_model_id>
```

## Learn More

* [Detail Instruction](./docs/concepts.md)
* [Examples](https://github.com/Clarifai/examples/tree/main/model_upload)
* [Initialize from example](./docs/cli.md)
* [CLI usage](./docs/cli.md)
* [Inference parameters](./docs/inference_parameters.md)
* [Model Types](./docs/model_types.md)
* [Dependencies](./docs/dependencies.md)
