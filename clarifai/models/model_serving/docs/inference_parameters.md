## Inference paramters

When making prediction, you may need to change some paramters to adjust the result. Those paramters will be passed through `paramters()` of a request in triton python model.

In order to send it to `**kwargs` of `get_predictions` in `inference.py`, there are 2 ways:
1. You can send any arbitrary parameters via clarifai API.
2. You can define some parameters and they will be visible and adjustable on Clarifai model views.

This document helps you to create your inference parameters that can be visibale and adjustable easily on Clarifai platform. `Again, you can still send any parameters via API but undefined parameters won't appear on Clarifai UI`. The defined parameters will be sent as `json` file when you use `clarifai-upload-model` cli.

### JSON file structure:
The file contains a list of object has 4 fields:
* `path` (str): name of your parameter, it must be valid as python variable
* `field_type` (int): the parameter data type is one of {1,2,3}, it means {boolean, string, number} respectively. `Number` means `int` or `float`
* `default_value`: a default value of the parameter.
* `description` (str): short sentence describes what the parameter does

An example of 3 parameters:
```json
[
  {
    "path": "boolean_var",
    "field_type": 1,
    "default_value": true,
    "description": "a boolean variable"
  },
  {
    "path": "string_var",
    "field_type": 2,
    "default_value": "string_1",
    "description": "a string variable"
  },
  {
    "path": "number_var",
    "field_type": 3,
    "default_value": 9.9,
    "description": "a float number variable"
  }
]
```

### Generate JSON file
1. Manually create the file based on above structure
2. By code:
```python
from clarifai.models.model_serving.model_config.inference_parameter import InferParamManager, InferParam, InferParamType

# 2.1. Fully setup
params = [
  InferParam(
    path="boolean_var",
    field_type=InferParamType.BOOL,
    default_value=True,
    description="a boolean varaiabe"
  ),
  InferParam(
    path="string_var",
    field_type=InferParamType.STRING,
    default_value="string_1",
    description="a string varaiabe"
  ),
  InferParam(
    path="number_var",
    field_type=InferParamType.NUMBER,
    default_value=9.9,
    description="a float number varaiabe"
  ),
]

ipm = InferParamManager(params=params)
ipm.export("your_file.json")

# 2.2. Shorten
# `NOTE`: in this way `description` field will be set as empty aka ""
# *You need to modify* `description` in order to be able to upload the settings to Clarifai
params = dict(boolean_var=True, string_var="string_1", number_var=9.9)
ipm = InferParamManager.from_kwargs(**params)
ipm.export("your_file.json")

```

3. In `test.py`. You can define your paramaters like `2.2. Shorten` in `inference_paramters` attribute of `CustomTestInferenceModel`, the file will be generated when you run the test. Keep in mind to change `description`
