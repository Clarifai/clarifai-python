## Inference paramaters

When making prediction, you may need to change some paramaters to adjust the result. Those paramaters will be passed through `paramaters()` of a request in triton python model.

In order to send it to `**kwargs` of `get_predictions` in `inference.py`, you can define some parameters and they will be visible and adjustable on Clarifai model view.

This document helps you to create your inference parameters that can be visibale and adjustable easily on Clarifai platform. The defined parameters will be sent as `json` file when you use `clarifai-upload-model` cli.

### JSON file structure:
The file contains a list of object has 4 fields:
* `path` (str): name of your parameter, it must be valid as python variable
* `field_type` (int): the parameter data type is one of {1,2,21,3}, it means {boolean, string, encrypted_string, number} respectively. `Number` means `int` or `float`. "Encrypted_string is a string that can be used to store your secrets, like API key. The API will not return the values for this as plaintext.
* `default_value`: a default value of the parameter.
* `description` (str): short sentence describes what the parameter does

An example of 4 parameters:
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
  },
  {
    "path": "secret_string_var",
    "field_type": 21,
    "default_value": "API_KEY",
    "description": "a string variable contains secret like API key"
  },
]
```

### Generate JSON file
1. Manually create the file based on above structure
2. By code:

#### 2.1. Fully setup
```python
from clarifai.models.model_serving.model_config.inference_parameter import InferParamManager, InferParam, InferParamType

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
  InferParam(
    path=secret_string_var",
    field_type=InferParamType.ENCRYPTED_STRING,
    default_value="API_KEY",
    description="a string variable contains secret like API key"
  ),
]

ipm = InferParamManager(params=params)
ipm.export("your_file.json")
```

##### 2.2. Shorten
`NOTE`: in this way `description` field will be set as empty aka "".
*You need to modify* `description` in order to be able to upload the settings to Clarifai.

`NOTE`: in this way `ENCRYPTED_STRING` type must be defined with "_" prefix

```python
params = dict(boolean_var=True, string_var="string_1", number_var=9.9, _secret_string_var="YOUR_KEY")
ipm = InferParamManager.from_kwargs(**params)
ipm.export("your_file.json")

```

3. In `test.py`. You can define your paramaters like `2.2. Shorten` in `inference_parameters` attribute of `CustomTestInferenceModel`, the file will be generated when you run the test. Keep in mind to change `description`

### Usage
Your defined parameters will be passed through `kwargs` of `InferenceModel.get_predictions` method
in `inference.py`
```python
class InferenceModel:
  def __init__():
    # initialization
    self.model = YourModel()

  @some_wrapper_function
  def get_predictions(self, input_data, **kwargs):
    # `kwargs` contains your inference parameters

    # get a value from kwargs
    number_var = kwargs.get("number_var", 9.9)

    # pass everything to a function
    output = self.model.predict(input_data, **kwargs)

    return SomeOutputType(output)

```

in `test.py`
```python
class CustomTestInferenceModel:
  inference_parameters = "" # input a path of json file from `2.1` or a dict from `2.2`

  ...

  def test_something(self):
    input = ...
    output = self.triton_get_predictions(input, number_var=1, string_var="test", _secret="KEY")
    self.assert(...)
```
