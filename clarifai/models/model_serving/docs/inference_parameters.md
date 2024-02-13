## Inference paramaters

In order to send it to `inference_parameters` of `predict` in `inference.py`, you can define some parameters and they will be visible and adjustable on Clarifai model view.

This document helps you to understand the concept of inference parameters and how to add it `clarifai_config.yaml`

## Overview

Each paramter has 4 fields:

* `path` (str): name of your parameter, it must be valid as python variable
* `field_type` (int): the parameter data type is one of {1,2,21,3}, it means {boolean, string, encrypted_string, number} respectively. `Number` means `int` or `float`. "Encrypted_string is a string that can be used to store your secrets, like API key. The API will not return the values for this as plaintext.
* `default_value`: a default value of the parameter.
* `description` (str): short sentence describes what the parameter does

An example of 4 type parameters:

```yaml
- path: boolean_var
  default_value: true
  field_type: 1
  description: a boolean variable
- path: string_var
  default_value: "a string"
  field_type: 2
  description: a string variable
- path: number_var
  default_value: 1
  field_type: 3
  description: a number variable
- path: secret_string_var
  default_value: "YOUR_SECRET"
  field_type: 21
  description: a string variable contains secret like API key
```

## Add them to the config file

For example with 4 sample paramaters above.

1. Manually:
Insert them to field inference_parameters of the file, e.g.

```yaml
clarifai_model:
  clarifai_model_id: ''
  clarifai_user_app_id: ''
  description: ''
  inference_parameters:
    - path: boolean_var
      default_value: true
      field_type: 1
      description: a boolean variable
    - path: string_var
      default_value: "a string"
      field_type: 2
      description: a string variable
    - path: number_var
      default_value: 1
      field_type: 3
      description: a number variable
    - path: secret_string_var
      default_value: "YOUR_SECRET"
      field_type: 21
      description: a string variable contains secret like API key
  labels: []
  type: text-to-image
serving_backend:
  triton:
    ...
```

2. Semi: If you have a large number of fields, adding them one by one with specific field types can be exhaustive and unsafe.

To address this, you can define them as a dictionary, where the key is the path and the value is the default value. Then, inject them into `BaseTest()` in `test.py` within your model repository. For example, suppose your test.py looks like this:

```python
class CustomTest(unittest.TestCase):

  def setUp(self) -> None:
    your_infer_parameter = dict()
    self.model = BaseTest(your_infer_parameter)

  def test_default_cases(self):
    self.model.test_with_default_inputs()

```

The `BaseTest` class takes inference parameters as a dict, then validating their values and finally save to the config file
With current samples, the file will turn to

```python
class CustomTest(unittest.TestCase):

  def setUp(self) -> None:
    your_infer_parameter = dict(boolean_var=True, string_var="a string", number_var=1, float_number_var=0.1, _secret_string_var="YOUR_SECRET")
    self.model = BaseTest(your_infer_parameter)

  ...
```

After run `test.py` with pytest. The config file looks like:

```yaml
clarifai_model:
  clarifai_model_id: ''
  clarifai_user_app_id: ''
  description: ''
  inference_parameters:
    - path: boolean_var
      default_value: true
      field_type: 1
      description: boolean_var
    - path: string_var
      default_value: "a string"
      field_type: 2
      description: string_var
    - path: number_var
      default_value: 1
      field_type: 3
      description: number_var
    - path: float_number_var
      default_value: 0.1
      field_type: 3
      description: float_number_var
    - path: _secret_string_var
      default_value: "YOUR_SECRET"
      field_type: 21
      description: _secret_string_var
  labels: []
  type: text-to-image
serving_backend:
  triton:
    ...
```

> [!Note]
> * `description` field is set as `path`
> * For `ENCRYPTED_STRING`, it must be defined with `"_" prefix`
