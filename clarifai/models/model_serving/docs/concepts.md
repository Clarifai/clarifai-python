# Overview

Model Serving is a straightforward interface that links user model implementations in Python with a high-performance serving framework (tritonserver). It seamlessly integrates with the Clarifai Platform, allowing users to deploy their models without any prerequisites in the serving framework.

```plaintext

|Model code in Python| ---> |Model Serving + Clarifai Platform| ---> |Served model|

```

# Understanding the concepts

While functioning as an interface, it comes with certain constraints that must be adhered to throughout the process.

## Model repository

First of all, the model repository structure obtained by running

```bash
clarifai create model --type ... --working-dir ...
```

In your working dir:

```bash
├── inference.py
├── clarifai_config.yaml
├── test.py
└── requirements.txt
```
Where:

* [inference.py](): The crucial file where users need to implement their Python code.
* [clarifai_config.yaml](): Contains all necessary configurations for model `test`, `build` and `upload`
* [test.py](): Predefined test cases to evaluate `inference.py`.
* [requirements.text]():  Equivalent to a normal Python project's requirements.txt.

## inference.py
Includes the ModelInference class, inherited from one of the Clarifai Models, providing utility wrapper functions and docstring to ensure that customized models work seamlessly within the platform server. The specific Clairfai Model is determined by the --type argument provided by users in the clarifai create model command.

Sample for `text-to-text` model

```python
class InferenceModel(TextToText):
  """User model inference class."""

  def __init__(self) -> None:
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    # current directory
    self.base_path: Path = os.path.dirname(__file__)

  def predict(self, input_data: list,
              inference_parameters: Dict[str, Union[str, float, int, bool]]) -> list:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[str, float, int, bool]]): your inference parameters

    Returns:
      list of TextOutput

    """

    raise NotImplementedError()

```

Users are required to implement two functions:

* `__init__`: a method to load the model, called once.
* `predict`: a function designed to generate predictions based on the provided inputs and inference parameters. This method includes a docstring inherited from its parent, providing information on input, parameters, and output types. Refer to the docstring to confirm that the outputs of this method adhere to the correct [Clarifai Output Type](../model_config/output.py), as errors may occur otherwise.

When making predictions through the Clarifai API, user inputs are transmitted to input_data as a List of strings for text input or a List of NumPy arrays for RGB image input, where each array has a shape of [W, H, 3]. Additionally, all inference parameters are conveyed through the inference_parameters argument of the predict method.
```plaintext

  list of user inputs e.g.              inference parameters e.g.
  `text-to-text` will be                {'top_k': 5, 'temperature': 0.7, 'do_sample': False, ...}
  ['text', 'test text',]                   |
                    |                      |
                    |                      |
                    |                      |
                    v                      v
def predict(self, input_data:list,  inference_parameters: Dict[str, str | float | int | bool]) -> list:
  ...
  # Predict with input data
  outputs = self.model(input_data, **inference_parameters)
  # Convert to Clarifai Output Type
  return [TextOutput(each) for each in outputs]
                              |
                              |
                              |
                              v
          Outputs are handled by the module -> platform backend to delivery back to user
```

For testing the implementation, it's recommended to execute pytest test.py or directly call the predict method of a ModelInference instance.

## clarifai_config.yaml

`yaml` file for essential configs

```yaml
clarifai_model:
  clarifai_model_id:
  clarifai_user_app_id:
  description:
  inference_parameters: (*)
  labels: (*)
  type: (**)
serving_backend:
  triton: (***)
    max_batch_size:
    image_shape:
```

Explanation:

`clarifai_model`: configs for building/testing/uploading process

* `clarifai_model_id` (str, optional): Model ID on the platform.
* `clarifai_user_app_id` (str, optional): User ID and App ID on the platform seperated by `/` for example `user_1/app_1`.
* `description` (str, optional): Model description.
  > These 3 attributes are used to upload model. If not provided, they can be passed in *upload* command.

* (*) `inference_parameters` (List[Dict], optional): inference parameters for your model prediction method. This attribute is used to *test* and *upload* if provided. Two ways to insert it:

  * Manual: Follow this [doc](./inference_parameters.md)
  * Semi Manual: in *test.py*, init BaseTest with dict of your desired parameters. Learn more about [test.py]()

* (*) `labels` (list): insert manually list of concept names ***required by*** these model types **visual-classifier**, **visual-detector**, **visual-segmenter** and **text-classifier**.

* (**) `type` (str): type of your model, generated when init working dir. ***MUST NOT MODIFY IT***

`serving_backend`: custom config for serving

* `triton`: (optional)

  * `max_batch_size` (int): Maximum number of inputs will go to `predict`. The default value is 1. Since `predict` method receives a list of inputs, if your model supports batch inference, you can set it to a value greater than 1 to leverage high-performance computation on the GPU.

  * `image_shape` (list): Applicable only for image input models. It is a list of the width and height of the input image. The default is [-1, -1], which means it accepts any size.
  > These 2 attributes can be set when initialize using **clarifai create model** command.

## test.py
The file is generated when initializing to test InfercenceModel in inference.py.

This test offers two essential features to enhance the testing and validation process:

**1. Implementation Validation**

Prior to initiating the build or upload processes, users can leverage this feature to thoroughly validate their implementation. This ensures the correctness and readiness of the model for deployment.

The test involves the validation of custom configuration in clarifai_config.yaml:

* Confirming that labels are provided for concept-output models.
* Verifying the format of inference_parameters.

Additionally, it validates the InferenceModel implementation:

* Ensuring the model is loaded correctly.
* Testing predict with dummy inputs.

**2. Inference Parameter Management**

Users can conveniently add or update inference parameters directly in the clarifai_config.yaml file. Additionally, the system performs automatic validation during the inference, ensuring the accuracy and compatibility of these parameters with the model's requirements. The test ensures **you can only use defined inference parameters with appropriate value**

### file structure

```python
class CustomTest(unittest.TestCase):

  def setUp(self) -> None:
    your_infer_parameter = dict()
    self.model = BaseTest(your_infer_parameter)

  def test_default_cases(self):
    self.model.test_with_default_inputs()

```

Explanation:

* `your_infer_parameter = dict()`: define your inference parameters as dict with key is parameter name and value is default value of it. For example, define params for hf text-generation model:

```python
your_infer_parameter = dict(top_p=0.95, temperature=1, return_text=False, prefix="test")
```

* `self.model = BaseTest(your_infer_parameter)` Loaded implemented model and convert inference parameters to *Clarifai inference parameters` format and save it in `clarifai_config.yaml`. See more [doc](./inference_parameters.md)

* `def test_default_cases(self):` Test your model with dummy input. If these dummy input value fails your model, kindly remove or comment out this function

Define new test:

Create a function with 'test' prefix, see `pytest` document to understand how to make a test case.
Call predict by `self.model.predict([list of input data], inference_paramters)`. For instance:

* Text input:

```python
def test_text_input(self):
  text: list = ["Tell me about Clarifai", "How deploy model to Clarifai"]
  outputs = self.model.predict(text, temperature=0.9) # In term of inference parameters for the above example, it will PASSED
  outputs = self.model.predict(text, top_k=10) # And this one will FAILED since `top_k` param is not defined when init self.model

```

* Image input:

```python
def test_image(self):
  image = cv2.imread("path/to/image")
  image = image[:, :, ::-1] # convert to RGB
  out = self.model.predict([image])
```

* MultiModal input:

```python
def test_image_and_text(self):
  image = cv2.imread("path/to/image")
  image = image[:, :, ::-1]
  text = "this is text"
  input = dict(text=text, image=image)
  out = self.model.predict([input])
```
