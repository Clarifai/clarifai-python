<h1 align="center">
  <a href="https://www.clarifai.com/"><img alt="Clarifai" title="Clarifai" src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png"></a>
</h1>

<h2 align="center">
Clarifai Python SDK</a>
</h2>



<p align="center">
  <a href="https://discord.gg/M32V7a7a" target="_blank"> <img src="https://img.shields.io/discord/1145701543228735582" alt="Discord">
  </a>
  <a href="https://pypi.org/project/clarifai" target="_blank"> <img src="https://img.shields.io/pypi/dm/clarifai" alt="PyPI - Downloads">
  </a>
</p>




This is the official Python client for interacting with our powerful [API](https://docs.clarifai.com). The Clarifai Python SDK offers a comprehensive set of tools to integrate Clarifai's AI platform to leverage computer vision capabilities like classification , detection ,segementation and natural language capabilities like classification , summarisation , generation , Q&A ,etc into your applications. With just a few lines of code, you can leverage cutting-edge artificial intelligence to unlock valuable insights from visual and textual content.

[Website](https://www.clarifai.com/) | [Demo](https://clarifai.com/demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/python-sdk/api-reference) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)


---



## Table Of Contents

* **[Installation](#rocket-installation)**
* **[Getting Started](#memo-getting-started)**
* **[Interacting with Datasets](#floppy_disk-interacting-with-datasets)**
* **[Interacting with Inputs](#floppy_disk-interacting-with-inputs)**
  * [Input Upload](#input-upload)
  * [Input Listing](#input-listing)
* **[Interacting with Models](#brain-interacting-with-models)**
  * [Model Predict](#model-predict)
  * [Model Training](#model-training)
  * [Model Listing](#models-listing)
* **[Interacting with Workflows](#fire-interacting-with-workflows)**
  * [Workflow Predict](#workflow-predict)
  * [Workflow Listing](#workflows-listing)
  * [Workflow Create](#workflow-create)
  * [Workflow Export](#workflow-export)
* **[More Examples](#pushpin-more-examples)**







## :rocket: Installation


Install from PyPi:

```bash
pip install -U clarifai
```

Install from Source:

```bash
git clone https://github.com/Clarifai/clarifai-python.git
cd clarifai-python
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```



## :memo: Getting started
Clarifai uses **Personal Access Tokens(PATs)** to validate requests. You can create and manage PATs under your Clarifai account security settings.

* 🔗 [Create PAT:](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/) ***Log into Portal &rarr; Profile Icon &rarr; Security Settings &rarr; Create Personal Access Token &rarr; Set the scopes &rarr; Confirm***

* 🔗 [Get User ID:](https://help.clarifai.com/hc/en-us/articles/4408131912727-How-do-I-find-my-user-id-app-id-and-PAT-) ***Log into Portal &rarr; Profile Icon &rarr; Account &rarr; Profile &rarr; User-ID***

Export your PAT as an environment variable. Then, import and initialize the API Client.

Set PAT as environment variable through terminal:

```cmd
export CLARIFAI_PAT={your personal access token}
```

```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.user import User
client = User(user_id="user_id")

# Get all apps
apps_generator = client.list_apps()
apps = list(apps_generator)
```


## :floppy_disk: Interacting with Datasets

Clarifai datasets help in managing datasets used for model training and evaluation. It provides functionalities like creating datasets,uploading datasets and exporting datasets as .zip files.

```python
# Note: CLARIFAI_PAT must be set as env variable.

# Create app and dataset
app = client.create_app(app_id="demo_app", base_workflow="Universal")
dataset = app.create_dataset(dataset_id="demo_dataset")

# execute data upload to Clarifai app dataset
dataset.upload_dataset(task='visual_segmentation', split="train", dataset_loader='coco_segmentation')

#upload text from csv
dataset.upload_from_csv(csv_path='csv_path', input_type='text', csv_type='raw', labels=True)

#upload data from folder
dataset.upload_from_folder(folder_path='folder_path', input_type='text', labels=True)

# Export Dataset
from clarifai.client.dataset import Dataset
# Note: clarifai-data-protobuf.zip is acquired through exporting datasets within the Clarifai Platform.
Dataset().export(save_path='output.zip', local_archive_path='clarifai-data-protobuf.zip')
```



## :floppy_disk: Interacting with Inputs

You can use ***inputs()*** for adding and interacting with input data. Inputs can be uploaded directly from a URL or a file. You can also view input annotations and concepts.

#### Input Upload
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.user import User
app = User(user_id="user_id").app(app_id="app_id")
input_obj = app.inputs()

#input upload from url
input_obj.upload_from_url(input_id = 'demo', image_url='https://samples.clarifai.com/metro-north.jpg')

#input upload from filename
input_obj.upload_from_file(input_id = 'demo', video_file='demo.mp4')

# text upload
input_obj.upload_text(input_id = 'demo', raw_text = 'This is a test')
```

#### Input Listing
```python
#listing inputs
input_generator = input_obj.list_inputs(page_no=1,per_page=10,input_type='image')
inputs_list = list(input_generator)

#listing annotations
annotation_generator = input_obj.list_annotations(batch_input=inputs_list)
annotations_list = list(annotation_generator)

#listing concepts
all_concepts = list(app.list_concepts())
```



## :brain: Interacting with Models

The **Model** Class allows you to perform predictions using Clarifai models. You can specify which model to use by providing the model URL or ID. This gives you flexibility in choosing models. The **App** Class also allows listing of all available Clarifai models for discovery.
For greater control over model predictions, you can pass in an `output_config` to modify the model output as demonstrated below.
#### Model Predict
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.model import Model

"""
Get Model information on details of model(description, usecases..etc) and info on training or
# other inference parameters(eg: temperature, top_k, max_tokens..etc for LLMs)
"""
gpt_4_model = Model("https://clarifai.com/openai/chat-completion/models/GPT-4")
print(gpt_4_model)


# Model Predict
model_prediction = Model("https://clarifai.com/anthropic/completion/models/claude-v2").predict_by_bytes(b"Write a tweet on future of AI", input_type="text")

# Customizing Model Inference Output
model_prediction = gpt_4_model.predict_by_bytes(b"Write a tweet on future of AI", "text", inference_params=dict(temperature=str(0.7), max_tokens=30))
# Return predictions having prediction confidence > 0.98
model_prediction = model.predict_by_filepath(filepath="local_filepath", input_type, output_config={"min_value": 0.98}) # Supports image, text, audio, video

# Supports prediction by url
model_prediction = model.predict_by_url(url="url", input_type) # Supports image, text, audio, video

# Return predictions for specified interval of video
video_input_proto = [input_obj.get_input_from_url("Input_id", video_url=BEER_VIDEO_URL)]
model_prediction = model.predict(video_input_proto, input_type="video", output_config={"sample_ms": 2000})
```
#### Model Training
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.app import App
from clarifai.client.model import Model

"""
Create model with trainable model_type
"""
app = App(user_id="user_id", app_id="app_id")
model = app.create_model(model_id="model_id", model_type_id="visual-classifier")
               (or)
model = Model('url')

"""
List training templates for the model_type
"""
templates = model.list_training_templates()
print(templates)

"""
Get parameters for the model.
"""
params = model.get_params(template='classification_basemodel_v1', save_to='model_params.yaml')

"""
Update the model params yaml and pass it to model.train()
"""
model_version_id = model.train('model_params.yaml')

"""
Training status and saving logs
"""
status = model.training_status(version_id=model_version_id,training_logs=True)
print(status)
```

#### Models Listing
```python
# Note: CLARIFAI_PAT must be set as env variable.

# List all model versions
all_model_versions = list(model.list_versions())

# Go to specific model version
model_v1 = client.app("app_id").model(model_id="model_id", model_version_id="model_version_id")

# List all models in an app
all_models = list(app.list_models())

# List all models in community filtered by model_type, description
all_llm_community_models = App().list_models(filter_by={"query": "LLM",
                                                        "model_type_id": "text-to-text"}, only_in_app=False)
all_llm_community_models = list(all_llm_community_models)
```


## :fire: Interacting with Workflows

Workflows offer a versatile framework for constructing the inference pipeline, simplifying the integration of diverse models. You can use the **Workflow** class to create and manage workflows using **YAML** configuration.
For starting or making quick adjustments to existing Clarifai community workflows using an initial YAML configuration, the SDK provides an export feature.

#### Workflow Predict
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.workflow import Workflow

# Workflow Predict
workflow = Workflow("workflow_url") # Example: https://clarifai.com/clarifai/main/workflows/Face-Sentiment
workflow_prediction = workflow.predict_by_url(url="url", input_type="image") # Supports image, text, audio, video

# Customizing Workflow Inference Output
workflow = Workflow(user_id="user_id", app_id="app_id", workflow_id="workflow_id",
                  output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
workflow_prediction = workflow.predict_by_filepath(filepath="local_filepath", input_type="text") # Supports image, text, audio, video
```

#### Workflows Listing
```python
# Note: CLARIFAI_PAT must be set as env variable.

# List all workflow versions
all_workflow_versions = list(workflow.list_versions())

# Go to specific workflow version
workflow_v1 = Workflow(workflow_id="workflow_id", workflow_version=dict(id="workflow_version_id"), app_id="app_id", user_id="user_id")

# List all workflow in an app
all_workflow = list(app.list_workflow())

# List all workflow in community filtered by description
all_face_community_workflows = App().list_workflows(filter_by={"query": "face"}, only_in_app=False) # Get all face related workflows
all_face_community_workflows = list(all_face_community_workflows)
```
#### Workflow Create
Create a new workflow specified by a yaml config file.
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.app import App
app = App(app_id="app_id", user_id="user_id")
workflow = app.create_workflow(config_filepath="config.yml")
```

#### Workflow Export
Export an existing workflow from Clarifai as a local yaml file.
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.workflow import Workflow
workflow = Workflow("https://clarifai.com/clarifai/main/workflows/Demographics")
workflow.export('demographics_workflow.yml')
```


## :pushpin: More Examples

See many more code examples in this [repo](https://github.com/Clarifai/examples).
Also see the official [Python SDK docs](https://clarifai-python.readthedocs.io/en/latest/index.html)
