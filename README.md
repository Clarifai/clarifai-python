<h1 align="center">
  <a href="https://www.clarifai.com/"><img alt="Clarifai" title="Clarifai" src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png"></a>
</h1>

<h2 align="center">
Clarifai Python SDK</a>
</h2>



<p align="center">
  <a href="https://clarifaicommunity.slack.com/" target="_blank"> <img src="https://img.shields.io/badge/slack-chat-green.svg?logo=slack&color=4ec528" alt="Slack">
  </a>

</p>




This is the official Python client for interacting with our powerful [API](https://docs.clarifai.com). The Clarifai Python SDK offers a comprehensive set of tools to integrate Clarifai's AI platform to leverage computer vision capabiities like classification , detection ,segementation and natural language capabilities like classification , summarisation , generation , Q&A ,etc into your applications. With just a few lines of code, you can leverage cutting-edge artificial intelligence to unlock valuable insights from visual and textual content.

---
**Website**: [https://www.clarifai.com](https://www.clarifai.com/)

**Demo**: [https://clarifai.com/demo](https://clarifai.com/demo)

**Sign up for a free Account**: [https://clarifai.com/signup](https://clarifai.com/signup)

**Developer Guide**: [https://docs.clarifai.com](https://docs.clarifai.com/)

**Clarifai Community**: [https://clarifai.com/explore](https://clarifai.com/explore)

**Python SDK Docs**: [https://clarifai-python.readthedocs.io/en/latest/index.html](https://clarifai-python.readthedocs.io/en/latest/index.html)


---

## Installation


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



## Getting started
Clarifai uses **Personal Access Tokens(PATs)** to validate requests. You can create and manage PATs under your Clarifai account security settings.

Export your PAT as an environment variable. Then, import and initialize the API Client.


```cmd
export CLARIFAI_PAT={your personal access token}
```

```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.user import User
client = User(user_id="user_id")

# Get all apps
apps = client.list_apps()

# Create app and dataset
app = client.create_app(app_id="demo_app", base_workflow="Universal")
dataset = app.create_dataset(dataset_id="demo_dataset")

# execute data upload to Clarifai app dataset
dataset.upload_dataset(task='visual_segmentation', split="train", dataset_loader='coco_segmentation')

#upload text from csv
dataset.upload_from_csv(csv_path='csv_path', labels=True)

#upload data from folder
dataset.upload_from_folder(folder_path='folder_path', input_type='text', labels=True)
```


### Interacting with Inputs

```python
from clarifai.client.user import User
app = User(user_id="user_id").app(app_id="app_id")
input_obj = app.inputs()

#input upload from url
input_obj.upload_from_url(input_id = 'demo', image_url='https://samples.clarifai.com/metro-north.jpg')

#input upload from filename
input_obj.upload_from_file(input_id = 'demo', video_file='demo.mp4')

#listing inputs
input_obj.list_inputs()

# text upload
input_obj.upload_text(input_id = 'demo', raw_text = 'This is a test')
```


### Interacting with Models

#### Model Predict
```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.model import Model

# Model Predict
model_prediction = Model("https://clarifai.com/anthropic/completion/models/claude-v2").predict_by_bytes(b"Write a tweet on future of AI", "text")

model = Model(user_id="user_id", app_id="app_id", model_id="model_id")
model_prediction = model.predict_by_url(url="url", input_type="image") # Supports image, text, audio, video

# Customizing Model Inference Output
model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                  output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
model_prediction = model.predict_by_filepath(filepath="local_filepath", input_type="text") # Supports image, text, audio, video

model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                    output_config={"sample_ms": 2000}) # Return predictions for specified interval
model_prediction = model.predict_by_url(url="VIDEO_URL", input_type="video")
```
#### Models Listing
```python
# Note: CLARIFAI_PAT must be set as env variable.

# List all model versions
all_model_versions = model.list_versions()

# Go to specific model version
model_v1 = client.app("app_id").model(model_id="model_id", model_version_id="model_version_id")

# List all models in an app
all_models = app.list_models()

# List all models in community filtered by model_type, description
all_llm_community_models = App().list_models(filter_by={"query": "LLM",
                                                        "model_type_id": "text-to-text"}, only_in_app=False)
```

### Interacting with Workflows

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
all_workflow_versions = workflow.list_versions()

# Go to specific workflow version
workflow_v1 = Workflow(workflow_id="workflow_id", workflow_version=dict(id="workflow_version_id"), app_id="app_id", user_id="user_id")

# List all workflow in an app
all_workflow = app.list_workflow()

# List all workflow in community filtered by description
all_face_community_workflows = App().list_workflows(filter_by={"query": "face"}, only_in_app=False) # Get all face related workflows
```


## More Examples
See many more code examples in this [repo](https://github.com/Clarifai/examples).
Also see the official [python SDK docs](https://clarifai-python.readthedocs.io/en/latest/index.html)
