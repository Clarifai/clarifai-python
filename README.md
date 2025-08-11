<h1 align="center">
  <a href="https://www.clarifai.com/"><img alt="Clarifai" title="Clarifai" src="https://github.com/user-attachments/assets/623b883b-7fe5-4b95-bbfa-8691f5779af4"></a>
</h1>


<h2 align="center">
Clarifai Python SDK
</h2>



<p align="center">
  <a href="https://discord.gg/M32V7a7a" target="_blank"> <img src="https://img.shields.io/discord/1145701543228735582" alt="Discord">
  </a>
  <a href="https://pypi.org/project/clarifai" target="_blank"> <img src="https://img.shields.io/pypi/dm/clarifai" alt="PyPI - Downloads">
  </a>
  <a href="https://img.shields.io/pypi/pyversions/clarifai" target="_blank"> <img src="https://img.shields.io/pypi/pyversions/clarifai" alt="PyPI - Versions">
  </a>
</p>




This is the official Python client for interacting with our powerful [API](https://docs.clarifai.com). The Clarifai Python SDK offers a comprehensive set of tools to integrate Clarifai's AI platform to leverage computer vision capabilities like classification , detection ,segementation and natural language capabilities like classification , summarisation , generation , Q&A ,etc into your applications. With just a few lines of code, you can leverage cutting-edge artificial intelligence to unlock valuable insights from visual and textual content.

[Website](https://www.clarifai.com/) | [Schedule Demo](https://www.clarifai.com/company/schedule-demo) | [Signup for a Free Account](https://clarifai.com/signup) | [API Docs](https://docs.clarifai.com/) | [Clarifai Community](https://clarifai.com/explore) | [Python SDK Docs](https://docs.clarifai.com/resources/api-references/python) | [Examples](https://github.com/Clarifai/examples) | [Colab Notebooks](https://github.com/Clarifai/colab-notebooks) | [Discord](https://discord.gg/XAPE3Vtg)

Give the repo a star ⭐
---



## Table Of Contents

* **[Installation](#rocket-installation)**
* **[Getting Started](#memo-getting-started)**
* **[Compute Orchestration](#rocket-compute-orchestration)**
  * [Cluster Operations](#cluster-operations)
  * [Nodepool Operations](#nodepool-operations)
  * [Depolyment Operations](#deployment-operations)
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
* **[Search](#mag-search)**
  * [Smart Image Search](#smart-image-search)
  * [Smart Text Search](#smart-text-search)
  * [Filters](#filters)
  * [Pagination](#pagination)
* **[Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)**
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
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Linting

For developers, use the precommit hook `.pre-commit-config.yaml` to automate linting.

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Now every time you run `git commit` your code will be automatically linted and won't commit if it fails.

You can also manually trigger linting using:

```bash
pre-commit run --all-files
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

OR <br>

PAT can be passed as constructor argument

```python
from clarifai.client.user import User
client = User(user_id="user_id", pat="your personal access token")
```


## :rocket: Compute Orchestration

Clarifai’s Compute Orchestration offers a streamlined solution for managing the infrastructure required for training, deploying, and scaling machine learning models and workflows.

This flexible system supports any compute instance — across various hardware providers and deployment methods — and provides automatic scaling to match workload demands. [More Details](https://www.clarifai.com/products/compute-orchestration)

#### Cluster Operations
```python
from clarifai.client.user import User
client = User(user_id="user_id",base_url="https://api.clarifai.com")

# Create a new compute cluster
compute_cluster = client.create_compute_cluster(compute_cluster_id="demo-id",config_filepath="computer_cluster_config.yaml")

# List Clusters
all_compute_clusters = list(client.list_compute_clusters())
print(all_compute_clusters)
```
##### [Example Cluster Config](https://github.com/Clarifai/examples/blob/main/ComputeOrchestration/configs/compute_cluster_config.yaml)



#### Nodepool Operations
```python
from clarifai.client.compute_cluster import ComputeCluster

# Initialize the ComputeCluster instance
compute_cluster = ComputeCluster(user_id="user_id",compute_cluster_id="demo-id")

# Create a new nodepool
nodepool = compute_cluster.create_nodepool(nodepool_id="demo-nodepool-id",config_filepath="nodepool_config.yaml")

#Get a nodepool
nodepool = compute_cluster.nodepool(nodepool_id="demo-nodepool-id")
print(nodepool)

# List nodepools
all_nodepools = list(compute_cluster.list_nodepools())
print(all_nodepools)
```
##### [Example Nodepool config](https://github.com/Clarifai/examples/blob/main/ComputeOrchestration/configs/nodepool_config.yaml)

#### Deployment Operations
```python
from clarifai.client.nodepool import Nodepool

# Initialize the Nodepool instance
nodepool = Nodepool(user_id="user_id",nodepool_id="demo-nodepool-id")

# Create a new deployment
deployment = nodepool.create_deployment(deployment_id="demo-deployment-id",config_filepath="deployment_config.yaml")

#Get a deployment
deployment = nodepool.deployment(nodepool_id="demo-deployment-id")
print(deployment)

# List deployments
all_deployments = list(nodepool.list_deployments())
print(all_deployments)

```
##### [Example Deployment config](https://github.com/Clarifai/examples/blob/main/ComputeOrchestration/configs/deployment_config.yaml)

#### Compute Orchestration CLI Operations
Refer Here: https://github.com/Clarifai/clarifai-python/tree/master/clarifai/cli


## :floppy_disk: Interacting with Datasets

Clarifai datasets help in managing datasets used for model training and evaluation. It provides functionalities like creating datasets,uploading datasets, retrying failed uploads from logs and exporting datasets as .zip files.

```python
# Note: CLARIFAI_PAT must be set as env variable.

# Create app and dataset
app = client.create_app(app_id="demo_app", base_workflow="Universal")
dataset = app.create_dataset(dataset_id="demo_dataset")

# execute data upload to Clarifai app dataset
from clarifai.datasets.upload.loaders.coco_detection import COCODetectionDataLoader
coco_dataloader = COCODetectionDataLoader("images_dir", "coco_annotation_filepath")
dataset.upload_dataset(dataloader=coco_dataloader, get_upload_status=True)


#Try upload and record the failed outputs in log file.
from clarifai.datasets.upload.utils import load_module_dataloader
cifar_dataloader = load_module_dataloader('./image_classification/cifar10')
dataset.upload_dataset(dataloader=cifar_dataloader,
                       get_upload_status=True,
                       log_warnings =True)

#Retry upload from logs for `upload_dataset`
# Set retry_duplicates to True if you want to ingest failed inputs due to duplication issues. by default it is set to 'False'.
dataset.retry_upload_from_logs(dataloader=cifar_dataloader, log_file_path='log_file.log',
                               retry_duplicates=True,
                               log_warnings=True)

#upload text from csv
dataset.upload_from_csv(csv_path='csv_path', input_type='text', csv_type='raw', labels=True)

#upload data from folder
dataset.upload_from_folder(folder_path='folder_path', input_type='text', labels=True)

# Export Dataset
dataset.export(save_path='output.zip')
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

#### Input Download
```python
#listing inputs
input_generator = input_obj.list_inputs(page_no=1,per_page=1,input_type='image')
inputs_list = list(input_generator)

#downloading_inputs
input_bytes = input_obj.download_inputs(inputs_list)
with open('demo.jpg','wb') as f:
  f.write(input_bytes[0])
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
model_prediction = Model("https://clarifai.com/anthropic/completion/models/claude-v2").predict_by_bytes(b"Write a tweet on future of AI")

# Customizing Model Inference Output
model_prediction = gpt_4_model.predict_by_bytes(b"Write a tweet on future of AI", inference_params=dict(temperature=str(0.7), max_tokens=30))
# Return predictions having prediction confidence > 0.98
model_prediction = model.predict_by_filepath(filepath="local_filepath", output_config={"min_value": 0.98}) # Supports image, text, audio, video

# Supports prediction by url
model_prediction = model.predict_by_url(url="url") # Supports image, text, audio, video

# Return predictions for specified interval of video
video_input_proto = [input_obj.get_input_from_url("Input_id", video_url=BEER_VIDEO_URL)]
model_prediction = model.predict(video_input_proto, output_config={"sample_ms": 2000})
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

#### Export your trained model
Model Export feature enables you to package your trained model into a `model.tar` file. This file enables deploying your model within a Triton Inference Server deployment.

```python
from clarifai.client.model import Model

model = Model('url')
model.export('output/folder/')
```

#### Evaluate your trained model

When your model is trained and ready, you can evaluate by the following code

```python
from clarifai.client.model import Model

model = Model('url')
model.evaluate(dataset_id='your-dataset-id')
```

Compare the evaluation results of your models.

```python
from clarifai.client.model import Model
from clarifai.client.dataset import Dataset
from clarifai.utils.evaluation import EvalResultCompare

models = ['model url1', 'model url2'] # or [Model(url1), Model(url2)]
dataset = 'dataset url' # or Dataset(dataset_url)

compare = EvalResultCompare(
  models=models,
  datasets=dataset,
  attempt_evaluate=True # attempt evaluate when the model is not evaluated with the dataset
  )
compare.all('output/folder/')
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
workflow_prediction = workflow.predict_by_url(url="url") # Supports image, text, audio, video

# Customizing Workflow Inference Output
workflow = Workflow(user_id="user_id", app_id="app_id", workflow_id="workflow_id",
                  output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
workflow_prediction = workflow.predict_by_filepath(filepath="local_filepath") # Supports image, text, audio, video
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

## :mag: Search

#### Smart Image Search
Clarifai's Smart Search feature leverages vector search capabilities to power the search experience. Vector search is a type of search engine that uses vectors to search and retrieve text, images, and videos.

Instead of traditional keyword-based search, where exact matches are sought, vector search allows for searching based on visual and/or semantic similarity by calculating distances between vector embedding representations of the data.

Here is an example of how to use vector search to find similar images:

```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.search import Search
search = Search(user_id="user_id", app_id="app_id", top_k=1, metric="cosine")

# Search by image url
results = search.query(ranks=[{"image_url": "https://samples.clarifai.com/metro-north.jpg"}])

for data in results:
  print(data.hits[0].input.data.image.url)
```

#### Smart Text Search
Smart Text Search is our proprietary feature that uses deep learning techniques to sort, rank, and retrieve text data based on their content and semantic similarity.

Here is an example of how to use Smart Text Search to find similar text:

```python
# Note: CLARIFAI_PAT must be set as env variable.

# Search by text
results = search.query(ranks=[{"text_raw": "I love my dog"}])
```

#### Filters

You can use filters to narrow down your search results. Filters can be used to filter by concepts, metadata, and Geo Point.

It is possible to add together multiple search parameters to expand your search. You can even combine negated search terms for more advanced tasks.

For example, you can combine two concepts as below.

```python
# query for images that contain concept "deer" or "dog"
results = search.query(ranks=[{"image_url": "https://samples.clarifai.com/metro-north.jpg"}],
                        filters=[{"concepts": [{"name": "deer", "value":1},
                                              {"name": "dog", "value":1}]}])

# query for images that contain concepts "deer" and "dog"
results = search.query(ranks=[{"image_url": "https://samples.clarifai.com/metro-north.jpg"}],
                        filters=[{"concepts": [{"name": "deer", "value":1}],
                                  "concepts": [{"name": "dog", "value":1}]}])
```

Input filters allows to filter by input_type, status of inputs and by inputs_dataset_id

```python
results = search.query(filters=[{'input_types': ['image', 'text']}])
```

#### Pagination

Below is an example of using Search with Pagination.

```python
# Note: CLARIFAI_PAT must be set as env variable.
from clarifai.client.search import Search
search = Search(user_id="user_id", app_id="app_id", metric="cosine", pagination=True)

# Search by image url
results = search.query(ranks=[{"image_url": "https://samples.clarifai.com/metro-north.jpg"}],page_no=2,per_page=5)

for data in results:
  print(data.hits[0].input.data.image.url)
```

## Retrieval Augmented Generation (RAG)

You can setup and start your RAG pipeline in 4 lines of code. The setup method automatically creates a new app and the necessary components under the hood. By default it uses the [mistral-7B-Instruct](https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct) model.

```python
from clarifai.rag import RAG

rag_agent = RAG.setup(user_id="USER_ID")
rag_agent.upload(folder_path="~/docs")
rag_agent.chat(messages=[{"role":"human", "content":"What is Clarifai"}])
```

If you have previously run the setup method, you can instantiate the RAG class with the prompter workflow URL:

```python
from clarifai.rag import RAG

rag_agent = RAG(workflow_url="WORKFLOW_URL")
```

## :pushpin: More Examples

See many more code examples in this [repo](https://github.com/Clarifai/examples).
Also see the official [Python SDK docs](https://clarifai-python.readthedocs.io/en/latest/index.html)

## :open_file_folder: Model Upload

Examples for uploading models and runners have been moved to this [repo](https://github.com/Clarifai/runners-examples).
Find our official documentation at [docs.clarifai.com/compute/models/upload](https://docs.clarifai.com/compute/models/upload).
