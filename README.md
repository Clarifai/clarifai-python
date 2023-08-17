![Clarifai logo](docs/logo.png)



# Clarifai API Python Client

This is the official Python client for interacting with our powerful recognition [API](https://docs.clarifai.com).
The Clarifai Python SDK offers a comprehensive set of tools to integrate Clarifai's AI-powered image, video, and text recognition capabilities into your applications. With just a few lines of code, you can leverage cutting-edge artificial intelligence to unlock valuable insights from visual and textual content.

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/developer/account/signup/
* Read the developer guide at: https://clarifai.com/developer/guide/

## Getting started
Clarifai uses Personal Access Tokens(PATs) to validate requests. You can create and manage PATs under your Clarifai account security settings.

Export your PAT as an environment variable. Then, import and initialize the API Client.


```cmd
export CLARIFAI_PAT={your personal access token}
```

```python
from clarifai.client.user import User
client = User(user_id="user_id")

# Get all apps
apps = client.list_apps()

# Create app and dataset
app = client.create_app(app_id="demo_app")
dataset = app.create_dataset(dataset_id="demo_dataset")
# execute data upload to Clarifai app dataset
dataset.upload_dataset(task='visual_segmentation', split="train", dataset_loader='coco_segmentation')
```

## Interacting with Models

### Model Predict
```python
from clarifai.client.model import Model

# Model Predict
model = Model(user_id="user_id", app_id="app_id", model_id="model_id")
model_prediction = model.predict_by_url(url="url", input_type="image") # Supports image, text, audio, video

# Customizing Model Inference Output
model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                  output_config={"min_value": 0.98}) # Return predictions having prediction confidence > 0.98
model_prediction = model.predict_by_filepath(filepath="local_filepath", input_type="text") # Supports image, text, audio, video

model = Model(user_id="user_id", app_id="app_id", model_id="model_id",
                    output_config={"sample_ms": 2000}) # Return predictions for specified interval
model_prediction = model.predict_by_url(url=BEER_VIDEO_URL, input_type="video")
```
### Models Listing
```python
# List all model versions
all_model_versions = model.list_versions()

# Go to specific model version
model_v1 = client.app("app_id").model("model_id").version("model_version_id")

# List all models in an app
all_models = app.list_models()

# List all models in community filtered by model_type, description
all_llm_community_models = App().list_models(filter_by={"query": "LLM",
                                                        "model_type_id": "text-to-text"}, only_in_app=False)
```