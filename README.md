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

## Interacting with Inputs

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

# uploading images from folder
input_obj.upload_images_from_folder(folder_path='demo_folder')
```
