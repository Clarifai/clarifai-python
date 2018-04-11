Clarifai API Python Client
====================


Overview
---------------------
This Python client provides a simple wrapper around our powerful image recognition <a href="https://developer.clarifai.com">API</a>.

The client supports basic tagging with existing models.

The client also uses Applications to store images and visually search across them. You can either do
a simple visual search or also add predictions for any, all or none, as noted in the directions
below.


Installation
---------------------
The API client is available on Pip. You can simply install it with a `pip install`
```
pip install clarifai --upgrade
```

For more details on the installation, please refer to https://clarifai-python.readthedocs.io/en/latest/install/

Setup
---------------------
The client uses your "CLARIFAI_API_KEY" to get an access token. Since this
expires every so often, the client is setup to renew the token for you automatically using your
credentials so you don't have to worry about it.

You can get the `api_key` from https://developer.clarifai.com and config them for client's use by

```bash
$ clarifai config
CLARIFAI_API_KEY: []: ************************************YQEd

```

The config will be stored under ~/.clarifai/config for client's use

Environmental variable CLARIFAI_API_KEY will override the settings in the config file.

For AWS or Windows users, please refer to https://clarifai-python.readthedocs.io/en/latest/install/ for more instructions.


Getting Started
---------------------
The following example will setup the client and predict from our general model
```python
from clarifai.rest import ClarifaiApp

app = ClarifaiApp()
model = app.public_models.general_model
response = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')
```

If wanting to predict a local file, use `predict_by_filename`.

The response is a JSON structure. Here's how to print all the predicted concepts associated with the image, together with their confidence values.

```python
concepts = response['outputs'][0]['data']['concepts']
for concept in concepts:
    print(concept['name'], concept['value'])
```

Documentation
---------------------
Read more code examples and references at https://clarifai-python.readthedocs.io/en/latest/

