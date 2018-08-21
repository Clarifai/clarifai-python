![Clarifai logo](docs/logo.png)

# Clarifai API Python Client

This Python client provides a simple wrapper around our powerful image recognition <a href="https://developer.clarifai.com">API</a>.

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/developer/account/signup/
* Read the developer guide at: https://clarifai.com/developer/guide/


[![PyPi version](https://pypip.in/v/clarifai/badge.png)](https://pypi.python.org/pypi/clarifai)
[![Build Status](https://travis-ci.org/Clarifai/clarifai-python.svg?branch=master)](https://travis-ci.org/Clarifai/clarifai-python)


Installation
---------------------
The API client is available on Pip. Install it with:
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

