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
Unzip the package to somewhere then run setup.py to install:
```
pip install clarifai==2.0.14
```


Setup
---------------------
The client uses your "CLARIFAI_APP_ID" and "CLARIFAI_APP_SECRET" to get an access token. Since this
expires every so often, the client is setup to renew the token for you automatically using your
credentials so you don't have to worry about it.

You can get the `id` and `secret` from https://developer.clarifai.com and config them for client's use by

```bash
$ clarifai config
CLARIFAI_APP_ID: []: ************************************YQEd
CLARIFAI_APP_SECRET: []: ************************************gCqT

```

The config will be stored under ~/.clarifai/config for client's use

Environmental variable CLARIFAI_APP_ID and CLARIFAI_APP_SECRET will override the settings in the config file.


Getting Started
---------------------
The following example will setup the client and predict from our general model
```python
from clarifai.rest import ClarifaiApp

app = ClarifaiApp()
app.tag_urls(['https://samples.clarifai.com/metro-north.jpg'])
```

Documentations
---------------------
Read more code examples and references at https://sdk.clarifai.com/python/docs/2.0.14/index.html

