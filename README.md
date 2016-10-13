API V2 Python Client
====================

This Python client provides a simple wrapper around our powerful image recognition <a href="http://developer.clarifai.com">API</a>.

The constructor takes your APP_ID and APP_SECRET created in your Clarifai Account. You can also
set these variables in your environment as:
CLARIFAI_APP_ID
CLARIFAI_APP_SECRET

You can also setup the base path to any of the provided API urls.
CLARIFAI_API_BASE

The default value for CLARIFAI_API_BASE is https://api.clarifai.com


This client uses your "CLARIFAI_APP_ID" and "CLARIFAI_APP_SECRET" to get an access token. Since this
expires every so often, the client is setup to renew the token for you automatically using your
credentials so you don't have to worry about it.

The client also uses Applications to store images and visually search across them. You can either do
a simple visual search or also add predictions for any, all or none, as noted in the directions
below.

Installation
---------------------
Unzip the package to somewhere then run setup.py to install:
```
pip install clarifai==2.0.4
```

Getting Started
---------------------
The following example will setup the client and predict from our general model
```python
from clarifai.rest import ClarifaiApp

app = ClarifaiApp()

model = app.models.get('general-v1.3')

print model.predict_by_url('https://samples.clarifai.com/metro-north.jpg')
```

Features
---------------------
Read the documentation for more code examples and references at https://sdk.clarifai.com/python/docs/2.0.4/index.html

