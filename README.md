![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=240)

# clarifai-python-utils


This is the official Clarifai Python utilities project. This repo includes higher level convenience classes, functions, and scripts to make using our [API](https://docs.clarifai.com) easier. This is built on top of the [Clarifai Python gRPC Client](https://github.com/Clarifai/clarifai-python-grpc).

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/signup
* Read the documentation at: https://docs.clarifai.com/


## Installation

```cmd
pip install -U clarifai-utils
```

## Installation from source (for development)
```cmd
python -m venv ~/virtualenv/clarifai-python-utils
source ~/virtualenv/clarifai-python-utils/bin/activate
cd clarifai-python-utils
python setup.py develop
```
## Versioning

This library doesn't use semantic versioning. The first two version numbers (`X.Y` out of `X.Y.Z`) follow the API (backend) versioning, and
whenever the API gets updated, this library follows it.

The third version number (`Z` out of `X.Y.Z`) is used by this library for any independent releases of library-specific improvements and bug fixes.

## Getting started

Here is a quick example of listing all the concepts in an application.

Set some env vars first
```cmd
export CLARIFAI_USER_ID={the user_id of the app_id of the app you want to access resources in}
export CLARIFAI_APP_ID={the app_id of the app you want to access resources in}
export CLARIFAI_PAT={your personal access token}
```

```python
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister

# Get auth information from those env vars.
auth = ClarifaiAuthHelper.from_env()

# Create the grpc client stub.
stub = auth.get_stub()

# Create the resource lister.
lister = ClarifaiResourceLister(stub, auth.metadata, auth.user_id, auth.app_id, page_size=16)

# List all the concepts in the app showing:
concepts = []
for c in lister.concepts_generator():
  concepts.append(c)
```
