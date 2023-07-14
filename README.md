![Clarifai logo](docs/logo.png)



# Clarifai API Python Client

This is the official Python client for interacting with our powerful recognition [API](https://docs.clarifai.com).
The Clarifai API offers image and video recognition as a service. Whether you have one image or billions,
you are only steps away from using artificial intelligence to recognize your visual content.

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/developer/account/signup/
* Read the developer guide at: https://clarifai.com/developer/guide/


## Getting started

Here is a quick example of listing all the concepts in an application.

Set some env vars first
```cmd
export CLARIFAI_PAT={your personal access token}
```

```python
from clarifai.client.api import ApiClient

client = ApiClient()

# List all users
print(client.list_users())

# List all apps for a user
print(client.user('user_id').list_apps())

# Get an app
app = client.user('user_id').app('app_id')
```
