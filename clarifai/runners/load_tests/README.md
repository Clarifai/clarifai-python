Load test runner using `locust`

# Install

```bash
pip install locust
```

# Run

Set below environment variables to test client

```bash
CLARIFAI_MODEL_ID
CLARIFAI_APP_ID
CLARIFAI_USER_ID
CLARIFAI_PAT
CLARIFAI_DEPLOYMENT_ID
CLARIFAI_API_BASE
```


## Test

To run one of the tests do:
```bash
locust -f tests/new_inference_locust.py
```

Then open in browser:
```
localhost:8089
```
