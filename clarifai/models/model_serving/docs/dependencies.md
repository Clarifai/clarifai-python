## Inference Execution Environments

Each model built for inference with triton requires certain dependencies & dependency versions be installed for successful inference execution.
An execution environment is created for each model to be deployed on Clarifai and all necessary dependencies as listed in the `requirements.txt` file are installed there.

## Supported python and torch versions

Currently, models must use python 3.8 (any 3.8.x).  Supported torch versions are 1.13.1 and 2.0.1.
If your model depends on torch, torch must be listed in your requirements.txt file (even if it is
already a dependency of another package).  An appropriate supported torch version will be selected
based on your requirements.txt.
