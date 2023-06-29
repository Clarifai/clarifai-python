## Inference Execution Environments

Each model built for inference with triton requires certain dependencies & dependency versions be installed for successful inference execution.
An execution environment is created for each model to be deployed on Clarifai and all necessary dependencies as listed in the `requirements.txt` file are installed there.

Pre-configured base environments with certain dependencies pre-installed are provided for users to build on top of as presented in the `triton_conda.yaml` file.

## Available pre-configured environments.

1. ```yaml
	 name: triton_conda-cp3.8-torch1.13.1-19f97078
	 ```
All dependencies in this environment can be [found here](../envs/triton_conda.yaml).

By default all `triton_conda.yaml` files in the generated model repository use the environment above as its currently the only one available.
Dependencies specified in the `requirements.txt` file are prioritized in case there's a difference in versions with those pre-installed in the base pre-configured environment.
