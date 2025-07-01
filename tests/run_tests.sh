#!/bin/bash
# You can now run this script to run the tests for the repo.
# if you have your `clarifai config use {context}` set then you can run
# eval $(clarifai config env) to source the variables into your shell and run_test.sh afterwards.
# This allows you to hit different deployments of the Clarifai platform and using your own account.

export PYTHONPATH=.

# if CLARIFAI_USER_ID does not exist, create it
if [ -z "$CLARIFAI_USER_ID" ]; then
    export CLARIFAI_USER_ID="$(uv run python scripts/key_for_tests.py --get-userid)"
fi
if [ -z "$CLARIFAI_PAT" ]; then
    export CLARIFAI_PAT="$(uv run python scripts/key_for_tests.py --create-pat)"
fi
# if CLARIFAI_API_BASE is not set, default to https://api.clarifai.com
if [ -z "$CLARIFAI_API_BASE" ]; then
    export CLARIFAI_API_BASE="https://api.clarifai.com"
fi
# if CLARIFAI_GRPC_BASE is not set, default to api.clarifai.com
if [ -z "$CLARIFAI_GRPC_BASE" ]; then
    export CLARIFAI_GRPC_BASE="api.clarifai.com"
fi

uv run pytest --cov=. --cov-report=xml:coverage/coverage.cobertura.xml --ignore=tests/runners/test_model_run_locally-container.py
