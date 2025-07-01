#!/bin/bash
set -e
# Before running make sure you run:
# uv pip install -r requirements.txt -r tests/requirements.txt

# You can now run this script to run the tests for the repo.
# if you have your `clarifai config use {context}` set then you can run
# eval $(clarifai config env) to source the variables into your shell and run_test.sh afterwards.
# This allows you to hit different deployments of the Clarifai platform and using your own account.

echo "Creating a new app and keys, so the test run is completely isolated from test runs."
# if CLARIFAI_USER_ID does not exist, create it
if [ -z "$CLARIFAI_USER_ID" ]; then
    # you need to have CLARIFAI_USER_EMAIL and CLARIFAI_USER_PASSWORD set in your environment
    if [ -z "$CLARIFAI_USER_EMAIL" ] || [ -z "$CLARIFAI_USER_PASSWORD" ]; then
        echo "Please set CLARIFAI_USER_EMAIL and CLARIFAI_USER_PASSWORD environment variables."
        exit 1
    fi
    export CLARIFAI_USER_ID="$(uv run python scripts/key_for_tests.py --get-userid)"
fi
# if CLARIFAI_PAT does not exist, create it
if [ -z "$CLARIFAI_PAT" ]; then
    # you need to have CLARIFAI_USER_EMAIL and CLARIFAI_USER_PASSWORD set in your environment
    if [ -z "$CLARIFAI_USER_EMAIL" ] || [ -z "$CLARIFAI_USER_PASSWORD" ]; then
        echo "Please set CLARIFAI_USER_EMAIL and CLARIFAI_USER_PASSWORD environment variables."
        exit 1
    fi
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

# First run the linter tests to make sure those pass.
uv run pre-commit run --all-files


echo "Running all the tests"
# See .github/workflows/run_tests.yml as there are more combinations of tests that can be run but
# this should cover the basics.
uv run pytest -s tests/ -v -n auto --durations=5 --timeout=1800 --ignore=tests/runners/test_model_run_locally-container.py


# TODO: cleanup better from these tests capturing the exit code first.
# test_result=$?
# echo "Running single test"
# python3 -m pytest tests/ -vvv -s -k "test_predict_image_url_with_min_value"
# echo "Deleting the created application"
# python3 scripts/app_and_key_for_tests.py --delete-app ${CLARIFAI_APP_ID}
# exit $test_result
