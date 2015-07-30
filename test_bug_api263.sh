#!/bin/bash

# just test the case we fail
wget -O /tmp/toddler-flowers.jpeg http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg
python bug_api263_reproduce.py

# install the client from the fixed branch and test with wider test cases
# pip install git+https://github.com/Clarifai/Clarifai_py.git@robert_unicode_localids_API263

# run the API test
# DJANGO_SETTINGS_MODULE=sites.clarifai_api.api.api.settings.dev nosetests conf/spire/tests/test_client.py:TestRemoteAPI

