#!/bin/bash

find . -name "*.pyc" | xargs rm -f
find . -type d -name "__pycache__" | xargs rm -fr

cp -pr setup.py README.md clarifai requirements.txt CHANGELOG.md ../clarifai-python/
cp -pr tests/rest_tests tests/README.md tests/requirements.txt ../clarifai-python/tests
cp -pr scripts/clarifai ../clarifai-python/scripts/
cp -pr scripts/app_and_key_for_tests.py ../clarifai-python/scripts/
cp -pr docs/*.rst docs/*.py docs/*.png ../clarifai-python/docs/
cp -pr .travis.yml ../clarifai-python
cp -pr assert-code-quality.sh ../clarifai-python
cp -pr .isort.cfg ../clarifai-python
cp -pr .pylintrc ../clarifai-python
cp -pr .style.yapf ../clarifai-python

rm -f ../clarifai-python/clarifai/client/README.md
rm -f ../clarifai-python/clarifai/client/sample_main.py

