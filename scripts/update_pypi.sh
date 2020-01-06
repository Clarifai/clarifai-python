#!/bin/bash

rm -fr build dist

python setup.py bdist_wheel sdist

twine upload dist/*

