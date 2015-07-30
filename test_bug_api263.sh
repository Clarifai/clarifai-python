#!/bin/bash

wget -O /tmp/toddler-flowers.jpeg http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg
python bug_api263_reproduce.py
