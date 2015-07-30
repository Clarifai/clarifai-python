#!/usr/bin/env python

import os
import sys

from clarifai.client import ClarifaiApi

def main():

  api = ClarifaiApi()

  f = open('/tmp/toddler-flowers.jpeg', 'r')
  local_ids = [u'fake-docid-for-url-sim']
  response = api.tag_and_embed_images([f], model='default', local_ids=local_ids)
  print response


if __name__ == '__main__':
  main()

