#!/usr/bin/env python

"""
sample python script for using Clarifai Python Client API

usage: sample_main.py [-h] [-t] [-c]

optional arguments:
  -h, --help   show this help message and exit
  -t, --tag    tag images
  -c, --color  color images

Examples:

Run tag api on an image url or image on disk
 python sample_main.py <url|filename>
 python sample_main.py -t|--tag <url|filename>

Run color api on an image url or image on disk
 python sample_main.py -c|--color <url|filename>

"""

import os
import sys
import json
import glob
import argparse

from clarifai.client import ClarifaiApi


def tag_images(api, imageurl):
  if imageurl.startswith('http'):
    response = api.tag_image_urls(imageurl)
  elif os.path.isfile(imageurl):
    with open(imageurl,'rb') as image_file:
      response = api.tag_images(image_file)

  return response

def color_images(api, imageurl):
  if imageurl.startswith('http'):
    response = api.color_urls(imageurl)
  elif os.path.isfile(imageurl):
    with open(imageurl,'rb') as image_file:
      response = api.color(image_file)

  return response


def main(argv):

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--tag", help="tag images", action='store_true')
  parser.add_argument("-c", "--color", help="color images", action='store_true')

  args, argv = parser.parse_known_args()

  if len(argv) == 1:
    imageurl = argv[0]
  else:
    imageurl = 'http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg'

  api = ClarifaiApi()

  if not args.color:
    response = tag_images(api, imageurl)
  elif args.color and not args.tag:
    response = color_images(api, imageurl)
  else:
    raise Exception('call with --tag or --color for the image')

  print(json.dumps(response))

if __name__ == '__main__':
  main(sys.argv)
