#!/usr/bin/env python

import os
import sys

from client import ClarifaiApi


def tag_images_in_directory(path, api):
  images = []
  path = path.rstrip('/')
  for fname in os.listdir(path):
    images.append((open(os.path.join(path, fname)), fname))
  return api.tag_images(images)


def main(argv):
  if len(argv) > 1:
    imageurl = argv[1]
  else:
    imageurl = 'http://clarifai.com/img/toddler-flowers.jpg'

  api = ClarifaiApi()

  if imageurl.startswith('http'):
    response = api.tag_image_urls(imageurl)
  elif imageurl.endswith('/'):
    response = tag_images_in_directory(imageurl, api)
  else:
    with open(imageurl) as image_file:
      response = api.tag_images(image_file)
  print response

if __name__ == '__main__':
  main(sys.argv)
