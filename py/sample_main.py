#!/usr/bin/env python

import sys

from client import ClarifaiApi

def main(argv):
  if len(argv) > 1:
    imageurl = argv[1]
  else:
    imageurl = 'http://clarifai.com/img/toddler-flowers.jpg'

  api = ClarifaiApi()
  if imageurl.startswith('http'):
    response = api.tag_image_url(imageurl)
  else:
    with open(imageurl) as image_file:
      response = api.tag_image(image_file)
  print response

if __name__ == '__main__':
  main(sys.argv)
