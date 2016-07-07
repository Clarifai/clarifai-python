#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
unittest for Clarifai API Python Client
"""

import os
import glob
import hashlib
import unittest
from clarifai.client import ClarifaiApi, ApiError

class TestClarifaiApi(unittest.TestCase):
  """
  test the Clarifai API Python client with all supported features
  """

  image_url = 'http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg'
  video_url = 'http://techslides.com/demos/sample-videos/small.mp4'

  def get_client(self, *args, **kwargs):
    return ClarifaiApi()

  def test_api_connection(self):
    api = ClarifaiApi()
    self.assertTrue(api)

  def test_get_info(self):
    api = ClarifaiApi()
    response = api.get_info()
    self.assertTrue(response.get('api_version'))
    self.assertTrue(len(response) > 0)

  def test_tag_one_image(self):
    """ tag one image, from url and disk """
    # tag image from online URL
    image_url = 'http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg'
    api = ClarifaiApi()
    response = api.tag_image_urls(image_url)
    self.assertTrue(response)
    self.assertTrue(response['results'][0]['url'] == image_url)

    # tag image from local fs
    image_file = 'tests/data/toddler-flowers.jpeg'
    api = ClarifaiApi()
    if os.path.exists(image_file):
      with open(image_file, 'rb') as fb:
        response = api.tag_images(fb)
        self.assertTrue(response)

  def test_tag_images(self):
    """ tag multiple images, from url and disk """
    # tag images from online URL
    image_url_base = 'http://clarifai-img.s3.amazonaws.com/test'
    image_files = ['metro-north.jpg', 'octopus.jpg', 'tahoe.jpg', 'thai-market.jpg']
    image_urls = [os.path.join(image_url_base, one_file) for one_file in image_files]

    api = ClarifaiApi()
    response = api.tag_image_urls(image_urls)
    self.assertTrue(response)

    # tag images frmo local fs
    image_dir = 'tests/data'
    image_files = ['metro-north.jpg', 'octopus.jpg', 'tahoe.jpg', 'thai-market.jpg']

    api = ClarifaiApi()
    if os.path.exists(image_dir):
      image_files = [open(os.path.join(image_dir, one_file), 'rb') for one_file in image_files]
      response = api.tag_images(image_files)
      self.assertTrue(response)
      for fd in image_files:
        fd.close()

  def test_unicode_urls(self):
    image_url = u'http://www.alvaronoboa.com/wp-content/uploads/2013/02/Álvaro-Noboa-y-Annabella-Azín-Votaciones-41-1024x682.jpg'

    api = ClarifaiApi()
    response = api.tag_image_urls(image_url)
    self.assertTrue(response)
    self.assertTrue(response['results'][0]['url'] == image_url)

  def test_tag_gif(self):
    """ tag one GIF animation file """
    # source: http://media.giphy.com/media/fRZn2vraBGiA0/giphy.gif
    image_url = 'http://media.giphy.com/media/fRZn2vraBGiA0/giphy.gif'

    api = ClarifaiApi()
    response = api.tag_image_urls(image_url)
    self.assertTrue(response)
    self.assertTrue(response['results'][0]['url'] == image_url)

    image_file = 'tests/data/water-ocean-turtle.gif'
    api = ClarifaiApi()
    if os.path.exists(image_file):
      with open(image_file, 'rb') as fb:
        response = api.tag_images(fb)
        self.assertTrue(response)

  def test_tag_one_video(self):
    # video source: http://techslides.com/demos/sample-videos/small.mp4
    video_url = 'http://techslides.com/demos/sample-videos/small.mp4'

    api = ClarifaiApi()
    response = api.tag_image_urls(video_url)
    self.assertTrue(response)
    self.assertTrue(response['results'][0]['url'] == video_url)

  def test_tag_one_video_from_localfs(self):
    # video source: http://techslides.com/demos/sample-videos/small.mp4
    video_file = 'tests/data/small.mp4'
    api = ClarifaiApi()
    if os.path.exists(video_file):
      with open(video_file, 'rb') as fb:
        response = api.tag_images(fb)
        self.assertTrue(response)

  def check_unauth(self, error):
    """ Some users can't use some features, so check that message. """
    self.assertEqual(error.msg['status_code'], 'ALL_ERROR')
    self.assertEqual(error.msg['status_msg'], u'Not authorized to use argument. If you believe this is should not be the case, please contact support: ')

  # You need special permission to run these tests.
  def test_embed_one_image(self):
    image_url = 'http://clarifai-img.s3.amazonaws.com/test/toddler-flowers.jpeg'
    api = self.get_client()
    try:
      response = api.embed_image_urls(image_url)
      self.assertTrue(response)
      self.assertTrue(response['results'][0]['url'] == image_url)
    except ApiError, e: # User does not have permission.
      self.check_unauth(e)

  def test_embed_one_image_from_localfs(self):
    image_file = 'tests/data/toddler-flowers.jpeg'
    api = self.get_client()
    if os.path.exists(image_file):
      with open(image_file, 'rb') as fb:
        try:
          response = api.embed_images(fb)
          self.assertTrue(response)
        except ApiError, e:
          self.check_unauth(e)

  def test_tag_n_embed_one_image(self):
    image_url_base = 'http://clarifai-img.s3.amazonaws.com/test'
    image_files = ['metro-north.jpg', 'octopus.jpg', 'tahoe.jpg', 'thai-market.jpg']
    image_urls = [os.path.join(image_url_base, one_file) for one_file in image_files]

    api = self.get_client()
    try:
      response = api.tag_and_embed_image_urls(image_urls)
      self.assertTrue(response)
    except ApiError, e:
      self.check_unauth(e)

  def test_tag_n_embed_from_localfs(self):
    image_dir = 'tests/data'
    image_files = ['metro-north.jpg', 'octopus.jpg', 'tahoe.jpg', 'thai-market.jpg']

    api = self.get_client()
    if os.path.exists(image_dir):
      image_files = [open(os.path.join(image_dir, one_file), 'rb') for one_file in image_files]
      try:
        response = api.tag_and_embed_images(image_files)
        self.assertTrue(response)
      except ApiError, e:
        self.check_unauth(e)
      finally:
        for fd in image_files:
          fd.close()

  def test_send_feedback(self):
    """ test sending various feedback """

    urls = ['http://clarifai-img.s3.amazonaws.com/test/metro-north.jpg', \
            'http://clarifai-img.s3.amazonaws.com/test/metro-north.jpg', \
            'http://clarifai-img.s3.amazonaws.com/test/octopus.jpg']

    api = ClarifaiApi()

    response = api.feedback(urls=urls[0], add_tags='train')
    self.assertTrue(response)

    response = api.feedback(urls=urls[0], remove_tags='speed,test')
    self.assertTrue(response)

    response = api.feedback(urls=urls[0], add_tags='train', remove_tags='speed,test')
    self.assertTrue(response)

    docids = [hashlib.md5(url.encode('utf-8')).hexdigest() for url in urls]

    response = api.feedback(urls=urls[:2], similar_docids=docids[:2])
    self.assertTrue(response)

    response = api.feedback(urls=urls[1:], dissimilar_docids=docids[1:])
    self.assertTrue(response)

    response = api.feedback(urls=urls, similar_docids=docids[:2], dissimilar_docids=docids[1:])
    self.assertTrue(response)

  def test_i18n(self):
    api = ClarifaiApi(language='aaa')

    api = self.get_client()
    request_body = api._setup_multi_data([], ['urls'], language='en')
    self.assertEqual(request_body['language'], 'en', 'language field was not set')

    languages = api.get_languages()
    self.assertTrue(len(languages), 'did not return any languages')
    self.assertTrue('en' in languages, 'english code not included in languages')

  def test_color(self):
    """ test color api """

    api = ClarifaiApi()

    # test color api with image urls
    urls = ['http://clarifai-img.s3.amazonaws.com/test/metro-north.jpg', \
            'http://clarifai-img.s3.amazonaws.com/test/metro-north.jpg', \
            'http://clarifai-img.s3.amazonaws.com/test/octopus.jpg']
    for url in urls:
      response = api.color_urls(url)
      self.assertTrue(response)
      self.assertTrue(response['results'][0]['colors'])

    # test color api with local files
    files = glob.glob('tests/data/*.jpg')
    for onefile in files:
      response = api.color(open(onefile, 'rb'))
      self.assertTrue(response)
      self.assertTrue(response['results'][0]['colors'])

  def test_concept_ids(self):
    """new models should return concept_ids"""
    api = self.get_client()

    api.set_model('general-v1.3')
    response = api.tag_image_urls(self.image_url)
    tag = response['results'][0]['result']['tag']
    self.assertTrue('concept_ids' in tag, 'concept_ids not included in new model')
    self.assertTrue(tag['concept_ids'][0].startswith('ai_'), 'concept id doesn\'t start with ai_')

    response = api.tag_image_urls(self.video_url)
    self.assertTrue(response['results'][0]['result']['tag']['concept_ids'][0][0].startswith('ai_'),
                    "video concept_ids didn't start wit ai_")


    api.set_model('general-v1.1')
    response = api.tag_image_urls(self.image_url)
    tag = response['results'][0]['result']['tag']
    self.assertTrue('concept_ids' in tag, 'concept_ids not included in v1.1 model')

    response = api.tag_image_urls(self.video_url)
    tag = response['results'][0]['result']['tag']
    self.assertTrue('concept_ids' in tag, 'concept_ids not included in v1.1 model')


if __name__ == '__main__':
  unittest.main()
