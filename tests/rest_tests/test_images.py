import base64
import logging
import time
import unittest
import uuid
from io import BytesIO

from clarifai.rest import ApiClient, ApiError, ClarifaiApp, Geo, GeoPoint
from clarifai.rest import Image as ClarifaiImage
from clarifai.rest import UserError

from . import sample_inputs


class TestImages(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.api = ApiClient(log_level=logging.WARN)
    cls.app = ClarifaiApp(log_level=logging.WARN)

  @classmethod
  def tearDownClass(cls):
    """ Cleanup """

  def test_init_images(self):
    """ initialize Image object in different ways """
    img1 = ClarifaiImage(url=sample_inputs.FACEBOOK_IMAGE_URL)
    with open(sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH, 'rb') as f:
      img2 = ClarifaiImage(file_obj=f)
    img3 = ClarifaiImage(filename=sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH)

    with open(sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH, 'rb') as f:
      toddler_flowers_file_bytes = f.read()
    toddler_flowers_base64_bytes = base64.b64encode(toddler_flowers_file_bytes)
    img4 = ClarifaiImage(base64=toddler_flowers_base64_bytes)

    # init with crops
    img1 = ClarifaiImage(url=sample_inputs.FACEBOOK_IMAGE_URL, crop=[0.1, 0.3, 0.5, 0.7])

    f = open(sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH, 'rb')
    img2 = ClarifaiImage(file_obj=f, crop=[0.1, 0.3, 0.5, 0.7])
    f.close()

    img3 = ClarifaiImage(
        filename=sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH, crop=[0.1, 0.3, 0.5, 0.7])

    img4 = ClarifaiImage(base64=toddler_flowers_base64_bytes, crop=[0.1, 0.3, 0.5, 0.7])

    self.assertListEqual(img1.crop, [0.1, 0.3, 0.5, 0.7])
    self.assertListEqual(img2.crop, [0.1, 0.3, 0.5, 0.7])
    self.assertListEqual(img3.crop, [0.1, 0.3, 0.5, 0.7])
    self.assertListEqual(img4.crop, [0.1, 0.3, 0.5, 0.7])

    # init with urls with spaces
    img5 = ClarifaiImage(url=' ' + sample_inputs.FACEBOOK_IMAGE_URL)
    self.assertEqual(img5.url, sample_inputs.FACEBOOK_IMAGE_URL)

    img6 = ClarifaiImage(url=' ' + sample_inputs.FACEBOOK_IMAGE_URL + ' ')
    self.assertEqual(img6.url, sample_inputs.FACEBOOK_IMAGE_URL)

  def test_get_all_inputs(self):
    """ test get all inputs """

    # try to iterate all inputs
    ccount = 0
    for input in self.app.inputs.get_all():
      self.assertTrue(isinstance(input, ClarifaiImage))
      ccount += 1
      if ccount >= 50:
        break

    self.assertGreaterEqual(ccount, 0)

  def test_get_inputs_by_page(self):
    """ test get all images """

    # test the by_page() fetcher on first page
    for image in self.app.inputs.get_by_page():
      self.assertTrue(isinstance(image, ClarifaiImage))

  def test_attributes_via_create_image(self):
    """ test attribute after image creation """
    image = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(image.url, sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(len(image.input_id) in [22, 32], True)
    # FIXME(robert): fix this after the backend fix
    # self.assertTrue(image.allow_dup_url)
    self.assertIsNone(image.geo)
    self.assertIsNone(image.file_obj)
    self.assertIsNone(image.concepts)
    self.assertIsNone(image.not_concepts)

    image_id = image.input_id

    img = self.app.inputs.get(image_id)
    self.assertEqual(image_id, img.input_id)
    # FIXME(robert): fix this after the backend fix
    # self.assertTrue(image.allow_dup_url)
    self.assertIsNone(image.geo)
    self.assertIsNone(image.file_obj)
    self.assertIsNone(image.concepts)
    self.assertIsNone(image.not_concepts)

  def test_post_image_with_url_spaces(self):
    """ add image from url with leading or trailing spaces """

    image = self.app.inputs.create_image_from_url(
        url=' ' + sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(image.url, sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(len(image.input_id) in [22, 32], True)

    image2 = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL + ' ', allow_duplicate_url=True)
    self.assertEqual(image.url, sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(len(image.input_id) in [22, 32], True)

    self.app.inputs.delete(input_id=image.input_id)
    self.app.inputs.delete(input_id=image2.input_id)

  def test_post_get_delete_image(self):
    """ add image, fetch it, and delete it """

    image = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(image.url, sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(len(image.input_id) in [22, 32], True)

    image2 = self.app.inputs.get(input_id=image.input_id)
    self.assertEqual(image2.input_id, image.input_id)

    self.app.inputs.delete(input_id=image2.input_id)

  def test_delete_multiple_image(self):
    """ add a few and delete them all with bulk """
    img1 = self.app.inputs.create_image_from_url(
        url=sample_inputs.WEDDING_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(img1.url, sample_inputs.WEDDING_IMAGE_URL)

    img2 = self.app.inputs.create_image_from_url(
        url=sample_inputs.FACEBOOK_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(img2.url, sample_inputs.FACEBOOK_IMAGE_URL)

    img3 = self.app.inputs.create_image_from_url(
        url=sample_inputs.DOG_TIFF_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(img3.url, sample_inputs.DOG_TIFF_IMAGE_URL)

    ret = self.app.inputs.delete(input_id=[img1.input_id, img2.input_id, img3.input_id])
    time.sleep(3)

  def test_post_bulk_images(self):
    """ post bulk post images """
    img1 = ClarifaiImage(
        url=sample_inputs.METRO_IMAGE_URL, concepts=['train', 'railway'], allow_dup_url=True)
    img2 = ClarifaiImage(
        url=sample_inputs.WEDDING_IMAGE_URL,
        concepts=['wedding'],
        not_concepts=['food'],
        allow_dup_url=True)
    ret_imgs = self.app.inputs.bulk_create_images([img1, img2])

    self.assertEqual(len(list(ret_imgs)), 2)
    # sleep here to ensure inputs are all properly added before deleting them
    time.sleep(2)

    for img in ret_imgs:
      self.assertTrue(isinstance(img, ClarifaiImage))
      try:
        self.app.inputs.delete(img.input_id)
      except ApiError:
        pass

  def test_post_dup_url_with_check(self):
    """ by default the dup url check is enabled, the dup url will be rejected """
    image_id = uuid.uuid4().hex
    img1 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    with self.assertRaises(ApiError):
      image_id = uuid.uuid4().hex
      img2 = self.app.inputs.create_image_from_url(
          image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=False)

    self.app.inputs.delete(img1.input_id)

  def test_post_dup_url_without_check(self):
    """ we can skip the dup checking """

    image_id = uuid.uuid4().hex
    img1 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    image_id = uuid.uuid4().hex
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    image_id = uuid.uuid4().hex
    img3 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)
    self.app.inputs.delete(img3.input_id)

  def test_post_cropped_image(self):
    """ add cropped image """

    image = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, crop=[0.2, 0.4, 0.3, 0.6], allow_duplicate_url=True)
    self.assertEqual(image.url, sample_inputs.METRO_IMAGE_URL)
    self.assertEqual(len(image.input_id) in [22, 32], True)

    image2 = self.app.inputs.get(input_id=image.input_id)
    self.assertEqual(image2.input_id, image.input_id)

    self.app.inputs.delete(input_id=image2.input_id)

  def test_post_image_with_id(self):
    """ add image with id """

    image_id = uuid.uuid4().hex
    res = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        crop=[0.2, 0.4, 0.3, 0.6],
        allow_duplicate_url=True)

    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(res.input_id, image_id)

    res2 = self.app.inputs.get(image_id)
    self.assertEqual(res.input_id, res2.input_id)

    self.app.inputs.delete(res2.input_id)

  def test_post_image_with_tags(self):
    """ add image with tags """

    image_id = uuid.uuid4().hex

    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL,
        concepts=['aa1', 'aa2'],
        not_concepts=['bb1', 'bb2'],
        image_id=image_id,
        allow_duplicate_url=True)

    res = self.app.inputs.get(image_id)
    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(image_id, res.input_id)

    self.app.inputs.delete(image_id)

  def test_post_image_with_metadata(self):
    """ add image with meta data, retrieve it and compare the meta data """

    image_id = uuid.uuid4().hex
    meta = {
        'myid': image_id,
        'key2': {
            'key3': 4,
            'key4#$!': True,
        }
    }

    self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL,
        metadata=meta,
        image_id=image_id,
        allow_duplicate_url=True)
    try:
      res = self.app.inputs.get(image_id)
    finally:
      self.app.inputs.delete(image_id)

    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(image_id, res.input_id)
    self.assertEqual(meta, res.metadata)

  def test_post_image_with_geo(self):
    """ add image with geo info, retrieve it and compare the meta data """

    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(-30, 40))
    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, geo=geo, image_id=image_id, allow_duplicate_url=True)

    res = self.app.inputs.get(image_id)
    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(image_id, res.input_id)

    geo_res = res.geo
    self.assertAlmostEqual(geo_res.geo_point.longitude, -30, delta=0.0001)
    self.assertAlmostEqual(geo_res.geo_point.latitude, 40, delta=0.0001)

    self.app.inputs.delete(image_id)

  def test_post_image_with_geo_and_metadata(self):
    """ add image with geo info, retrieve it and compare the meta data """

    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(-30, 40))
    meta = {'myid': image_id, 'key_id': 'test_meta'}

    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL,
        geo=geo,
        metadata=meta,
        image_id=image_id,
        allow_duplicate_url=True)

    res = self.app.inputs.get(image_id)
    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(image_id, res.input_id)

    geo_res = res.geo
    self.assertTrue(isinstance(geo_res, Geo))
    self.assertAlmostEqual(geo_res.geo_point.longitude, -30, delta=0.0001)
    self.assertAlmostEqual(geo_res.geo_point.latitude, 40, delta=0.0001)

    self.assertEqual(meta, res.metadata)

    self.app.inputs.delete(image_id)

  def test_post_images_url(self):
    """ upload images from url """

    # upload by url
    img = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertEqual(img.url, sample_inputs.METRO_IMAGE_URL)
    img_ret = self.app.inputs.get(input_id=img.input_id)
    self.assertEqual(img_ret.url, img.url)
    self.app.inputs.delete(img.input_id)

  def test_post_images_filename(self):
    """ upload images from filename """

    # upload by filename
    img = self.app.inputs.create_image_from_filename(sample_inputs.TAHOE_IMAGE_FILE_PATH)
    self.assertTrue(img.url.startswith("https://s3.amazonaws.com/clarifai-api/img"))
    img_ret = self.app.inputs.get(input_id=img.input_id)
    self.assertEqual(img_ret.url, img.url)
    self.app.inputs.delete(img.input_id)

  def test_post_images_bytes(self):
    """ upload images from bytes """

    # upload by bytes
    with open(sample_inputs.TAHOE_IMAGE_FILE_PATH, 'rb') as f:
      file_bytes = f.read()
    img = self.app.inputs.create_image_from_bytes(file_bytes)
    self.assertTrue(img.url.startswith("https://s3.amazonaws.com/clarifai-api/img"))
    img_ret = self.app.inputs.get(input_id=img.input_id)
    self.assertEqual(img_ret.url, img.url)
    self.app.inputs.delete(img.input_id)

  def test_post_images_base64(self):
    """ upload images from base64 """

    # upload by base64 bytes
    with open(sample_inputs.TAHOE_IMAGE_FILE_PATH, 'rb') as f:
      file_bytes = f.read()
    base64_bytes = base64.b64encode(file_bytes)
    img = self.app.inputs.create_image_from_base64(base64_bytes)
    self.assertTrue(img.url.startswith("https://s3.amazonaws.com/clarifai-api/img"))
    img_ret = self.app.inputs.get(input_id=img.input_id)
    self.assertEqual(img_ret.url, img.url)
    self.app.inputs.delete(img.input_id)

  def test_check_status(self):
    """ check process status """
    counts = self.app.inputs.check_status()
    counts.dict()

  def test_delete_all_images(self):
    """ test delete all images and verify the input count """
    # ret = self.app.inputs.delete_all()
    # time.sleep(5)
    # FIXME(robert): this has to be tested in a separate app

  def test_patch_metadata(self):
    """ test patching metadata """

    # add an image without label
    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertTrue(isinstance(img, ClarifaiImage))
    self.assertEqual(img.url, sample_inputs.METRO_IMAGE_URL)

    # patch with metadata (adding first)
    img_res = self.app.inputs.merge_metadata(
        input_id=img.input_id, metadata={
            u'key1': 123,
            u'key2': 234
        })
    self.assertDictEqual(img_res.metadata, {u'key1': 123, u'key2': 234})

    # patch with metadata (merge)
    img_res = self.app.inputs.merge_metadata(
        input_id=img.input_id, metadata={
            u'key3': 345,
            u'key4': 456
        })
    self.assertDictContainsSubset({u'key1': 123, u'key2': 234}, img_res.metadata)
    self.assertDictContainsSubset({u'key3': 345, u'key4': 456}, img_res.metadata)

    # patch with metadata (merge)
    img_res = self.app.inputs.merge_metadata(
        input_id=img.input_id, metadata={
            u'key3': 345,
            u'key4': 456
        })
    self.assertDictContainsSubset({u'key1': 123, u'key2': 234}, img_res.metadata)
    self.assertDictContainsSubset({u'key3': 345, u'key4': 456}, img_res.metadata)

    # delete image
    self.app.inputs.delete(img.input_id)

  def test_image_concepts_applied_correctly(self):
    """ concepts and not_concepts should be applied correctly in create_image """

    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL,
        concepts=['cat', 'animal'],
        not_concepts=['vehicle'],
        allow_duplicate_url=True)
    self.assertSetEqual(set(img.concepts), {'cat', 'animal'})
    self.assertSetEqual(set(img.not_concepts), {'vehicle'})

    self.app.inputs.delete(img.input_id)

  def test_patch_image(self):
    """ update concepts for an image """

    # add an image without label
    img = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)
    self.assertTrue(isinstance(img, ClarifaiImage))
    self.assertEqual(img.url, sample_inputs.METRO_IMAGE_URL)

    img2 = self.app.inputs.get(input_id=img.input_id)
    self.assertTrue(isinstance(img2, ClarifaiImage))

    img3 = self.app.inputs.create_image_from_url(
        url=sample_inputs.WEDDING_IMAGE_URL, allow_duplicate_url=True)
    self.assertTrue(isinstance(img, ClarifaiImage))
    self.assertEqual(img3.url, sample_inputs.WEDDING_IMAGE_URL)

    # add tags
    res3 = self.app.inputs.merge_concepts(
        input_id=img.input_id, concepts=['cat', 'animal'], not_concepts=['vehicle'])
    self.assertTrue(isinstance(res3, ClarifaiImage))
    self.assertSetEqual(set(res3.concepts), {'cat', 'animal'})

    # overwrite tags
    res3 = self.app.inputs.merge_concepts(
        input_id=img.input_id,
        concepts=['dog', 'animal'],
        not_concepts=['vehicle'],
        overwrite=True)
    self.assertTrue(isinstance(res3, ClarifaiImage))
    # input_new = self.app.inputs.get(res3.input_id)
    # self.assertSetEqual(set(input_new.concepts), {'dog', 'animal'})

    # delete tags
    res3 = self.app.inputs.delete_concepts(input_id=img.input_id, concepts=['animal'])
    self.assertTrue(isinstance(res3, ClarifaiImage))
    # self.assertSetEqual(set(res3.concepts), {'dog'})

    # bulk merge tags
    res3 = self.app.inputs.bulk_merge_concepts([img.input_id, img3.input_id],
                                               [[('cat', True),
                                                 ('animal', False)], [('vehicle', False)]])
    self.assertTrue(isinstance(res3, list))
    self.assertTrue(all([isinstance(one, ClarifaiImage) for one in res3]))
    # self.assertSetEqual(set(res3[0].concepts), {'cat', 'dog'})
    # self.assertSetEqual(set(res3[0].not_concepts), {'animal'})
    # self.assertSetEqual(set(res3[1].not_concepts), {'vehicle'})

    # bulk delete tags
    res3 = self.app.inputs.bulk_delete_concepts([img.input_id, img3.input_id],
                                                [['cat', 'animal'], ['vehicle']])
    self.assertTrue(isinstance(res3, list))
    self.assertTrue(all([isinstance(one, ClarifaiImage) for one in res3]))

    # delete image
    self.app.inputs.delete(img2.input_id)
    self.app.inputs.delete(img3.input_id)

  def test_user_errors(self):
    with self.assertRaises(UserError):
      ClarifaiImage(url="blah", file_obj="hey")
    with self.assertRaises(UserError):  # not using open('rb')
      ClarifaiImage(file_obj=open(sample_inputs.TODDLER_FLOWERS_IMAGE_FILE_PATH, mode='r'))

  def test_base64_from_fileobj(self):

    # send it from a remote url
    data = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content

    image = self.app.inputs.create_image(ClarifaiImage(file_obj=BytesIO(data)))
    self.assertEqual(len(image.input_id) in [22, 32], True)
    self.assertTrue(image.url.startswith("https://s3.amazonaws.com/clarifai-api/img"))

    res2 = self.app.inputs.get(input_id=image.input_id)
    self.assertEqual(image.input_id, res2.input_id)

  def test_base64_from_pil(self):

    # Send it from PIL image buffer
    imgbytes = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content

    res = self.app.inputs.create_image_from_bytes(imgbytes)
    self.assertTrue(isinstance(res, ClarifaiImage))
    self.assertEqual(len(res.input_id) in [22, 32], True)
    self.assertTrue(res.url.startswith("https://s3.amazonaws.com/clarifai-api/img"))

    res2 = self.app.inputs.get(input_id=res.input_id)
    self.assertEqual(res.input_id, res2.input_id)

  def test_partial_errors(self):
    """ upload a few failed urls and fetch by pages
        make sure the partial error is coming and well handled
    """

    img1 = ClarifaiImage(url=sample_inputs.DOG2_IMAGE_URL, allow_dup_url=True)
    img2 = ClarifaiImage(url=sample_inputs.DOG2_NONEXISTENT_IMAGE_URL, allow_dup_url=True)

    imgs = self.app.inputs.bulk_create_images([img1] * 5 + [img2] * 2 + [img1] * 5)
    bad_ids = []
    for img in imgs:
      if img.url == sample_inputs.DOG2_NONEXISTENT_IMAGE_URL:
        bad_ids.append(img.input_id)
    self.assertEqual(len(bad_ids), 2)

    self.app.wait_until_inputs_upload_finish(max_wait=30)

    # Mixed status exception will be raised
    found_error = False
    found_error_id = False
    for img_one in self.app.inputs.get_all():
      if img_one.status.code != 30000:
        found_error = True
        if img_one.input_id in bad_ids:
          found_error_id = True
    self.assertTrue(found_error)
    self.assertTrue(found_error_id)

    # no exception will be raised by default
    found_error = False
    for img_one in self.app.inputs.get_all(ignore_error=True):
      if img_one.status.code != 30000:
        found_error = True
    self.assertFalse(found_error)
