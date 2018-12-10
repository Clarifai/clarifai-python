# -*- coding: utf-8 -*-

import base64
import logging
import os
import tempfile
import unittest
import uuid

import clarifai.rest
from clarifai.rest import ClarifaiApp, Geo, GeoBox, GeoLimit, GeoPoint

from . import sample_inputs


class TestSearch(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  @classmethod
  def tearDownClass(cls):
    """ Cleanup """

  def test_search_annotated_concepts(self):
    """ test search by annotated concepts only """

    self.app.inputs.search_by_annotated_concepts(concept='cat')
    self.app.inputs.search_by_annotated_concepts(concepts=['cat'])
    self.app.inputs.search_by_annotated_concepts(concept_id='ai_mFqxrph2')
    self.app.inputs.search_by_annotated_concepts(concept_ids=['ai_mFqxrph2'])

    # go with unicode string
    self.app.inputs.search_by_annotated_concepts(concept=u'cat')
    self.app.inputs.search_by_annotated_concepts(concepts=[u'cat'])
    self.app.inputs.search_by_annotated_concepts(concept_id=u'ai_mFqxrph2')
    self.app.inputs.search_by_annotated_concepts(concept_ids=[u'ai_mFqxrph2'])

    # search with NOT
    self.app.inputs.search_by_annotated_concepts(concept='cat', value=False)
    self.app.inputs.search_by_annotated_concepts(concepts=['cat', 'dog'], values=[False, False])

  def test_search_predicted_concepts(self):
    """ test search by predicted concepts only """

    # upload some cat and dog
    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    self.app.inputs.search_by_predicted_concepts(concept='train')
    self.app.inputs.search_by_predicted_concepts(concepts=['train'])
    self.app.inputs.search_by_predicted_concepts(concept_id='ai_mFqxrph2')
    self.app.inputs.search_by_predicted_concepts(concept_ids=['ai_mFqxrph2'])

    # go with unicode string
    self.app.inputs.search_by_predicted_concepts(concept=u'train')
    self.app.inputs.search_by_predicted_concepts(concepts=[u'train'])
    self.app.inputs.search_by_predicted_concepts(concept_id=u'ai_mFqxrph2')
    self.app.inputs.search_by_predicted_concepts(concept_ids=[u'ai_mFqxrph2'])

    # search with NOT
    self.app.inputs.search_by_predicted_concepts(concept='train', value=False)
    self.app.inputs.search_by_predicted_concepts(
        concepts=['train', 'railway'], values=[False, False])
    self.app.inputs.search_by_predicted_concepts(
        concepts=['train', 'railway'], values=[True, False])
    self.app.inputs.search_by_predicted_concepts(
        concepts=['train', 'railway'], values=[False, True])

    # clean up
    self.app.inputs.delete(img1.input_id)

  def test_search_predicted_concepts_multi_lang(self):
    """ test search by predicted concepts with non-English """

    # upload the metro north
    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    # search in simplified Chinese
    imgs = self.app.inputs.search_by_predicted_concepts(concept=u'铁路列车', lang='zh')
    # disable this until citus migration fully finishes
    # self.assertGreaterEqual(len(list(imgs)), 1)

    imgs = self.app.inputs.search_by_predicted_concepts(concepts=[u'铁路'], lang='zh')
    # disable this until citus migration fully finishes
    # self.assertGreaterEqual(len(list(imgs)), 1)

    # search in Japanese
    imgs = self.app.inputs.search_by_predicted_concepts(concept=u'列車', lang='ja')
    # disable this until citus migration fully finishes
    # self.assertGreaterEqual(len(list(imgs)), 1)

    imgs = self.app.inputs.search_by_predicted_concepts(concepts=[u'鉄道'], lang='ja')
    # disable this until citus migration fully finishes
    # self.assertGreaterEqual(len(list(imgs)), 1)

    # clean up
    self.app.inputs.delete(img1.input_id)

  def test_search_by_image_url(self):
    """ test search by image url """

    # search by image url
    self.app.inputs.search_by_image(url=sample_inputs.METRO_IMAGE_URL)

    # search by image url with leading space
    self.app.inputs.search_by_image(url=' ' + sample_inputs.METRO_IMAGE_URL)

    # search by image url with trailing space
    self.app.inputs.search_by_image(url=sample_inputs.METRO_IMAGE_URL + ' ')

  def test_search_by_image_id(self):
    """ test search by image id """

    # search by exising input id
    image = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, allow_duplicate_url=True)

    self.app.inputs.search_by_image(image_id=image.input_id)

    # clean up
    self.app.inputs.delete(image.input_id)

  def test_search_by_image_bytes(self):
    """ test search by image bytes """

    # search by raw bytes
    data = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content
    self.app.inputs.search_by_image(imgbytes=data)

  def test_search_by_image_base64(self):
    """ test search by image base64 bytes """

    # search by base64 bytes
    data = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content
    base64_data = base64.b64encode(data)
    self.app.inputs.search_by_image(base64bytes=base64_data)

  def test_search_by_image_file(self):
    """ test search by image filename and fileobj"""

    data = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    f.write(data)
    f.close()

    self.app.inputs.search_by_image(filename=filename)

    fio = open(filename, 'rb')
    self.app.inputs.search_by_image(fileobj=fio)
    fio.close()
    os.unlink(filename)

  def test_search_input_original_url(self):
    """ search input original url """

    # search by url
    self.app.inputs.search_by_original_url(sample_inputs.METRO_IMAGE_URL)
    self.app.inputs.search_by_original_url(sample_inputs.WEDDING_IMAGE_URL)

  def test_search_input_metadata(self):
    """ search input meta data """

    # upload image
    image_id = uuid.uuid4().hex
    meta = {'key': 'value'}
    res = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        metadata=meta,
        allow_duplicate_url=True)

    # simple meta search
    search_res = self.app.inputs.search_by_metadata({"key": "value"})
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    search_res = self.app.inputs.search_by_metadata({"key": "value", "key22": "value22"})
    self.assertEqual(len(search_res), 0)

    # nested meta search
    search_res = self.app.inputs.search_by_metadata({"key1": {"key2": {"key3": "value_level34"}}})
    self.assertEqual(len(search_res), 0)

    self.app.inputs.delete(image_id)

  def test_search_input_nested_metadata(self):
    """ search input nested meta data """

    # upload image
    image_id = uuid.uuid4().hex
    meta = {'key': 'value', 'key2': 'value2', 'key3': 'value3'}
    img1 = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        metadata=meta,
        allow_duplicate_url=True)

    # simple meta search
    search_res = self.app.inputs.search_by_metadata({"key": "value", "key2": "value2"})
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    # nested meta search
    search_res = self.app.inputs.search_by_metadata({"key1": {"key2": {"key3": "value_level3"}}})
    self.assertEqual(len(search_res), 0)

    # delete first test image
    self.app.inputs.delete(img1.input_id)

    # upload 2nd image
    image_id = uuid.uuid4().hex
    meta = {"key1": {"key2": {"key3": "value_level3", "key3.2": "value3.2"}, "key2.2": "value2.2"}}
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        metadata=meta,
        allow_duplicate_url=True)

    # nested meta search
    search_res = self.app.inputs.search_by_metadata({"key1": {"key2": {"key3": "value_level3"}}})
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    # delete second test image
    self.app.inputs.delete(img2.input_id)

  def test_search_geo_point(self):
    """ search input geo point data """

    # upload image
    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(40.7128, 74.0059))
    img1 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, geo=geo, allow_duplicate_url=True)

    # simple geo search
    search_res = self.app.inputs.search_by_geo(GeoPoint(40.7128, 74.0059))
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    search_res = self.app.inputs.search_by_geo(GeoPoint(40.7128, 74.0059), GeoLimit("mile", 10))
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    # delete first test image
    self.app.inputs.delete(img1.input_id)

    # upload 2nd image
    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(40.7129, 74.0058))
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, geo=geo, allow_duplicate_url=True)

    search_res = self.app.inputs.search_by_geo(
        GeoPoint(40.7128, 74.0059), GeoLimit("kilometer", 9))
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    # delete second test image
    self.app.inputs.delete(img2.input_id)

  def test_search_geo_box(self):
    """ search input geo point data """

    # upload an image
    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(40.7129, 74.0058))
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, geo=geo, allow_duplicate_url=True)

    geo_box = GeoBox(GeoPoint(40.7028, 74.0009), GeoPoint(40.7328, 74.0359))
    search_res = self.app.inputs.search_by_geo(geo_box=geo_box)
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)

    # delete second test image
    self.app.inputs.delete(img2.input_id)

  def test_search_image_url_and_geo(self):
    """ search over image url combined with geo """

    # upload an image
    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(40.7129, 74.0058))
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id, url=sample_inputs.METRO_IMAGE_URL, geo=geo, allow_duplicate_url=True)

    term1 = clarifai.rest.InputSearchTerm(url=sample_inputs.METRO_IMAGE_URL)
    geo_box = GeoBox(GeoPoint(40.7028, 74.0009), GeoPoint(40.7328, 74.0359))
    term2 = clarifai.rest.InputSearchTerm(geo=Geo(geo_box=geo_box))
    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)

    search_res = self.app.inputs.search(query)
    self.assertGreater(len(search_res), 0)
    match = False
    for img in search_res:
      if img.input_id == image_id or img.url == img2.url:
        match = True
        break
    self.assertTrue(match)

    # delete second test image
    self.app.inputs.delete(img2.input_id)

  def test_search_geo_with_input_output_tag(self):
    """ search over input and output tag together with geo info """

    # upload an image
    image_id = uuid.uuid4().hex
    geo = Geo(GeoPoint(40.7129, 74.0058))
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        geo=geo,
        concepts=["train"],
        allow_duplicate_url=True)

    geo_box = GeoBox(GeoPoint(40.7028, 74.0009), GeoPoint(40.7328, 74.0359))
    term1 = clarifai.rest.InputSearchTerm(geo=Geo(geo_box=geo_box))
    term2 = clarifai.rest.InputSearchTerm(concept="train")
    term3 = clarifai.rest.OutputSearchTerm(concept="railway")
    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)
    query.add_term(term3)

    search_res = self.app.inputs.search(query, page=1, per_page=20)
    # disable this until citus migration fully finishes
    # self.assertGreater(len(search_res), 0)
    # match = False
    # for img in search_res:
    #  if img.input_id == image_id or img.url == img2.url:
    #    match = True
    #    break
    # self.assertTrue(match)

    # delete second test image
    self.app.inputs.delete(img2.input_id)

  def test_input_query_term(self):
    """ test input query term, same as search_by_annotated_concepts """

    term1 = clarifai.rest.InputSearchTerm(concept='cat')

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)

    self.app.inputs.search(query)

  def test_output_query_term(self):
    """ test output query term, same as search_by_predicted_concepts """

    term1 = clarifai.rest.OutputSearchTerm(concept='train')

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)

    self.app.inputs.search(query)

  def test_search_score(self):
    """ test a query, verify the score is available and is descending """

    # upload an image
    image_id = uuid.uuid4().hex
    img1 = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.METRO_IMAGE_URL,
        concepts=["train"],
        allow_duplicate_url=True)

    image_id = uuid.uuid4().hex
    img2 = self.app.inputs.create_image_from_url(
        image_id=image_id,
        url=sample_inputs.WEDDING_IMAGE_URL,
        concepts=["train"],
        allow_duplicate_url=True)

    # search input term and verify the scores are always 1
    term1 = clarifai.rest.InputSearchTerm(concept="train")
    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)

    res = self.app.inputs.search(query)

    # disable this until citus migration fully finishes
    # self.assertGreaterEqual(len(res), 2)
    scores = [one_image.score for one_image in res]
    self.assertEqual(scores, [1] * len(res))

    # search output term and verify the scores are descending
    term1 = clarifai.rest.OutputSearchTerm(concept="train")
    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)

    res = self.app.inputs.search(query)

    # self.assertGreaterEqual(len(res), 2)
    # scores = [one_image.score for one_image in res]
    # self.assertEqual(scores, sorted(scores, reverse=True))

    # clean up
    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_search_by_annotated_and_predicted_concepts(self):
    """ more complex query, with both input and output concept search """

    # upload some cat and dog
    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, concepts=['trains'], allow_duplicate_url=True)

    term1 = clarifai.rest.InputSearchTerm(concept='trains')
    term2 = clarifai.rest.OutputSearchTerm(concept='platform', value=False)
    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)

    self.app.inputs.search(query)

    # delete them
    self.app.inputs.delete(img1.input_id)

  def test_complex_combinations(self):
    """ with more complex queries """

    # upload some cat and dog
    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.WEDDING_IMAGE_URL, concepts=['cat'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        sample_inputs.DOG_TIFF_IMAGE_URL, allow_duplicate_url=True)

    term1 = clarifai.rest.InputSearchTerm(concept='cat')
    term2 = clarifai.rest.OutputSearchTerm(concept='ceremony', value=True)
    term3 = clarifai.rest.OutputSearchTerm(url=sample_inputs.FACEBOOK_IMAGE_URL)

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)
    query.add_term(term3)

    self.app.inputs.search(query)

    # more with all the input search terms
    term1 = clarifai.rest.InputSearchTerm(concept='cat')
    term2 = clarifai.rest.InputSearchTerm(concept='dog', value=False)
    term3 = clarifai.rest.InputSearchTerm(metadata={'name': 'value'})

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)
    query.add_term(term3)

    self.app.inputs.search(query)

    # more with all the output search terms
    term1 = clarifai.rest.OutputSearchTerm(concept='group', value=False)
    term2 = clarifai.rest.OutputSearchTerm(concept='ceremony', value=True)
    term3 = clarifai.rest.OutputSearchTerm(url=sample_inputs.WEDDING_IMAGE_URL)

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)
    query.add_term(term3)

    self.app.inputs.search(query)

    # more with all the search term combination
    term1 = clarifai.rest.InputSearchTerm(concept='cat')
    term2 = clarifai.rest.InputSearchTerm(concept='dog', value=False)
    term3 = clarifai.rest.InputSearchTerm(metadata={'name': 'value'})
    term4 = clarifai.rest.OutputSearchTerm(concept='group', value=False)
    term5 = clarifai.rest.OutputSearchTerm(concept='ceremony', value=True)
    term6 = clarifai.rest.OutputSearchTerm(url=sample_inputs.WEDDING_IMAGE_URL)

    query = clarifai.rest.SearchQueryBuilder()
    query.add_term(term1)
    query.add_term(term2)
    query.add_term(term3)
    query.add_term(term4)
    query.add_term(term5)
    query.add_term(term6)

    self.app.inputs.search(query)

    # delete them
    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)
