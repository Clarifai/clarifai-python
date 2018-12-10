# -*- coding: utf-8 -*-

import logging
import unittest
import uuid

from clarifai.rest import ApiError, ClarifaiApp, Concept


class TestConcepts(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def _verify_concept(self, obj):
    """ verify if it is a legitimate Concept object """

    self.assertTrue(isinstance(obj, Concept))
    self.assertTrue(hasattr(obj, 'concept_id'))
    self.assertTrue(hasattr(obj, 'concept_name'))
    self.assertTrue(hasattr(obj, 'app_id'))
    self.assertTrue(hasattr(obj, 'created_at'))
    self.assertTrue(hasattr(obj, 'value'))

  def test_concept_object(self):
    """ test concept object """
    con = Concept("my_concept_1")
    self._verify_concept(con)
    data = con.dict()
    self.assertIn('name', data)

    con = Concept("my_concept_1", concept_id='id2', value=0)
    self._verify_concept(con)
    data = con.dict()
    self.assertIn('name', data)
    self.assertIn('id', data)
    self.assertIn('value', data)

  def test_get_all_concepts(self):
    """ test get all concepts """

    # try to iterate all concepts
    ccount = 0
    for concept in self.app.concepts.get_all():
      self._verify_concept(concept)
      ccount += 1
      if ccount >= 50:
        break

    self.assertTrue(ccount > 0)

  def test_get_concepts_by_page(self):
    """ test the pagination of get concepts """

    # default page 1 with 20 items
    concepts = self.app.concepts.get_by_page()
    self.assertTrue(len(concepts) <= 20)

    for concept in concepts:
      self._verify_concept(concept)

    # with less than 20 items per page
    concepts = self.app.concepts.get_by_page(page=2, per_page=5)
    self.assertTrue(len(concepts) <= 5)

    for concept in concepts:
      self._verify_concept(concept)

    concepts = self.app.concepts.get_by_page(page=999999999, per_page=100)
    self.assertEqual(len(concepts), 0)

  def test_get_concept(self):
    """ test get one concept by id """

    concept = self.app.concepts.get('ai_vhvTrLRT')
    self.assertEqual(concept.concept_name, 'affection')

  def test_get_concept_multi_lang(self):
    """ test get one concept by id when it's not English """

    # add a concept with only ID
    new_cid = u'test_狗'
    try:
      concept = self.app.concepts.create(concept_id=new_cid)
    except ApiError as e:
      if e.response.status_code == 400:
        concept = self.app.concepts.get(new_cid)
      else:
        raise e

    self.assertEqual(concept.concept_id, new_cid)

    # try to get a few public model concepts
    concept = self.app.concepts.get(new_cid)
    self.assertEqual(concept.concept_name, new_cid)

    # add a concept with only ID
    new_cid = u'的な'
    try:
      concept = self.app.concepts.create(concept_id=new_cid)
    except ApiError as e:
      if e.response.status_code == 400:
        concept = self.app.concepts.get(new_cid)
      else:
        raise e

    self.assertEqual(concept.concept_id, new_cid)

    # try to get a few public model concepts
    concept = self.app.concepts.get(new_cid)
    self.assertEqual(concept.concept_name, new_cid)

  def test_add_concept_with_id(self):
    """ test add a concept with only ID"""

    new_cid = 'test_' + uuid.uuid4().hex
    concept = self.app.concepts.create(concept_id=new_cid)
    self.assertEqual(concept.concept_id, new_cid)

    cnew = self.app.concepts.get(new_cid)
    self.assertEqual(cnew.concept_id, new_cid)
    self.assertEqual(cnew.concept_name, new_cid)

  def test_add_concept_with_id_and_name(self):
    """ test add a concept with ID and name"""

    # add a concept with ID and name
    new_cid = 'test_' + uuid.uuid4().hex
    new_cname = 'test_name_' + uuid.uuid4().hex
    concept = self.app.concepts.create(concept_id=new_cid, concept_name=new_cname)
    self.assertEqual(concept.concept_id, new_cid)

    cnew = self.app.concepts.get(new_cid)
    self.assertEqual(cnew.concept_id, new_cid)
    self.assertEqual(cnew.concept_name, new_cname)

  def test_add_concept_should_raise_on_duplicate_id(self):
    """ test raise on duplicate ID """

    new_cid = 'test_' + uuid.uuid4().hex

    self.app.concepts.create(concept_id=new_cid)

    # ERROR: add a duplicate concept id
    with self.assertRaises(ApiError):
      self.app.concepts.create(concept_id=new_cid)

  def test_add_concepts(self):
    """ test add a few concepts """

    # add a concept with only ID

    concept_ids = ['test_' + uuid.uuid4().hex for _ in range(10)]
    concept_names = ['name_' + uuid.uuid4().hex for _ in range(10)]

    concepts = self.app.concepts.bulk_create(concept_ids, concept_names)
    self.assertEqual(len(concepts), 10)

    for concept in concepts:
      self.assertTrue(concept.concept_id.startswith('test_'))
      self.assertTrue(concept.concept_name.startswith('name_'))

  def test_search_concepts(self):
    """ test search concept """
    concepts = self.app.concepts.search('dog*')
    self.assertGreaterEqual(len(list(concepts)), 2)

  def test_search_concepts_with_explicit_en_lang(self):
    """ test search concepts with English language set explicitly """
    concepts = self.app.concepts.search('dog*', lang='en')
    self.assertGreaterEqual(len(list(concepts)), 2)

  def test_search_concepts_with_zh_lang(self):
    """ test search concepts using non-English language """
    concepts = self.app.concepts.search(u'狗*', lang='zh')
    self.assertGreaterEqual(len(list(concepts)), 3)

  def test_update_one_concept(self):
    """ update one concept """

    # add a concept with only ID
    try:
      new_cid = 'test_' + uuid.uuid4().hex
      concept = self.app.concepts.create(concept_id=new_cid)
      self.assertEqual(concept.concept_id, new_cid)
    except ApiError as e:
      if e.response.status_code == 400 and e.error_code == 40003:
        # We already have this concept in db
        pass
      else:
        raise e

    new_name = 'test_new_name' + uuid.uuid4().hex
    concept = self.app.concepts.update(concept_id=new_cid, concept_name=new_name)

    self.assertEqual(concept.concept_id, new_cid)
    self.assertEqual(concept.concept_name, new_name)

  def test_update_more_concepts(self):
    """ update more than one concept """

    # add a concept with only ID
    try:
      new_cid1 = 'test_' + uuid.uuid4().hex
      concept1 = self.app.concepts.create(concept_id=new_cid1)
      self.assertEqual(concept1.concept_id, new_cid1)
    except ApiError as e:
      if e.response.status_code == 400 and e.error_code == 40003:
        # We already have this concept in db
        pass
      else:
        raise e

    try:
      new_cid2 = 'test_' + uuid.uuid4().hex
      concept2 = self.app.concepts.create(concept_id=new_cid2)
      self.assertEqual(concept2.concept_id, new_cid2)
    except ApiError as e:
      if e.response.status_code == 400 and e.error_code == 40003:
        # We already have this concept in db
        pass
      else:
        raise e

    new_name = 'test_new_name' + uuid.uuid4().hex
    concepts = self.app.concepts.bulk_update(
        concept_ids=[new_cid1, new_cid2], concept_names=[new_name] * 2)

    self.assertEqual(concepts[0].concept_id, new_cid1)
    self.assertEqual(concepts[0].concept_name, new_name)
    self.assertEqual(concepts[1].concept_id, new_cid2)
    self.assertEqual(concepts[1].concept_name, new_name)

  def test_updating_public_concept_should_raise(self):
    """ updating a public concept should raise """
    cid = 'ai_98Xb0K3q'
    new_name = 'does_not_matter'

    with self.assertRaises(ApiError):
      self.app.concepts.update(concept_id=cid, concept_name=new_name)
