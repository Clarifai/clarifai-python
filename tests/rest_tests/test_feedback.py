# -*- coding: utf-8 -*-
"""
Test for feedback API endpoints
"""

import logging
import unittest

from clarifai.rest import (BoundingBox, ClarifaiApp, Concept, Face, FaceAgeAppearance,
                           FaceGenderAppearance, FaceIdentity, FaceMulticulturalAppearance,
                           FeedbackInfo, FeedbackType, Region, RegionInfo)

from . import sample_inputs


class TestFeedback(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  def test_send_concept_feedback(self):
    """ send concept feedback to models """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='sessionSS',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    ret = m.send_concept_feedback(
        input_id='bb',
        url=sample_inputs.DOG2_IMAGE_URL,
        concepts=['dog', 'puppy'],
        not_concepts=['cat', 'tiger'],
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    ret = m.send_concept_feedback(
        input_id='bb',
        url=sample_inputs.DOG2_IMAGE_URL,
        concepts=['dog', 'puppy'],
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    ret = m.send_concept_feedback(
        input_id='bb',
        url=sample_inputs.DOG2_IMAGE_URL,
        not_concepts=['cat'],
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_region_feedback(self):
    """ send region feedback with concepts """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    concepts = [Concept(concept_id='ab', value=False)]
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8), feedback_type='accurate'),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    concepts = [Concept(concept_id='ab', value=False), Concept(concept_id='bb', value=True)]
    regions = [
        Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), concepts=concepts),
        Region(RegionInfo(bbox=BoundingBox(0.1, 0.2, 0.8, 0.85)), concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_different_region_feedbacks(self):
    """ send different kind of region feedbacks
        - accurate
        - misplaced
        - not_detected
        - false_positive
    """
    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    # positive
    concepts = [Concept(concept_id='ab', value=False)]
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8), feedback_type='accurate'),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # misplaced
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.8, 0.9), feedback_type='misplaced'),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # misplaced
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.1, 0.2, 0.4, 0.5), feedback_type='not_detected'),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # false_positive
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.1, 0.2, 0.4, 0.5), feedback_type='false_positive'),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_different_region_feedbacks_with_enum_vars(self):
    """ send different kind of region feedbacks with enums
        - accurate
        - misplaced
        - not_detected
        - false_positive
    """
    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    # positive
    concepts = [Concept(concept_id='ab', value=False)]
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8), feedback_type=FeedbackType.accurate),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # misplaced
    regions = [
        Region(
            RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.8, 0.9), feedback_type=FeedbackType.misplaced),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # misplaced
    regions = [
        Region(
            RegionInfo(
                bbox=BoundingBox(0.1, 0.2, 0.4, 0.5), feedback_type=FeedbackType.not_detected),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # false_positive
    regions = [
        Region(
            RegionInfo(
                bbox=BoundingBox(0.1, 0.2, 0.4, 0.5), feedback_type=FeedbackType.false_positive),
            concepts=concepts)
    ]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        concepts=['matid2'],
        not_concepts=['lambo'],
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_age_feedback(self):
    """ send face feedback """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    identities = [Concept(concept_id='xx', value=True), Concept(concept_id='x2', value=False)]
    ages = [Concept(concept_id='1', value=True)]
    face = Face(identity=FaceIdentity(identities), age_appearance=FaceAgeAppearance(ages))

    regions = [Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), face=face)]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

    # send multiple items
    identities = [
        Concept(concept_id='xx', value=True),
        Concept(concept_id='x2', value=False),
        Concept(concept_id='x4', value=True)
    ]
    ages = [Concept(concept_id='1', value=True), Concept(concept_id='2', value=False)]
    face = Face(identity=FaceIdentity(identities), age_appearance=FaceAgeAppearance(ages))

    regions = [Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), face=face)]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_gender_feedback(self):
    """ send face feedback """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    identities = [Concept(concept_id='xx', value=True), Concept(concept_id='x2', value=False)]
    genders = [Concept(concept_id='male', value=True), Concept(concept_id='female', value=False)]
    face = Face(identity=FaceIdentity(identities), gender_appearance=FaceGenderAppearance(genders))

    regions = [Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), face=face)]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_multicultural_feedback(self):
    """ send face feedback """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    identities = [Concept(concept_id='xx', value=True), Concept(concept_id='x2', value=False)]
    cultures = [
        Concept(concept_id='american', value=True),
        Concept(concept_id='asian', value=False)
    ]
    face = Face(
        identity=FaceIdentity(identities),
        multicultural_appearance=FaceMulticulturalAppearance(cultures))

    regions = [Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), face=face)]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_complete_multicultural_feedback(self):
    """ send face feedback """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='annotation',
        output_id='oooooooid')

    m = self.app.models.get('general-v1.3')

    identities = [Concept(concept_id='xx', value=True), Concept(concept_id='x2', value=False)]
    ages = [Concept(concept_id='1', value=True), Concept(concept_id='2', value=False)]
    genders = [Concept(concept_id='male', value=True), Concept(concept_id='female', value=False)]
    cultures = [
        Concept(concept_id='american', value=True),
        Concept(concept_id='asian', value=False)
    ]

    face = Face(
        identity=FaceIdentity(identities),
        age_appearance=FaceAgeAppearance(ages),
        gender_appearance=FaceGenderAppearance(genders),
        multicultural_appearance=FaceMulticulturalAppearance(cultures))

    regions = [Region(RegionInfo(bbox=BoundingBox(0.3, 0.2, 0.7, 0.8)), face=face)]

    ret = m.send_region_feedback(
        input_id='xyz',
        url=sample_inputs.DOG_TIFF_IMAGE_URL,
        regions=regions,
        feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)

  def test_send_search_feedback(self):
    """ send search feedback """

    feedback_info = FeedbackInfo(
        end_user_id='robert_python_test_key',
        session_id='from_your_browser',
        event_type='search_click',
        search_id='searchID')

    ret = self.app.inputs.send_search_feedback(input_id='xyz', feedback_info=feedback_info)
    self.assertEqual(ret['status']['code'], 10000)
