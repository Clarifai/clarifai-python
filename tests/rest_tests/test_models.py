# -*- coding: utf-8 -*-
import base64
import logging
import os
import tempfile
import time
import unittest
import uuid

from mock import Mock

from clarifai.rest import ApiError, ClarifaiApp, Image, Model

from . import sample_inputs


class TestModels(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.app = ClarifaiApp(log_level=logging.WARN)

  @classmethod
  def tearDownClass(cls):
    """ Cleanup """

  def test_get_all_models(self):
    """ test get the list of models """

    # test the get_all generator
    models = self.app.models.get_all()

    for model in models:
      self.assertTrue(isinstance(model, Model))

      self.assertTrue(hasattr(model, 'model_id'))
      self.assertTrue(hasattr(model, 'model_version'))

      self.assertIsNotNone(model.model_id)

    # test the fetch by page
    models = self.app.models.get_by_page()

    for model in models:
      self.assertTrue(isinstance(model, Model))

      self.assertTrue(hasattr(model, 'model_id'))
      self.assertTrue(hasattr(model, 'model_version'))

      self.assertIsNotNone(model.model_id)

  def test_get_all_public_models(self):
    """ make sure no private models are fetched """

    # create a model with no concept
    model_id = uuid.uuid4().hex
    model1 = self.app.models.create(model_id)
    model_id_retrieved = model1.model_id
    self.assertEqual(model_id, model_id_retrieved)

    # test the get_all generator
    models = self.app.models.get_all(public_only=True)

    for model in models:
      self.assertTrue(isinstance(model, Model))
      self.assertTrue(model.app_id == '' or model.app_id == 'main')

    # clean up
    self.app.models.delete(model_id)

  def test_get_all_private_models(self):
    """ make sure no public models are fetched """

    # create a model with no concept
    model_id = uuid.uuid4().hex
    model1 = self.app.models.create(model_id)
    model_id_retrieved = model1.model_id
    self.assertEqual(model_id, model_id_retrieved)

    # test the get_all generator
    models = self.app.models.get_all(private_only=True)

    self.assertGreaterEqual(len(list(models)), 1)

    for model in models:
      self.assertTrue(isinstance(model, Model))
      self.assertTrue(model.app_id != '' and model.app_id != 'main')

    # clean up
    self.app.models.delete(model_id)

  def test_get_model_by_id(self):
    """ get model by model id """

    for model in self.app.models.get_all():

      model_id = model.model_id

      try:
        model_fetch = self.app.models.get(model_id)
      except ApiError as e:
        if e.response.status_code == 404:
          # in the concurrent tests, this may get some transient models that will be
          # deleted right away
          # so just ignore the 404 errors
          continue
        else:
          raise e

      self.assertTrue(isinstance(model_fetch, Model))
      self.assertEqual(model_id, model_fetch.model_id)

  def test_get_model_by_id_multi_lang(self):
    """ get model by model id in other language """

    new_model_id = u'我的新模型'
    try:
      m = self.app.models.get(new_model_id)
    except ApiError as e:
      if e.response.status_code == 404:
        m = self.app.models.create(new_model_id)
        self.assertEqual(m.model_id, new_model_id)

    model_fetch = self.app.models.get(new_model_id)
    self.assertTrue(isinstance(model_fetch, Model))
    self.assertEqual(new_model_id, model_fetch.model_id)

  def test_get_model_by_id_slash(self):
    """ get model by model id with slash """

    new_model_id = 'the/model'
    try:
      m = self.app.models.get(new_model_id)
    except ApiError as e:
      if e.response.status_code == 404:
        m = self.app.models.create(new_model_id)
        self.assertEqual(m.model_id, new_model_id)

    model_fetch = self.app.models.get(new_model_id)
    # use model
    self.app.models.get(new_model_id).get_info()
    self.assertTrue(isinstance(model_fetch, Model))
    self.assertEqual(new_model_id, model_fetch.model_id)

  def test_get_model_output_info_by_id(self):
    """ get model output info by model id """

    for model in self.app.models.get_all():

      model_id = model.model_id

      try:
        model_fetch = self.app.models.get(model_id)
        ret = model_fetch.get_info(verbose=True)
      except ApiError as e:
        if e.response.status_code == 404:
          # in the concurrent tests, this may get some transient models that will be
          # deleted right away
          # so just ignore the 404 errors
          continue
        else:
          raise e

      self.assertIn('model', ret)
      self.assertIn('output_info', ret['model'])
      if 'data' in ret['model']['output_info']:
        self.assertIn('concepts', ret['model']['output_info']['data'])
        self.assertNotEqual(0, len(ret['model']['output_info']['data']['concepts']))

      self.assertIn('type', ret['model']['output_info'])

  def test_get_model_by_id_and_version(self):
    """ get model by model id and version """

    models = self.app.models.get_all()

    for cnt, model in enumerate(models):

      if cnt > 3:
        break

      try:
        versions = model.list_versions()
      except ApiError as e:
        if e.response.status_code == 404:
          # in the concurrent tests, this may get some transient models that will be
          # deleted right away
          # so just ignore the 404 errors
          continue
        else:
          raise e

      for v_cnt, version in enumerate(versions['model_versions']):

        if v_cnt > 3:
          break

        version_id = version['id']

        try:
          v = model.get_version(version_id)
        except ApiError as e:
          if e.response.status_code == 404:
            # in the concurrent tests, this may get some transient models that will
            # be deleted right away
            # so just ignore the 404 errors
            continue
          else:
            raise e

        self.assertIn('model_version', v)
        self.assertIn('id', v['model_version'])
        self.assertIn('created_at', v['model_version'])
        self.assertIn('status', v['model_version'])

  def test_predict(self):
    """ predict with various image format """

    model = self.app.models.get('general-v1.3')

    # predict by url
    model.predict_by_url(sample_inputs.METRO_IMAGE_URL)

    # predict with url with leading space
    model.predict_by_url('  ' + sample_inputs.METRO_IMAGE_URL)

    # predict with url with trailing space
    model.predict_by_url('  ' + sample_inputs.METRO_IMAGE_URL + ' ')

    # predict by file raw bytes
    raw_bytes = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content
    model.predict_by_bytes(raw_bytes)

    # predict by base64 bytes
    base64_bytes = base64.b64encode(raw_bytes)
    model.predict_by_base64(base64_bytes)

    # predict by local filename
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    f.write(raw_bytes)
    f.close()

    model.predict_by_filename(filename)

    os.unlink(filename)

  def test_get_model_with_id(self):
    m = self.app.models.get(model_id='abcdefg')
    self.assertTrue(isinstance(m, Model))
    self.assertEqual(m.model_id, 'abcdefg')

  def test_get_model_with_name(self):
    m = self.app.models.get('general-v1.3')
    self.assertTrue(isinstance(m, Model))

  def test_get_model_with_id_directly(self):
    m = self.app.models.get('aaa03c23b3724a16a56b629203edc62c')
    self.assertTrue(isinstance(m, Model))
    self.assertEqual(m.model_id, 'aaa03c23b3724a16a56b629203edc62c')

  def test_predict_with_model_id(self):
    """ test initialize model object """
    # make model from model_id
    m = Model(self.app.api, model_id='eee28c313d69466f836ab83287a54ed9')
    m.predict_by_url(sample_inputs.METRO_IMAGE_URL)

  def test_predict_multiple_times_single_image(self):
    """ construct one image object and predict multiple times """
    model = self.app.models.get('general-v1.3')

    # predict by url
    f = tempfile.NamedTemporaryFile(delete=False)
    filename = f.name
    raw_bytes = self.app.api.session.get(sample_inputs.METRO_IMAGE_URL).content
    f.write(raw_bytes)
    f.close()

    img = Image(filename=filename)

    res = model.predict([img, img])
    self.assertEqual(10000, res['status']['code'])

    res = model.predict([img, img])
    self.assertEqual(10000, res['status']['code'])

    os.unlink(filename)

  def test_search_model(self):
    """ search model """

    models = self.app.models.search("general-v1.3")

    for model in models:
      self.assertTrue(isinstance(model, Model))

  def test_create_model(self):

    # create a model with no concept
    model_id = uuid.uuid4().hex
    model1 = self.app.models.create(model_id)
    model_id_retrieved = model1.model_id
    self.assertEqual(model_id, model_id_retrieved)
    self.app.models.delete(model_id)

    # create a model with model_id and no concept
    model_id = uuid.uuid4().hex
    model2 = self.app.models.create(model_name="my_model2", model_id=model_id)
    model_id_retrieved = model2.model_id
    self.assertEqual(model_id, model_id_retrieved)
    self.app.models.delete(model_id)

    # create a model with concepts

    img1 = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, concepts=['cat', 'animal'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        url=sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id=model_id, concepts=['cat', 'dog', 'animal'])

    model_id_retrieved = model.model_id
    self.assertEqual(model_id, model_id_retrieved)
    self.app.models.delete(model_id)

    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_create_model_with_hyper_parameters(self):

    # create a model without hyper parameter
    model_id = uuid.uuid4().hex
    model1 = self.app.models.create(model_id, hyper_parameters=None)

    self.assertEqual(model1.model_id, model_id)

    # create a model with hyper parameters
    model_id = uuid.uuid4().hex
    parameters = {
        'MAX_NITEMS': 1000000.0,
        'MIN_NITEMS': 1000,
        'N_EPOCHS': 5,
        'custom_training_cfg': 'custom_training_1layer',
        'custom_training_cfg_args': {}
    }

    model2 = self.app.models.create(model_id, hyper_parameters=parameters)

    self.assertEqual(model2.model_id, model_id)
    self.assertIsNotNone(model2.hyper_parameters)
    self.assertDictEqual(model2.hyper_parameters, parameters)

    m2 = self.app.models.get(model_id)

    self.assertEqual(m2.model_id, model_id)
    self.assertIsNotNone(m2.hyper_parameters)
    self.assertDictEqual(m2.hyper_parameters, parameters)

    # create a model with hyper parameters
    model_id = uuid.uuid4().hex
    parameters = {
        'MAX_NITEMS': 300000,
        'MIN_NITEMS': 2000,
        'N_EPOCHS': 10,
        'custom_training_cfg': 'custom_training_1layer',
        'custom_training_cfg_args': {}
    }

    model3 = self.app.models.create(model_id, hyper_parameters=parameters)

    self.assertEqual(model3.model_id, model_id)
    self.assertIsNotNone(model3.hyper_parameters)
    self.assertDictEqual(model3.hyper_parameters, parameters)

    m3 = self.app.models.get(model_id)

    self.assertEqual(m3.model_id, model_id)
    self.assertIsNotNone(m3.hyper_parameters)
    self.assertDictEqual(m3.hyper_parameters, parameters)

    # clean up
    self.app.models.delete(model1.model_id)
    self.app.models.delete(model2.model_id)
    self.app.models.delete(model3.model_id)

  def test_get_model_concepts(self):
    """ test get concepts from the model """
    img1 = self.app.inputs.create_image_from_url(
        url=sample_inputs.METRO_IMAGE_URL, concepts=['cat', 'animal'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        url=sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id=model_id, concepts=['cat', 'animal', 'dog'])

    ids = model.get_concept_ids()
    self.assertSetEqual(set(ids), set(['cat', 'animal', 'dog']))

    # clean up
    self.app.models.delete(model_id)

    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_train_model(self):
    """ train models """

    # create a model with no concept
    model_id = uuid.uuid4().hex
    model2 = self.app.models.create(model_id=model_id)

    # train well created model with no concept
    with self.assertRaises(ApiError) as ae:
      res = model2.train()
      self.assertEqual(res['status']['code'], 21202)

    # create a model with concepts but no samples
    try:
      self.app.concepts.get('cats1')
    except ApiError:
      self.app.concepts.create('cats1')

    try:
      self.app.concepts.get('dogs1')
    except ApiError:
      self.app.concepts.create('dogs1')

    self.app.models.delete(model_id)

    # train in async way
    model_id = uuid.uuid4().hex
    model2 = self.app.models.create(model_id=model_id, concepts=['cats1', 'dogs1'])
    res = model2.train(sync=False)
    self.assertEqual(res['status']['code'], 10000)
    self.assertEqual(res['model']['model_version']['status']['code'], 21103)
    self.app.models.delete(model_id)

    # train in sync way with no example
    model_id = uuid.uuid4().hex
    model3 = self.app.models.create(model_id=model_id, concepts=['cats1', 'dogs1'])
    res = model3.train()
    self.app.models.delete(model_id)

    # train a good model
    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, concepts=['cat'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model4 = self.app.models.create(model_id=model_id, concepts=['cat', 'dog'])
    res = model4.train()
    self.app.models.delete(model_id)

    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_train_timeout(self):
    ''' train(sync=True) timeout handling '''

    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, concepts=['cat'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id)
    model.add_concepts(['cat', 'dog'])

    # mock the response of res_ver = self.api.get_model_version(model_id, model_version)
    #                      res_ver['model_version']['status']['code']
    start_ts = time.time()
    timeout = 1

    def get_mocked_model_version(model_id, version_id):
      # 21101: being trained
      # 21100: trained

      ret_training = {'model_version': {'id': 'xxxx', 'status': {'code': 21103}}}

      ret_trained = {'model_version': {'id': 'xxxx', 'status': {'code': 21100}}}

      time_now_ts = time.time()
      if time_now_ts - start_ts > timeout:
        return ret_trained
      else:
        return ret_training

    # patch the relevant functions under api endpoint
    model.api = Mock()
    model.api.create_model_version.side_effect = self.app.api.create_model_version
    model.api.get_model_version.side_effect = get_mocked_model_version

    # within timeout
    start_ts = time.time()
    timeout = 3
    ret = model.train()
    self.assertEqual(ret.model_status_code, 21100)

    # call train with 10 sec timeout
    start_ts = time.time()
    timeout = 10
    ret = model.train(timeout=3)
    self.assertEqual(ret.model_status_code, 21103)

    # beyond default timeout
    start_ts = time.time()
    timeout = 100
    ret = model.train()
    self.assertEqual(ret.model_status_code, 21103)

    # recover the patched api
    model.api = self.app.api

    # clean up
    self.app.models.delete(model_id)
    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_delete_model(self):
    # create a model and delete it
    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id)
    self.app.models.delete(model.model_id)

  def test_delete_models(self):
    # create models
    model1 = self.app.models.create(uuid.uuid4().hex)
    model2 = self.app.models.create(uuid.uuid4().hex)

    # assert presence
    all_model_ids = [m.model_id for m in self.app.models.get_all(private_only=True)]
    self.assertIn(model1.model_id, all_model_ids)
    self.assertIn(model2.model_id, all_model_ids)

    # delete models
    delete_response = self.app.models.bulk_delete([model1.model_id, model2.model_id])
    self.assertEqual(delete_response['status']['code'], 10000)
    time.sleep(1)

    # assert not present
    all_model_ids = [m.model_id for m in self.app.models.get_all(private_only=True)]
    self.assertNotIn(model1.model_id, all_model_ids)
    self.assertNotIn(model2.model_id, all_model_ids)

  def test_patch_model_concepts(self):
    """ create a model, add ,delete, and overwrite concepts """

    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, concepts=['cat'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id)
    model.add_concepts(['cat', 'dog'])
    model.delete_concepts(['cat'])
    model.merge_concepts(['cat', 'dog'])
    model.merge_concepts(['cat', 'dog'], overwrite=False)
    model.merge_concepts(['cat', 'dog'], overwrite=True)

    self.app.models.get(model.model_id)

    self.app.models.delete(model_id)
    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_create_and_update_model(self):
    """ add a model with no concept, and then add two concepts, verify with get_model """

    img1 = self.app.inputs.create_image_from_url(
        sample_inputs.METRO_IMAGE_URL, concepts=['cat', 'animal'], allow_duplicate_url=True)
    img2 = self.app.inputs.create_image_from_url(
        sample_inputs.WEDDING_IMAGE_URL, concepts=['dog'], allow_duplicate_url=True)

    model_id = uuid.uuid4().hex
    model = self.app.models.create(model_id)
    model = model.merge_concepts(concept_ids=['cat', 'dog'])
    model = model.delete_concepts(concept_ids=['dog'])

    self.assertEqual(model.model_id, model_id)

    ret2 = model.get_info(verbose=True)
    tags = [one['id'] for one in ret2['model']['output_info']['data']['concepts']]
    self.assertIn('cat', tags)
    self.assertNotIn('dog', tags)

    # modify model attributes
    model = model.update(model_name="new_model_name")
    self.assertTrue(isinstance(model, Model))

    model = model.update(model_name="new_model_name2")
    self.assertTrue(isinstance(model, Model))

    model = model.update(concepts_mutually_exclusive=False)
    self.assertTrue(isinstance(model, Model))
    self.assertFalse(model.concepts_mutually_exclusive)

    model = model.update(concepts_mutually_exclusive=True)
    self.assertTrue(isinstance(model, Model))
    self.assertTrue(model.concepts_mutually_exclusive)

    model = model.update(closed_environment=False)
    self.assertTrue(isinstance(model, Model))
    self.assertFalse(model.closed_environment)

    model = model.update(closed_environment=True)
    self.assertTrue(isinstance(model, Model))
    self.assertTrue(model.closed_environment)

    model = model.update(action='overwrite', concept_ids=['cat'])
    self.assertTrue(isinstance(model, Model))

    model = model.update(action='overwrite', concept_ids=[])
    self.assertTrue(isinstance(model, Model))

    model = model.update(
        model_name="new_model_name3", concepts_mutually_exclusive=False, concept_ids=['cat'])
    self.assertTrue(isinstance(model, Model))

    model = model.update()
    self.assertTrue(isinstance(model, Model))

    # clean up
    self.app.models.delete(model_id)

    self.app.inputs.delete(img1.input_id)
    self.app.inputs.delete(img2.input_id)

  def test_predict_model(self):
    """ test predict with general model """

    # model = self.app.models.get('general-v1.3')
    model = self.app.models.get(model_id='aaa03c23b3724a16a56b629203edc62c')
    image = Image(url=sample_inputs.METRO_IMAGE_URL)

    res = model.predict([image])
    self.assertEqual(10000, res['status']['code'])

    # model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(min_value=0.96))
    # model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(min_value=0.96))
    # res = model.predict(inputs=[image], model_output_info=model_output_info)

  def test_model_name_cache(self):
    """ test model name cache
        by quering model, only first time should go to the remote api
    """

    self.app.models.clear_model_cache()
    self.assertFalse(self.app.models.model_id_cache)

    model = self.app.models.get('face')

    self.assertTrue(self.app.models.model_id_cache)
    self.assertTrue(self.app.models.model_id_cache.get('face', 'facedetect'))

    model2 = self.app.models.get('face', model_type='embed')
    self.assertTrue(self.app.models.model_id_cache.get('face', 'embed'))
    self.assertNotEqual(model.model_id, model2.model_id)

    model3 = self.app.models.get('general-v1.3')
    self.assertTrue(self.app.models.model_id_cache.get('general-v1.3', 'concept'))
    self.assertEqual(model3.model_name, 'general-v1.3')

    # This introduces a race condition below, so the above check that it is just in the
    # cache should be good enough.
    # time_start = time.time()
    # model = self.app.models.get('general-v1.3')
    # time_elapsed2 = time.time() - time_start

    # # assume the 2nd fetch with cache is always faster than the first one
    # self.assertTrue(time_elapsed2 < time_elapsed1)

  def test_model_cache_init(self):
    """ test model cache init
        make sure the model cache is not empty after the app is initialized
    """

    app2 = ClarifaiApp(log_level=logging.WARN)

    model_cache = app2.models.model_id_cache

    # model_cache not empty
    self.assertNotEqual(model_cache, {})

    outstanding_public_models = ['general-v1.3', 'color', 'nsfw-v1.0']

    cached_model_names = [m[0] for m in model_cache.keys()]

    # model cache has the outstanding models cached
    self.assertIsNotNone(set(cached_model_names).difference(outstanding_public_models))

    del app2

  def test_predict_with_i18n(self):
    """ test general prediction with a few non-English languages
    """

    # go with simplified Chinese
    model = self.app.models.get('general-v1.3')

    res = model.predict_by_url(sample_inputs.METRO_IMAGE_URL, lang='zh')
    concepts = res['outputs'][0]['data']['concepts']
    self.assertTrue(isinstance(concepts, list))

    concept1 = concepts[0]
    self.assertEqual(concept1['id'], u'ai_HLmqFqBf')
    self.assertEqual(concept1['name'], u'铁路列车')

    concept2 = concepts[1]
    self.assertEqual(concept2['id'], u'ai_fvlBqXZR')
    self.assertEqual(concept2['name'], u'铁路')

    # go with Japanese
    res = model.predict_by_url(sample_inputs.METRO_IMAGE_URL, lang='ja')
    concepts = res['outputs'][0]['data']['concepts']
    self.assertTrue(isinstance(concepts, list))

    concept1 = concepts[0]
    self.assertEqual(concept1['id'], u'ai_HLmqFqBf')
    self.assertEqual(concept1['name'], u'列車')

    concept2 = concepts[1]
    self.assertEqual(concept2['id'], u'ai_fvlBqXZR')
    self.assertEqual(concept2['name'], u'鉄道')

    # go with Russian
    res = model.predict_by_url(sample_inputs.METRO_IMAGE_URL, lang='ru')
    concepts = res['outputs'][0]['data']['concepts']
    self.assertTrue(isinstance(concepts, list))

    concept1 = concepts[0]
    self.assertEqual(concept1['id'], u'ai_HLmqFqBf')
    self.assertEqual(concept1['name'], u'поезд')

    concept2 = concepts[1]
    self.assertEqual(concept2['id'], u'ai_fvlBqXZR')
    self.assertEqual(concept2['name'], u'железная дорога')

  def test_model_evaluate(self):
    model_id = uuid.uuid4().hex
    image_id1 = uuid.uuid4().hex
    image_id2 = uuid.uuid4().hex

    try:
      # Create inputs.
      self.app.inputs.create_image_from_url(
          image_id=image_id1,
          url=sample_inputs.METRO_IMAGE_URL,
          concepts=['cat', 'animal'],
          allow_duplicate_url=True)
      self.app.inputs.create_image_from_url(
          image_id=image_id2,
          url=sample_inputs.WEDDING_IMAGE_URL,
          concepts=['dog'],
          allow_duplicate_url=True)

      # Create and train a model.
      model = self.app.models.create(model_id=model_id, concepts=['cat', 'dog', 'animal'])
      model = model.train(timeout=1200)

      # Run model evaluation and assert the response.
      res = model.evaluate()
      self.assertIsNotNone(res['model_version'])
      self.assertIsNotNone(res['model_version']['status'])
      self.assertIsNotNone(res['model_version']['status']['code'])
      self.assertIsNotNone(res['model_version']['metrics'])
      self.assertIsNotNone(res['model_version']['metrics']['status'])
      self.assertIsNotNone(res['model_version']['metrics']['status']['code'])
    finally:
      # Clean up.
      self.app.models.delete(model_id)

      self.app.inputs.delete(image_id1)
      self.app.inputs.delete(image_id2)

  def test_model_inputs_are_added_and_gettable(self):
    model_id = uuid.uuid4().hex

    image_id1 = uuid.uuid4().hex
    image_id2 = uuid.uuid4().hex

    concept_id_1 = uuid.uuid4().hex
    concept_id_2 = uuid.uuid4().hex

    try:
      # Create inputs.
      self.app.inputs.create_image_from_url(
          image_id=image_id1,
          url=sample_inputs.METRO_IMAGE_URL,
          concepts=[concept_id_1],
          allow_duplicate_url=True)
      self.app.inputs.create_image_from_url(
          image_id=image_id2,
          url=sample_inputs.WEDDING_IMAGE_URL,
          concepts=[concept_id_2],
          allow_duplicate_url=True)

      # Create a model.
      model = self.app.models.create(model_id=model_id, concepts=[concept_id_1, concept_id_2])

      # Get model's inputs. Note: use a large page because there we just list all inputs
      # in app with this call and depending on which test ran before this we might not have
      # image_id1 and image_id2 in the 20 default per_page size.
      response = model.get_inputs(per_page=1000)

      input_ids = [input_['id'] for input_ in response['inputs']]
      assert image_id1 in input_ids
      assert image_id2 in input_ids

    finally:
      # Clean up.
      self.app.models.delete(model_id)

      self.app.inputs.delete(image_id1)
      self.app.inputs.delete(image_id2)

  def test_get_info_with_version(self):
    model = self.app.public_models.general_model
    model.model_version = 'aa9ca48295b37401f8af92ad1af0d91d'

    response = model.get_info(True)
    model_version = response['model']['model_version']
    assert model_version['id'] == model.model_version
