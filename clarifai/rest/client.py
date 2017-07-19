# -*- coding: utf-8 -*-

"""
Clarifai API Python Client
"""

import os
import time
import json
import copy
import base64
import logging

import requests
import platform
from enum import Enum
from io import BytesIO
from pprint import pformat
from configparser import ConfigParser
from posixpath import join as urljoin
from past.builtins import basestring
from jsonschema import validate
from future.moves.urllib.parse import urlparse
from distutils.version import StrictVersion
from .geo import GeoPoint, GeoBox, GeoLimit, Geo

logger = logging.getLogger('clarifai')
logger.handlers = []
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

logging.getLogger("requests").setLevel(logging.WARNING)

CLIENT_VERSION = '2.0.30'
OS_VER = os.sys.platform
PYTHON_VERSION = '.'.join(map(str, [os.sys.version_info.major, os.sys.version_info.minor, \
                                    os.sys.version_info.micro]))
GITHUB_TAG_ENDPOINT = 'https://api.github.com/repos/clarifai/clarifai-python/git/refs/tags'

DEFAULT_TAG_MODEL = 'general-v1.3'


class ClarifaiApp(object):

  """ Clarifai Application Object

      This is the entry point of the Clarifai Client API.
      With authentication to an application, you can access
      all the models, concepts, and inputs in this application through
      the attributes of this class.

    |  To access the models: use ``app.models``
    |  To access the inputs: use ``app.inputs``
    |  To access the concepts: use ``app.concepts``
    |

  """

  def __init__(self, app_id=None, app_secret=None, base_url=None, api_key=None, quiet=True):

    # check upgrade
    self.check_upgrade()

    self.api = ApiClient(app_id=app_id, app_secret=app_secret, base_url=base_url, api_key=api_key, quiet=quiet)
    self.auth = Auth(self.api)

    self.concepts = Concepts(self.api)
    self.inputs = Inputs(self.api)
    self.models = Models(self.api)

  def check_upgrade(self):
    ''' Check for a client upgrade.
        If the client has been installed for more than one week, the check will be
        triggered.
        If the newer version is available, a prompt message will pop up as a
        warning message in STDERR. The API call will not be paused or interrupted.
    '''

    try:
      # check the latest version
      res = requests.get(GITHUB_TAG_ENDPOINT)

      # ignore the rare github api outage because this is noncritical check
      if res.status_code != 200:
        logger.debug("github.com or network might be down. Please check connectivity.")
        logger.debug("HTTP %d: %s" % (res.status_code, res.content))
        return

      tags = res.json()
      tag_latest = tags[-1]
      tag_latest_release = tag_latest['ref'].split('/')[-1]
      if tag_latest_release.startswith('v'):
        tag_latest_release = tag_latest_release[1:]

      # compare and warn
      if StrictVersion(CLIENT_VERSION) < StrictVersion(tag_latest_release):
        logger.warn("Hey! Clarifai Python Client v%s upgrade available.", tag_latest_release)
    except Exception as e:
      # as this is non critical check, ignore all exceptions that occur
      logger.debug(str(e))
      pass

  """
  Below are the shortcut functions for a more smooth transition of the v1 users
  Also they are convenient functions for the tag only users so they do not have
  to know the extra concepts of Inputs, Models, etc.
  """
  def tag_urls(self, urls, model_name=DEFAULT_TAG_MODEL, model_id=None):
    ''' tag urls with user specified models
        by default tagged by 'general-v1.3' model

    Args:
      urls: a list of URLs for tagging.
            The max length of the list is 128, which is the max batch size.

      model: the model name to tag with.
             The default model is general model for general tagging purpose

    Returns:
      the JSON string from the predict call

    Examples:
      >>> urls = ['https://samples.clarifai.com/metro-north.jpg',
      >>>         'https://samples.clarifai.com/dog2.jpeg']
      >>> app.tag_urls(urls)

    '''

    # validate input
    if not isinstance(urls, list) or (len(urls) > 1 and not isinstance(urls[0], basestring)):
      raise UserError('urls must be a list of string urls')

    if len(urls) > 128:
      raise UserError('max batch size is 128')

    images = [Image(url=url) for url in urls]

    if model_id is not None:
      model = Model(self.api, model_id=model_id)
    else:
      model = self.models.get(model_name)

    res = model.predict(images)
    return res

  def tag_files(self, files, model_name=DEFAULT_TAG_MODEL, model_id=None):
    ''' tag files on disk with user specified models
        by default tagged by 'general-v1.3' model

    Args:
      files: a list of local file names for tagging.
             The max length of the list is 128, which is the max batch size

      model: the model name to tag with.
             The default model is general model for general tagging purpose

    Returns:
      the JSON string from the predict call

    Examples:
      >>> files = ['/tmp/metro-north.jpg',
      >>>          '/tmp/dog2.jpeg']
      >>> app.tag_urls(files)
    '''

    # validate input
    if not isinstance(files, list) or (len(files) > 1 and not isinstance(files[0], basestring)):
      raise UserError('files must be a list of string file names')

    if len(files) > 128:
      raise UserError('max batch size is 128')

    images = [Image(filename=filename) for filename in files]

    if model_id is not None:
      model = Model(model_id=model_id)
    else:
      model = self.models.get(model_name)

    res = model.predict(images)
    return res

  def wait_until_inputs_delete_finish(self):
    """ Block until a current inputs deletion operation finishes

    The criteria for unblocking is 0 inputs returned from GET /inputs

    Args:
      void

    Returns:
      None
    """

    inputs = self.inputs.get_by_page()

    while len(inputs) > 0:
      time.sleep(0.2)
      inputs = self.inputs.get_by_page()

  def wait_until_inputs_upload_finish(self, max_wait=666666):
    """ Block until the inputs upload finishes

    The criteria for unblocking is 0 "to_process" inputs
    from GET /inputs/status

    Args:
      void

    Returns:
      None
    """
    to_process = 1
    elapsed = 0
    time_start = time.time()

    while to_process != 0 and elapsed > max_wait:
      status = self.inputs.check_status()
      to_process = status.to_process
      elapsed = time.time() - time_start
      time.sleep(1)

  def wait_until_models_delete_finish(self):
    """ Block until the inputs deletion finishes

    The criteria for unblocking is 0 models returned from GET /models

    Args:
      void

    Returns:
      None
    """

    private_models = list(self.models.get_all(private_only=True))

    while len(private_models) > 0:
      time.sleep(0.2)
      private_models = list(self.models.get_all(private_only=True))


class Auth(object):

  """ Clarifai Authentication

      This class is initialized as an attribute of the clarifai application object,
      accessed with app.auth
  """

  def __init__(self, api):
    self.api = api

  def get_token(self):
    ''' get token string

    Returns:
      The token as a string
    '''

    res = self.api.get_token()
    if res.get('access_token'):
      token = res['access_token']
    else:
      token = None

    return token

class Input(object):

  """ The Clarifai Input object
  """

  def __init__(self, input_id=None, concepts=None, not_concepts=None, metadata=None, geo=None,
               regions=None, feedback_info=None):
    ''' Construct an Image/Video object. it must have one of url or file_obj set.
    Args:
      input_id: unique id to set for the image. If None then the server will create and return
      one for you.
      concepts: a list of concept names this asset is associated with
      not_concepts: a list of concept names this asset does not associate with
      metadata: metadata as a JSON object to associate arbitrary info with the input
      geo: geographical info for the input, as a Geo() object
      regions: regions of Region object
      feedback_info: FeedbackInfo object
    '''

    self.input_id = input_id

    if not isinstance(concepts, (list, tuple)) and concepts is not None:
      raise UserError('concepts should be a list or tuple')

    if not isinstance(not_concepts, (list, tuple)) and not_concepts is not None:
      raise UserError('not_concepts should be a list or tuple')

    if not isinstance(metadata, dict) and metadata is not None:
      raise UserError('metadata should be a dictionary')

    # validate geo
    if not isinstance(geo, Geo) and geo is not None:
      raise UserError('geo should be a Geo object')

    # validate more
    if not isinstance(regions, list) and regions is not None and not isinstance(regions[0], Region):
      raise UserError('regions should be a list of Region')

    if not isinstance(feedback_info, FeedbackInfo) and feedback_info is not None:
      raise UserError('feedback_info should be a FeedbackInfo object')

    self.concepts = concepts
    self.not_concepts = not_concepts
    self.metadata = metadata
    self.geo = geo
    self.feedback_info = feedback_info
    self.regions = regions
    self.score = 0
    self.status = None

  def dict(self):
    ''' Return the data of the Input as a dict ready to be input to json.dumps. '''
    data = {'data':{}}

    if self.input_id is not None:
      data['id'] = self.input_id

    # fill the tags
    if self.concepts is not None:
      pos_terms = [(term, True) for term in self.concepts]
    else:
      pos_terms = []

    if self.not_concepts is not None:
      neg_terms = [(term, False) for term in self.not_concepts]
    else:
      neg_terms = []

    terms = pos_terms + neg_terms
    if terms:
      data['data']['concepts'] = [{'id':name, 'value':value} for name, value in terms]

    if self.metadata:
      data['data']['metadata'] = self.metadata

    if self.geo:
      data['data'].update(self.geo.dict())

    if self.feedback_info:
      data.update(self.feedback_info.dict())

    if self.regions:
      data['data']['regions'] = [r.dict() for r in self.regions]

    return data


class Image(Input):

  def __init__(self, url=None, file_obj=None, base64=None, filename=None, crop=None,
               image_id=None, concepts=None, not_concepts=None,
               regions=None,
               metadata=None, geo=None, feedback_info=None, allow_dup_url=False):
    '''
      url: the url to a publically accessible image.
      file_obj: a file-like object in which read() will give you the bytes.
      crop: a list of float in the range 0-1.0 in the order [top, left, bottom, right] to crop out
            the asset before use.
    '''

    super(Image, self).__init__(image_id, concepts, not_concepts, metadata=metadata, geo=geo,
                                regions=regions,
                                feedback_info=feedback_info)

    if crop is not None and (not isinstance(crop, list) or len(crop) != 4):
      raise UserError("crop arg must be list of 4 floats or None")

    self.url = url.strip() if url else url
    self.filename = filename
    self.file_obj = file_obj
    self.base64 = base64
    self.crop = crop
    self.allow_dup_url = allow_dup_url

    # override the filename with the fileobj as fileobj
    if self.filename is not None:
      if not os.path.exists(self.filename):
        raise UserError("Invalid file path %s. Please check!")
      elif not os.path.isfile(self.filename):
        raise UserError("Not a regular file %s. Please check!")

      self.file_obj = open(self.filename, 'rb')
      self.filename = None

    if self.file_obj is not None:
      if not hasattr(self.file_obj, 'getvalue') and not hasattr(self.file_obj, 'read'):
        raise UserError("Not sure how to read your file_obj")

      if hasattr(self.file_obj, 'mode') and self.file_obj.mode != 'rb':
        raise UserError(("If you're using open(), then you need to read bytes using the 'rb' mode. "
                         "For example: open(filename, 'rb')"))

  def dict(self):

    data = super(Image, self).dict()

    image = {'image':{}}

    if self.file_obj is not None:
      # DO NOT put 'read' as first condition
      # as io.BytesIO() has both read() and getvalue() and read() gives you an empty buffer...

      # rewind the fileobj first
      self.file_obj.seek(0)

      if hasattr(self.file_obj, 'getvalue'):
        base64_imgstr = base64.b64encode(self.file_obj.getvalue()).decode('UTF-8')
      elif hasattr(self.file_obj, 'read'):
        base64_imgstr = base64.b64encode(self.file_obj.read()).decode('UTF-8')
      else:
        raise UserError("Not sure how to read your file_obj")

      image['image']['base64'] = base64_imgstr
    elif self.base64 is not None:
      image['image']['base64'] = self.base64.decode('UTF-8')
    else:
      image['image']['url'] = self.url

    if self.crop is not None:
      image['image']['crop'] = self.crop

    image['image']['allow_duplicate_url'] = self.allow_dup_url

    data['data'].update(image)
    return data


class Video(Input):

  def __init__(self, url=None, file_obj=None, base64=None, filename=None, video_id=None):
    '''
      url: the url to a publicly accessible video.
      file_obj: a file-like object in which read() will give you the bytes.
      base64: base64 encoded string for the video
      filename: a local file name
      video_id: user-defined identifier of this video
    '''

    super(Video, self).__init__(input_id=video_id)

    self.url = url.strip() if url else url
    self.filename = filename
    self.file_obj = file_obj
    self.base64 = base64

    # override the filename with the fileobj as fileobj
    if self.filename is not None:
      if not os.path.exists(self.filename):
        raise UserError("Invalid file path %s. Please check!")
      elif not os.path.isfile(self.filename):
        raise UserError("Not a regular file %s. Please check!")

      self.file_obj = open(self.filename, 'rb')
      self.filename = None

    if self.file_obj is not None:
      if not hasattr(self.file_obj, 'getvalue') and not hasattr(self.file_obj, 'read'):
        raise UserError("Not sure how to read your file_obj")

      if hasattr(self.file_obj, 'mode') and self.file_obj.mode != 'rb':
        raise UserError(("If you're using open(), then you need to read bytes using the 'rb' mode. "
                         "For example: open(filename, 'rb')"))

  def dict(self):

    data = super(Video, self).dict()

    video = {'video':{}}

    if self.file_obj is not None:
      # rewind the fileobj first
      self.file_obj.seek(0)

      # DO NOT put 'read' as first condition
      # as io.BytesIO() has both read() and getvalue() and read() gives you an empty buffer...
      if hasattr(self.file_obj, 'getvalue'):
        base64_imgstr = base64.b64encode(self.file_obj.getvalue()).decode('UTF-8')
      elif hasattr(self.file_obj, 'read'):
        base64_imgstr = base64.b64encode(self.file_obj.read()).decode('UTF-8')
      else:
        raise UserError("Not sure how to read your file_obj")

      video['video']['base64'] = base64_imgstr
    elif self.base64 is not None:
      video['video']['base64'] = self.base64.decode('UTF-8')
    else:
      video['video']['url'] = self.url

    data['data'].update(video)
    return data


class FeedbackType(Enum):
  """ Enum class for feedback type """

  accurate = 1
  misplaced = 2
  not_detected = 3
  false_positive = 4


class FeedbackInfo(object):
  """
  FeedbackInfo holds the metadata of a feedback
  """

  def __init__(self, end_user_id=None, session_id=None, event_type=None,
                output_id=None, search_id=None):

    self.end_user_id = end_user_id
    self.session_id = session_id
    self.event_type = event_type
    self.output_id = output_id
    self.search_id = search_id

  def dict(self):

    data = {
      "feedback_info":{
        "end_user_id": self.end_user_id,
        "session_id":  self.session_id,
        "event_type":  self.event_type,
      }
    }

    if self.output_id:
      data['feedback_info']['output_id'] = self.output_id

    if self.search_id:
      data['feedback_info']['search_id'] = self.search_id

    return data


class SearchTerm(object):
  """
|  Clarifai search term interface. This is the base class for InputSearchTerm and OutputSearchTerm

  It is used to build SearchQueryBuilder
  """

  def __init__(self):
    pass

  def dict(self):
    pass


class InputSearchTerm(SearchTerm):
  """
  Clarifai Input Search Term for an image search.
  For input search, you can specify search terms for url string match, input_id string match,
  concept string match, concept_id string match, and geographic information.
  Value indicates whether the concept search is a NOT search

  Examples:
    >>> # search for url, string match
    >>> InputSearchTerm(url='http://blabla')
    >>> # search for input ID, string match
    >>> InputSearchTerm(input_id='site1_bla')
    >>> # search for annotated concept
    >>> InputSearchTerm(concept='tag1')
    >>> # search for not the annotated concept
    >>> InputSearchTerm(concept='tag1', value=False)
    >>> # search for metadata
    >>> InputSearchTerm(metadata={'key':'value'})
    >>> # search for geo
    >>> InputSearchTerm(geo=Geo(geo_point=GeoPoint(-40, 30), geo_limit=GeoLimit('withinMiles', 10)))
  """

  def __init__(self, url=None, input_id=None, concept=None, concept_id=None, value=True, \
               metadata=None, geo=None):
    self.url = url
    self.input_id = input_id
    self.concept = concept
    self.concept_id = concept_id
    self.value = value
    self.metadata = metadata
    self.geo = geo

  def dict(self):
    if self.url:
      obj = { "input": {
                "data": {
                  "image": {
                    "url": self.url
                  }
                }
              }
            }
    elif self.input_id:
      obj = { "input": {
                "id": self.input_id,
                "data": {
                  "image": {}
                }
              }
            }
    elif self.concept:
      obj = { "input": {
                "data": {
                    "concepts": [ {"name":self.concept, "value":self.value} ]
                }
              }
            }
    elif self.concept_id:
      obj = { "input": {
                "data": {
                    "concepts": [ {"id":self.concept_id, "value":self.value} ]
                }
              }
            }
    elif self.metadata:
      obj = { "input": {
                "data": {
                    "metadata": self.metadata
                }
              }
            }
    elif self.geo:
      obj = { "input": {
                "data": {
                }
              }
            }
      obj['input']['data'].update(self.geo.dict())

    return obj

class OutputSearchTerm(SearchTerm):
  """
  Clarifai Output Search Term for image search.
  For output search, you can specify search term for url, base64, and input_id for
  visual search,
  or specify concept and concept_id for string match.
  Value indicates whether the concept search is a NOT search

  Examples:
    >>> # search for visual similarity from url
    >>> OutputSearchTerm(url='http://blabla')
    >>> # search for visual similarity from base64 encoded image
    >>> OutputSearchTerm(base64='sdfds')
    >>> # search for visual similarity from input id
    >>> OutputSearchTerm(input_id='site1_bla')
    >>> # search for predicted concept
    >>> OutputSearchTerm(concept='tag1')
    >>> # search for not the predicted concept
    >>> OutputSearchTerm(concept='tag1', value=False)
  """

  def __init__(self, url=None, base64=None, input_id=None, concept=None, concept_id=None, \
               value=True, crop=None):
    self.url = url
    self.base64 = base64
    self.input_id = input_id
    self.concept = concept
    self.concept_id = concept_id
    self.value = value
    self.crop = crop

  def dict(self):
    if self.url:
      obj = { "output": {
                "input": {
                  "data": {
                    "image": {
                      "url": self.url
                    }
                  }
                }
              }
            }

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    if self.base64:
      obj = { "output": {
                "input": {
                  "data": {
                    "image": {
                      "base64": self.base64
                    }
                  }
                }
              }
            }

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    elif self.input_id:
      obj = { "output": {
                "input": {
                  "id": self.input_id,
                  "data": {
                    "image": {
                    }
                  }
                }
              }
            }

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    elif self.concept:
      obj = { "output": {
                "data": {
                  "concepts": [
                    {"name": self.concept, "value":self.value}
                  ]
                }
              }
            }

    elif self.concept_id:
      obj = { "output": {
                "data": {
                  "concepts": [
                    {"id": self.concept_id, "value":self.value}
                  ]
                }
              }
            }

    return obj


class SearchQueryBuilder(object):
  """
  Clarifai Image Search Query Builder

  This builder is for advanced search use ONLY.

  If you are looking for simple concept search, or simple image similarity search,
  you should use one of the existing functions ``search_by_annotated_concepts``, ``search_by_predicted_concepts``,
  ``search_by_image`` or ``search_by_metadata``

  Currently the query builder only supports a list of query terms with AND.
  InputSearchTerm and OutputSearchTerm are the only terms supported by the query builder

  Examples:
    >>> qb = SearchQueryBuilder()
    >>> qb.add_term(term1)
    >>> qb.add_term(term2)
    >>>
    >>> app.inputs.search(qb)
    >>>
    >>> # for search over translated output concepts
    >>> qb = SearchQueryBuilder(language='zh')
    >>> qb.add_term(term1)
    >>> qb.add_term(term2)
    >>>
    >>> app.inputs.search(qb)

  """

  def __init__(self, language=None):
    self.terms = []
    self.language = language

  def add_term(self, term):
    ''' add a search term to the query.
        This can search by input or by output.
        Construct the term argument with an InputSearchTerm
        or OutputSearchTerm object.
    '''
    if not isinstance(term, InputSearchTerm) and \
       not isinstance(term, OutputSearchTerm):
      raise UserError('first level search term could be only InputSearchTerm, OutputSearchTerm')

    self.terms.append(term)

  def dict(self):
    ''' construct the raw query for the RESTful API '''

    query = { "ands":
                [term.dict() for term in self.terms]
            }

    if self.language is not None:
      query.update({'language':self.language})

    return query


class Models(object):

  def __init__(self, api):
    self.api = api

    # the cache of the model name -> model id mapping
    # to avoid an extra model query on every prediction by model name
    self.model_id_cache = self.init_model_cache()

  def init_model_cache(self):
    ''' Initialize the model cache for the public models

        This will go through all public models and cache them

        Returns:
          JSON object containing the name, type, and id of all cached models
    '''

    model_cache = {}

    # this is a generator, will NOT raise Exception
    models = self.get_all(public_only=True)

    try:
      for m in models:
        model_name = m.model_name
        model_type = m.output_info['type']
        model_id = m.model_id
        model_cache.update({(model_name, model_type):model_id})

        # for general-v1.3 concept model, make an extra cache entry
        if model_name == 'general-v1.3' and model_type == 'concept':
          model_cache.update({(model_name, None):model_id})
    except ApiError as e:
      if e.error_code == 11007:
        logger.debug("not authorized to call GET /models. Unable to cache models")
        models = []
        pass
      else:
        raise e

    return model_cache

  def clear_model_cache(self):
    ''' clear model_name -> model_id cache

        WARNING: This is an internal function, user should not call this

        We cache model_name to model_id mapping for API efficiency.
        The first time you call a models.get() by name, the name to ID
        mapping is saved so next time there is no query. Then user does not
        have to query the model ID every time when they want to work on it.

        Returns:
          None
    '''

    self.model_id_cache = {}

  def create(self, model_id, model_name=None, concepts=None, \
             concepts_mutually_exclusive=False, \
             closed_environment=False, \
             hyper_parameters=None):

    ''' Create a new model

    Args:
      model_id: ID of the model
      model_name: optional name of the model
      concepts: optional concepts to be associated with this model
      concepts_mutually_exclusive: True or False, whether concepts are mutually exclusive
      closed_environment: True or False, whether to use negatives for prediction
      hyper_parameters: hyper parameters for the model, with a json object

    Returns:
      Model object

    Examples:
      >>> # create a model with no concepts
      >>> app.models.create('my_model1')
      >>> # create a model with a few concepts
      >>> app.models.create('my_model2', concepts=['bird', 'fish'])
      >>> # create a model with closed environment
      >>> app.models.create('my_model3', closed_environment=True)
    '''
    if not model_name:
      model_name = model_id

    res = self.api.create_model(model_id, model_name, concepts, \
                                concepts_mutually_exclusive, closed_environment, hyper_parameters)

    if res.get('model'):
      model = self._to_obj(res['model'])
    elif res.get('status'):
      status = res['status']
      raise UserError('code: %d, desc: %s, details: %s' % \
                      (status['code'], status['description'], status['details']))

    return model

  def _is_public(self, model):
    ''' use app_id to determine whether it is a public model

        For public model, the app_id is either '' or 'main'
        For private model, the app_id is not empty but not 'main'
    '''
    app_id = model.app_id

    if app_id == '' or app_id == 'main':
      return True
    else:
      return False

  def get_all(self, public_only=False, private_only=False):
    ''' Get all models in the application

    Args:
      public_only: only yield public models
      private_only: only yield private models that tie to your own account

    Returns:
      a generator function that yields Model objects

    Examples:
      >>> for model in app.models.get_all():
      >>>   print model.model_name
    '''

    page = 1
    per_page = 20

    while True:
      res = self.api.get_models(page, per_page)

      if not res['models']:
        break

      for one in res['models']:
        model = self._to_obj(one)

        if public_only is True and not self._is_public(model):
          continue

        if private_only is True and self._is_public(model):
          continue

        yield model

      page += 1

  def get_by_page(self, public_only=False, private_only=False, page=1, per_page=20):
    ''' get paginated models from the application

    When the number of models gets high, you may want to get
    the paginated results from all the models

    Args:
      public_only: only yield public models
      private_only: only yield private models that tie to your own account
      page: page number
      per_page: number of models returned in one page

    Returns:
      a list of Model objects

    Examples:
      >>> models = app.models.get_by_page(2, 20)
    '''

    res = self.api.get_models(page, per_page)
    results = [self._to_obj(one) for one in res['models']]

    if public_only is True:
      results = filter(lambda m: self._is_public(m), results)
    elif private_only is True:
      results = filter(lambda m: not self._is_public(m), results)

    return results

  def delete(self, model_id, version_id=None):
    ''' delete the model, or a specific version of the model

        Without model version id specified, all the versions associated with this model
        will be deleted as well.

        With model version id specified, it will delete a
        particular model version from the model

        Args:
          model_id: the unique ID of the model
          version_id: the unique ID of the model version

        Returns:
          the raw JSON response from the server

        Examples:
          >>> # delete a model
          >>> app.models.delete('model_id1')
          >>> # delete a model version
          >>> app.models.delete('model_id1', version_id='version1')
    '''

    if version_id is None:
      res = self.api.delete_model(model_id)
    else:
      res = self.api.delete_model_version(model_id, version_id)

    return res

  def delete_all(self):
    ''' Delete all models and the versions associated with each one

        After this operation, you will have no models in the
        application

        Returns:
          the raw JSON response from the server

        Examples:
          >>> app.models.delete_all()
    '''

    res = self.api.delete_all_models()
    return res

  def get(self, model_name=None, model_id=None, model_type=None):
    ''' Get a model, by ID or name

    Args:
      model_name: name of the model
      model_id: unique identifier of the model
      model_type: type of the model

    Returns:
      the Model object

    Examples:
      >>> # get general-v1.3 model
      >>> app.models.get('general-v1.3')
    '''

    # if the model ID is specified, just make the Model
    if model_id:
      model = Model(self.api, model_id=model_id)
      return model

    # search for the model_name together with the model_type
    if self.model_id_cache.get((model_name, model_type)):
      model_id = self.model_id_cache[(model_name, model_type)]
      model = Model(self.api, model_id=model_id)
      return model

    try:
      res = self.api.get_model(model_name)
      model = self._to_obj(res['model'])
    except ApiError as e:

      if e.response.status_code == 401:
        raise e

      if e.response.status_code == 404:
        res = self.search(model_name, model_type)

        if res is None:
          raise e

        if len(res) > 0:
          # exclude embed and cluster model when it's not explicitly searched for
          if model_type is None:
            res = list(filter(lambda one: (one.output_info['type'] != u'embed') & (one.output_info['type'] != u'cluster'), res))

        if len(res) > 1:
          logging.error('Model search results with multiple models. Please refine your search')
          return None

        model = res[0]
        self.model_id_cache.update({(model_name, model_type):model.model_id})

    return model

  def search(self, model_name, model_type=None):
    '''
        Search the model by name and optionally type. Default is to search concept models
        only. All the custom model trained are concept models.

        Args:
          model_name: name of the model. name is not unique.
          model_type: default to None, equivalent to wildcards search

        Returns:
          a list of Model objects or None

        Examples:
          >>> # search for general-v1.3 models
          >>> app.models.search('general-v1.3')
          >>>
          >>> # search for color model
          >>> app.models.search('color', model_type='color')
          >>>
          >>> # search for face model
          >>> app.models.search('face-v1.3', model_type='facedetect')
    '''

    res = self.api.search_models(model_name, model_type)
    if res.get('models'):
      results = [self._to_obj(one) for one in res['models']]
    else:
      results = None

    return results

  def _to_obj(self, item):
    ''' convert a model json object to Model object '''
    return Model(self.api, item)

def _escape(param):
  return param.replace('/', '%2F')


class Inputs(object):

  def __init__(self, api):
    self.api = api

  def create_image(self, image):
    ''' create an image from Image object

    Args:
      image: a Clarifai Image object

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image(Image(url='https://samples.clarifai.com/metro-north.jpg'))
    '''

    ret = self.api.add_inputs([image])

    img = self._to_obj(ret['inputs'][0])
    return img

  def create_image_from_url(self, url, image_id=None, concepts=None, not_concepts=None, crop=None, \
                            metadata=None, geo=None, allow_duplicate_url=False):
    ''' create an image from Image url

    Args:
      url: image url
      image_id: ID of the image
      concepts: a list of concept names this image is associated with
      not_concepts: a list of concept names this image is not associated with
      crop: crop information, with four corner coordinates
      metadata: meta data with a dictionary
      geo: geo info with a dictionary
      allow_duplicate_url: True or False, the flag to allow a duplicate url to be imported

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image_from_url(url='https://samples.clarifai.com/metro-north.jpg')
      >>>
      >>> # create image with geo point
      >>> app.inputs.create_image_from_url(url='https://samples.clarifai.com/metro-north.jpg',
      >>>   geo=Geo(geo_point=GeoPoint(22.22, 44.44))
    '''

    url = url.strip() if url else url

    image = Image(url=url, image_id=image_id, concepts=concepts, not_concepts=not_concepts, \
                  crop=crop, metadata=metadata, geo=geo, allow_dup_url=allow_duplicate_url)

    return self.create_image(image)

  def create_image_from_filename(self, filename, image_id=None, concepts=None, not_concepts=None, \
                                 crop=None, metadata=None, geo=None, allow_duplicate_url=False):
    ''' create an image by local filename

    Args:
      filename: local filename
      image_id: ID of the image
      concepts: a list of concept names this image is associated with
      not_concepts: a list of concept names this image is not associated with
      crop: crop information, with four corner coordinates
      metadata: meta data with a dictionary
      geo: geo info with a dictionary
      allow_duplicate_url: True or False, the flag to allow a duplicate url to be imported

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image_filename(filename="a.jpeg")
    '''

    fileio = open(filename, 'rb')
    image = Image(file_obj=fileio, image_id=image_id, concepts=concepts, \
                  not_concepts=not_concepts, crop=crop, metadata=metadata, geo=geo, \
                  allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_bytes(self, img_bytes, image_id=None, concepts=None, not_concepts=None, \
                              crop=None, metadata=None, geo=None, allow_duplicate_url=False):
    ''' create an image by image bytes

    Args:
      img_bytes: raw bytes of an image
      image_id: ID of the image
      concepts: a list of concept names this image is associated with
      not_concepts: a list of concept names this image is not associated with
      crop: crop information, with four corner coordinates
      metadata: meta data with a dictionary
      geo: geo info with a dictionary
      allow_duplicate_url: True or False, the flag to allow a duplicate url to be imported

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image_bytes(img_bytes="raw image bytes...")
    '''

    fileio = BytesIO(img_bytes)
    image = Image(file_obj=fileio, image_id=image_id, concepts=concepts, \
                  not_concepts=not_concepts, crop=crop, metadata=metadata, geo=geo, \
                  allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_base64(self, base64_bytes, image_id=None, concepts=None, \
                               not_concepts=None, crop=None, metadata=None, geo=None, \
                               allow_duplicate_url=False):
    ''' create an image by base64 bytes

    Args:
      base64_bytes: base64 encoded image bytes
      image_id: ID of the image
      concepts: a list of concept names this image is associated with
      not_concepts: a list of concept names this image is not associated with
      crop: crop information, with four corner coordinates
      metadata: meta data with a dictionary
      geo: geo info with a dictionary
      allow_duplicate_url: True or False, the flag to allow a duplicate url to be imported

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image_bytes(base64_bytes="base64 encoded image bytes...")
    '''

    image = Image(base64=base64_bytes, image_id=image_id, concepts=concepts, \
                  not_concepts=not_concepts, crop=crop, metadata=metadata, geo=geo, \
                  allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def bulk_create_images(self, images):
    ''' Create images in bulk

    Args:
      images: a list of Image objects

    Returns:
      a list of the Image objects that were just created

    Examples:
      >>> img1 = Image(url="", concepts=['cat', 'kitty'])
      >>> img2 = Image(url="", concepts=['dog'], not_concepts=['cat'])
      >>> app.inputs.bulk_create_images([img1, img2])
    '''

    lens = len(images)
    if lens > 128:
      raise UserError('the maximum number of inputs in a batch is 128')

    res = self.api.add_inputs(images)
    images = [self._to_obj(one) for one in res['inputs']]
    return images

  def check_status(self):
    ''' check the input upload status

    Args:
      Void

    Returns:
      InputCounts object

    Examples:
      >>> status = app.inputs.check_status()
      >>> print status.code
      >>> print status.description
    '''

    ret = self.api.get_inputs_status()
    counts = InputCounts(ret)
    return counts


  def get_all(self, ignore_error=False):
    ''' Get all inputs


    Args:
      ignore_error: ignore errored inputs. For example some images may fail to be imported due to bad url

    Returns:
      a generator function that yields Input objects

    Examples:
      >>> for image in app.inputs.get_all():
      >>>   print image.input_id
    '''

    page = 1
    per_page = 20

    while True:
      try:
        res = self.api.get_inputs(page, per_page)
      except ApiError as e:
        if e.response.status_code == 207 and e.error_code == 10010:
          res = e.response.json()
        else:
          raise e

      if not res['inputs']:
        break

      for one in res['inputs']:
        input = self._to_obj(one)

        if ignore_error is True and input.status.code != 30000:
          continue

        yield input

      page += 1


  def get_by_page(self, page=1, per_page=20, ignore_error=False):
    ''' Get inputs with pagination

    Args:
      page: page number
      per_page: number of inputs to retrieve per page
      ignore_error: ignore errored inputs. For example some images may fail to be imported due to bad url

    Returns:
      a list of Input objects

    Examples:
      >>> for image in app.inputs.get_by_page(2, 10):
      >>>   print image.input_id
    '''

    try:
      res = self.api.get_inputs(page, per_page)
    except ApiError as e:
      if e.response.status_code == 207 and e.error_code == 10010:
        res = e.response.json()
      else:
        raise e

    results = []
    for one in res['inputs']:
      input = self._to_obj(one)

      if ignore_error is True and input.status.code != 30000:
        continue

      results.append(input)

    return results

  def delete(self, input_id):
    ''' delete an input with input ID

    Args:
      input_id: the unique input ID

    Returns:
      ApiStatus object

    Examples:
      >>> ret = app.inputs.delete('id1')
      >>> print ret.code
    '''

    if isinstance(input_id, list):
      res = self.api.delete_inputs(input_id)
    else:
      res = self.api.delete_input(input_id)

    return ApiStatus(res['status'])

  def delete_all(self):
    ''' delete all inputs from the application
    '''
    res = self.api.delete_all_inputs()
    return ApiStatus(res['status'])

  def get(self, input_id):
    ''' get an Input object by input ID

    Args:
      input_id: the unique identifier of the input

    Returns:
      an Image/Input object

    Examples:
      >>> image = app.inputs.get('id1')
      >>> print image.input_id

    '''

    res = self.api.get_input(input_id)
    one = res['input']
    return self._to_obj(one)

  def search(self, qb, page=1, per_page=20, raw=False):
    ''' search with a clarifai image query builder

        WARNING: this is the advanced search function. You will need to build a query builder
        in order to use this.

      There are a few simple search functions:
          search_by_annotated_concepts()
          search_by_predicted_concepts()
          search_by_image()
          search_by_metadata()

    Args:
      qb: clarifai query builder
      raw: raw result indicator

    Returns:
      a list of Input/Image object
    '''

    res = self.api.search_inputs(qb.dict(), page, per_page)

    # output raw result when the flag is set
    if raw is True:
      return res

    hits = [self._to_search_obj(one) for one in res['hits']]
    return hits

  def search_by_image(self, image_id=None, \
                            image=None, url=None, \
                            imgbytes=None, base64bytes=None, \
                            fileobj=None, filename=None, \
                            crop=None, page=1, per_page=20, raw=False):
    ''' Search for visually similar images

    By passing image_id, raw image bytes, base64 encoded bytes, image file io stream,
    image filename, or Clarifai Image object, you can use the visual search power of
    the Clarifai API.

    You can specify a crop of the image to search over

    Args:
      image_id: unique ID of the image for search
      image: Image object for search
      imgbytes: raw image bytes for search
      base64bytes: base63 encoded image bytes
      fileobj: file io stream, like open(file)
      filename: filename on local filesystem
      crop: crop of the image as a list of four floats representing the corner coordinates
      page: page number
      per_page: number of images returned per page
      raw: raw result indicator

    Returns:
      a list of Image object

    Examples:
      >>> # search by image url
      >>> app.inputs.search_by_image(url='http://blabla')
      >>> # search by local filename
      >>> app.inputs.search_by_image(filename='bla')
      >>> # search by raw image bytes
      >>> app.inputs.search_by_image(imgbytes='data')
      >>> # search by base64 encoded image bytes
      >>> app.inputs.search_by_image(base64bytes='data')
      >>> # search by file stream io
      >>> app.inputs.search_by_image(fileobj=open('file'))
    '''

    not_nones = [x for x in [image_id, image, url, imgbytes, base64bytes, fileobj, filename] if x is not None]
    if len(not_nones) != 1:
      raise UserError('Unable to construct an image')

    if image_id:
      qb = SearchQueryBuilder()
      term = OutputSearchTerm(input_id=image_id)
      qb.add_term(term)

      res = self.search(qb, page, per_page)
    elif image:
      qb = SearchQueryBuilder()

      if image.url:
        term = OutputSearchTerm(url=image.url)
      elif image.base64:
        term = OutputSearchTerm(base64=image.base64.decode('UTF-8'))
      elif image.file_obj:
        base64_bytes = ''

        if hasattr(image.file_obj, 'getvalue'):
          base64_bytes = base64.b64encode(image.file_obj.getvalue()).decode('UTF-8')
        elif hasattr(image.file_obj, 'read'):
          base64_bytes = base64.b64encode(image.file_obj.read()).decode('UTF-8')
        else:
          raise UserError("Not sure how to read your file_obj")

        term = OutputSearchTerm(base64=base64_bytes)

      qb.add_term(term)

      res = self.search(qb, page, per_page, raw)
    elif url:
      img = Image(url=url, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page, raw=raw)
    elif fileobj:
      img = Image(file_obj=fileobj, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page, raw=raw)
    elif imgbytes:
      fileio = BytesIO(imgbytes)
      img = Image(file_obj=fileio, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page, raw=raw)
    elif filename:
      fileio = open(filename, 'rb')
      img = Image(file_obj=fileio, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page, raw=raw)
    elif base64:
      img = Image(base64=base64bytes, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page, raw=raw)

    return res

  def search_by_original_url(self, url, page=1, per_page=20, raw=False):
    ''' search by the original url of the uploaded images

    Args:
      url: url of the image
      page: page number
      per_page: the number of images to return per page
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_original_url(url='http://bla')
    '''

    qb = SearchQueryBuilder()

    term = InputSearchTerm(url=url)
    qb.add_term(term)
    res = self.search(qb, page, per_page, raw)

    return res

  def search_by_metadata(self, metadata, page=1, per_page=20, raw=False):
    ''' search by meta data of the image rather than concept

    Args:
      metadata: a dictionary for meta data search.
            The dictionary could be a simple one with only one key and value,
            Or a nested dictionary with multi levels.
      page: page number
      per_page: the number of images to return per page
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_metadata(metadata={'name':'bla'})
      >>> app.inputs.search_by_metadata(metadata={'my_class1': { 'name' : 'bla' }})
    '''

    if isinstance(metadata, dict):
      qb = SearchQueryBuilder()

      term = InputSearchTerm(metadata=metadata)
      qb.add_term(term)
      res = self.search(qb, page, per_page, raw)
    else:
      raise UserError('Metadata must be a valid dictionary. Please double check.')

    return res

  def search_by_annotated_concepts(self, concept=None, concepts=None, \
                                   value=True, values=None, \
                                   concept_id=None, concept_ids=None, \
                                   page=1, per_page=20, raw=False):
    ''' search using the concepts the user has manually specified

    Args:
      concept: concept name to search
      concepts: a list of concept name to search
      concept_id: concept IDs to search
      concept_ids: a list of concept IDs to search
      value: whether the concept should be a positive tag or negative
      values: the list of values corresponding to the concepts
      page: page number
      per_page: number of images to return per page
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_annotated_concepts(concept='cat')
    '''

    if not concept and not concepts and concept_id and concept_ids:
      raise UserError('concept could not be null.')

    if concept or concepts:

      if concept and not isinstance(concept, basestring):
        raise UserError('concept should be a string')
      elif concepts and not isinstance(concepts, list):
        raise UserError('concepts must be a list')
      elif concepts and not all([isinstance(one, basestring) for one in concepts]):
        raise UserError('concepts must be a list of all string')

      if concept and concepts:
        raise UserError('you can either search by concept or concepts but not both')

      if concept:
        concepts = [concept]

      if not values:
        values = [value]

      qb = SearchQueryBuilder()

      for concept, value in zip(concepts, values):
        term = InputSearchTerm(concept=concept, value=value)
        qb.add_term(term)

    else:

      if concept_id and not isinstance(concept_id, basestring):
        raise UserError('concept should be a string')
      elif concept_ids and not isinstance(concept_ids, list):
        raise UserError('concepts must be a list')
      elif concept_ids and not all([isinstance(one, basestring) for one in concept_ids]):
        raise UserError('concepts must be a list of all string')

      if concept_id and concept_ids:
        raise UserError('you can either search by concept_id or concept_ids but not both')

      if concept_id:
        concept_ids = [concept_id]

      if not values:
        values = [value]

      qb = SearchQueryBuilder()

      for concept_id, value in zip(concept_ids, values):
        term = InputSearchTerm(concept_id=concept_id, value=value)
        qb.add_term(term)

    return self.search(qb, page, per_page, raw)

  def search_by_geo(self, geo_point=None, geo_limit=None, geo_box=None, page=1, per_page=20,
                     raw=False):
    ''' search by geo point and geo limit

    Args:
      geo_point: A GeoPoint object, which represents the (longitude, latitude) of a location
      geo_limit: A GeoLimit object, which represents a range to a GeoPoint
      geo_box: A GeoBox object, which represents a box area
      page: page number
      per_page: number of images to return per page
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_geo(GeoPoint(30, 40), GeoLimit("mile", 10))
    '''
    if geo_limit is None:
      geo_limit = GeoLimit("mile", 10)

    if geo_point and not isinstance(geo_point, GeoPoint):
      raise UserError('geo_point type not match GeoPoint. Please check data type.')

    if not isinstance(geo_limit, GeoLimit):
      raise UserError('geo_limit type not match GeoLimit. Please check data type.')

    if geo_box and not isinstance(geo_box, GeoBox):
      raise UserError('geo_box type not match GeoBox. Please check data type.')

    if geo_point is None and geo_box is None:
      raise UserError('at least geo_point or geo_box needs to be specified for the geo search.')

    if geo_point and geo_box:
      raise UserError('confusing. you cannot search by geo_point and geo_box together.')

    qb = SearchQueryBuilder()

    if geo_point is not None:
      term = InputSearchTerm(geo=Geo(geo_point=geo_point, geo_limit=geo_limit))
    elif geo_box is not None:
      term = InputSearchTerm(geo=Geo(geo_box=geo_box))

    qb.add_term(term)

    return self.search(qb, page, per_page, raw)

  def search_by_predicted_concepts(self, concept=None, concepts=None, \
                                         value=True, values=None,\
                                         concept_id=None, concept_ids=None, \
                                         page=1, per_page=20, lang=None, raw=False):
    ''' search over the predicted concepts

    Args:
      concept: concept name to search
      concepts: a list of concept names to search
      concept_id: concept id to search
      concept_ids: a list of concept ids to search
      value: whether the concept should be a positive tag or negative
      values: the list of values corresponding to the concepts
      page: page number
      per_page: number of images to return per page
      lang: language to search over for translated concepts
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_predicted_concepts(concept='cat')
      >>> # search over simplified Chinese label
      >>> app.inputs.search_by_predicted_concepts(concept=u'狗', lang='zh')
    '''
    if not concept and not concepts and concept_id and concept_ids:
      raise UserError('concept could not be null.')

    if concept and not isinstance(concept, basestring):
      raise UserError('concept should be a string')
    elif concepts and not isinstance(concepts, list):
      raise UserError('concepts must be a list')
    elif concepts and not all([isinstance(one, basestring) for one in concepts]):
      raise UserError('concepts must be a list of all string')

    if concept or concepts:
      if concept and concepts:
        raise UserError('you can either search by concept or concepts but not both')

      if concept:
        concepts = [concept]

      if not values:
        values = [value]

      qb = SearchQueryBuilder(language=lang)

      for concept, value in zip(concepts, values):
        term = OutputSearchTerm(concept=concept, value=value)
        qb.add_term(term)

    else:

      if concept_id and concept_ids:
        raise UserError('you can either search by concept_id or concept_ids but not both')

      if concept_id:
        concept_ids = [concept_id]

      if not values:
        values = [value]

      qb = SearchQueryBuilder()

      for concept_id, value in zip(concept_ids, values):
        term = OutputSearchTerm(concept_id=concept_id, value=value)
        qb.add_term(term)

    return self.search(qb, page, per_page, raw)

  def send_search_feedback(self, input_id, feedback_info=None):
    '''
    Send feedback for search

    Args:
      input_id: unique identifier for the input

    Returns:
      None
    '''

    feedback_input = Image(image_id=input_id, feedback_info=feedback_info)
    res = self.api.send_search_feedback(feedback_input)

    return res

  def update(self, image, action='merge'):
    '''
    Update the information of an input/image

    Args:
      image: an Image object that has concepts, metadata, etc.
      method: one of ['merge', 'overwrite']

              'merge' is to append the info onto the existing info, for either concept or metadata

              'overwrite' is to overwrite the existing metadata and concepts with the existing ones

    Returns:
      an Image object

    Examples:
      >>> new_img = Image(image_id="abc", concepts=['c1', 'c2'], not_concepts=['c3'], metadata={'key':'val'})
      >>> app.inputs.update(new_img, action='overwrite')
    '''
    res = self.api.patch_inputs(action=action, inputs=[image])

    one = res['inputs'][0]
    return self._to_obj(one)

  def bulk_update(self, images, action='merge'):
    ''' Update the input
    update the information of an input/image

    Args:
      images: a list of Image objects that have concepts, metadata, etc.
      action: one of ['merge', 'overwrite']

              'merge' is to append the info onto the exising info, for either concept or metadata

              'overwrite' is to overwrite the existing metadata and concepts with the existing ones

    Returns:
      an Image object

    Examples:
      >>> new_img1 = Image(image_id="abc1", concepts=['c1', 'c2'], not_concepts=['c3'], metadata={'key':'val'})
      >>> new_img2 = Image(image_id="abc2", concepts=['c1', 'c2'], not_concepts=['c3'], metadata={'key':'val'})
      >>> app.inputs.update([new_img1, new_img2], action='overwrite')
    '''
    ret = self.api.patch_inputs(action=action, inputs=images)
    objs = [self._to_obj(item) for item in ret['inputs']]
    return objs

  def delete_concepts(self, input_id, concepts):
    ''' delete concepts from an input/image

    Args:
      input_id: unique ID of the input
      concepts: a list of concept names

    Returns:
      an Image object
    '''

    res = self.update(Image(image_id=input_id, concepts=concepts), action='remove')
    return res

  def bulk_merge_concepts(self, input_ids, concept_lists):
    ''' bulk merge concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      concept_lists: a list of concept lists, each one corresponding to a listed input ID and filled with concepts to be added to that input

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_merge_concepts('id', [[('cat',True), ('dog',False)]])
    '''

    if len(input_ids) != len(concept_lists):
      raise UserError('Argument error. please check')

    inputs = []
    for input_id, concept_list in zip(input_ids, concept_lists):
      concepts = []
      not_concepts = []
      for concept_id, value in concept_list:
        if value is True:
          concepts.append(concept_id)
        else:
          not_concepts.append(concept_id)

      image = Image(image_id=input_id, concepts=concepts, not_concepts=not_concepts)
      inputs.append(image)

    res = self.bulk_update(inputs, action='merge')
    return res

  def bulk_delete_concepts(self, input_ids, concept_lists):
    ''' bulk delete concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      concept_lists: a list of concept lists, each one corresponding to a listed input ID and filled with concepts to be deleted from that input

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_delete_concepts(['id'], [['cat', 'dog']])
    '''

    # the reason list comprehension is not used is it breaks the 100 chars width
    inputs = []
    for input_id, concepts in zip(input_ids, concept_lists):
      one_input = Image(image_id=input_id, concepts=concepts)
      inputs.append(one_input)

    res = self.bulk_update(inputs, action='remove')
    return res

  def merge_concepts(self, input_id, concepts, not_concepts, overwrite=False):
    ''' Merge concepts for one input

    Args:
      input_id: the unique ID of the input
      concepts: the list of concepts
      not_concepts: the list of negative concepts
      overwrite: if True, this operation will replace the previous concepts. If False, it will append them.


    Returns:
      an Input object

    Examples:
      >>> app.inputs.merge_concepts('id', ['cat', 'kitty'], ['dog'])
    '''

    image = Image(image_id=input_id, concepts=concepts, not_concepts=not_concepts)

    if overwrite is True:
      action='overwrite'
    else:
      action='merge'

    res = self.update(image, action=action)
    return res

  def add_concepts(self, input_id, concepts, not_concepts):
    ''' Add concepts for one input

    This is just an alias of `merge_concepts` for easier understanding
    when you try to add some new concepts to an image

    Args:
      input_id: the unique ID of the input
      concepts: the list of concepts
      not_concepts: the list of negative concepts

    Returns:
      an Input object

    Examples:
      >>> app.inputs.add_concepts('id', ['cat', 'kitty'], ['dog'])
    '''
    return self.merge_concepts(input_id, concepts, not_concepts)

  def merge_metadata(self, input_id, metadata):
    ''' merge metadata for the image

    This is to merge/update the metadata of the given image

    Args:
      input_id: the unique ID of the input
      metadata: the metadata dictionary

    Examples:
      >>> # merge the metadata
      >>> # metadata will be appended to the existing key/value pairs
      >>> app.inputs.merge_metadata('id', {'key1':'value1', 'key2':'value2'})
    '''
    image = Image(image_id=input_id, metadata=metadata)

    action = 'merge'
    res = self.update(image, action=action)
    return res

  def _to_search_obj(self, one):
    ''' convert the search candidate to input object '''
    score = one['score']
    one_input = self._to_obj(one['input'])
    one_input.score = score
    return one_input

  def _to_obj(self, one):

    # get concepts
    concepts = []
    not_concepts = []
    if one['data'].get('concepts'):
      for concept in one['data']['concepts']:
        if concept['value'] == 1:
          concepts.append(concept['name'])
        else:
          not_concepts.append(concept['name'])

    if not concepts:
      concepts = None

    if not not_concepts:
      not_concepts = None

    # get metadata
    metadata = one['data'].get('metadata', None)

    # get geo
    geo = geo_json = one['data'].get('geo', None)

    if geo_json is not None:
      geo_schema = {
                    'additionalProperties': False,
                    'type': 'object',
                    'properties': {
                        'geo_point': {
                            'type': 'object',
                            'properties': {
                                'longitude': { 'type': 'number' },
                                'latitude': {'type': 'number'}
                                }
                            }
                        }
                   }

      validate(geo_json, geo_schema)
      geo = Geo(GeoPoint(geo_json['geo_point']['longitude'], geo_json['geo_point']['latitude']))

    input_id = one['id']
    if one['data'].get('image'):

      # get allow_dup_url
      allow_dup_url = one['data']['image'].get('allow_duplicate_url', False)

      if one['data']['image'].get('url'):
        if one['data']['image'].get('crop'):
          crop = one['data']['image']['crop']
          one_input = Image(image_id=input_id, url=one['data']['image']['url'], \
                            concepts=concepts, not_concepts=not_concepts, crop=crop, \
                            metadata=metadata, geo=geo, \
                            allow_dup_url=allow_dup_url)
        else:
          one_input = Image(image_id=input_id, url=one['data']['image']['url'], \
                            concepts=concepts, not_concepts=not_concepts, \
                            metadata=metadata, geo=geo, \
                            allow_dup_url=allow_dup_url)
      elif one['data']['image'].get('base64'):
        if one['data']['image'].get('crop'):
          crop = one['data']['image']['crop']
          one_input = Image(image_id=input_id, base64=one['data']['image']['base64'], \
                            concepts=concepts, not_concepts=not_concepts, crop=crop, \
                            metadata=metadata, geo=geo, \
                            allow_dup_url=allow_dup_url)
        else:
          one_input = Image(image_id=input_id, base64=one['data']['image']['base64'], \
                            concepts=concepts, not_concepts=not_concepts, \
                            metadata=metadata, geo=geo, \
                            allow_dup_url=allow_dup_url)
    elif one['data'].get('video'):
      raise UserError('Not supported yet')
    else:
      raise UserError('Unknown input type')

    if one.get('status'):
      one_input.status = ApiStatus(one['status'])

    return one_input

  def get_outputs(self, input_id):
    ''' get the output predictions for a particular input

    Args:
      input_id: the unique identifier of the input

    Returns:
      the input with the output predictions
    '''
    return self.api.get_outputs(input_id)

  def remove_outputs_concepts(self, input_id, concept_ids):
    '''
    Remove concepts from the outputs predictions.
    The concept ids must be present in your app

    Args:
      input_id: the unique identifier of the input
      concept_ids: the list of concept ids to be removed

    Returns:
      the patched input in JSON object
    '''
    return self.api.patch_outputs(input_id, action='remove', concept_ids=concept_ids)

  def merge_outputs_concepts(self, input_id, concept_ids):
    '''
    Merge new concepts into the outputs predictions.
    The concept ids must be present in your app

    Args:
      input_id: the unique identifier of the input
      concept_ids: the list of concept ids to be merged

    Returns:
      the patched input in JSON object
    '''
    return self.api.patch_outputs(input_id, action='merge', concept_ids=concept_ids)

class Concepts(object):

  def __init__(self, api):
    self.api = api

  def get_all(self):
    ''' Get all concepts associated with the application

    Args:
      void

    Returns:
      all concepts in a generator function
    '''

    page = 1
    per_page = 20

    while True:
      res = self.api.get_concepts(page, per_page)

      if not res['concepts']:
        break

      for one in res['concepts']:
        yield self._to_obj(one)

      page += 1

  def get_by_page(self, page=1, per_page=20):
    ''' get concept with pagination

    Args:
      page: page number
      per_page: number of concepts to retrieve per page

    Returns:
      a list of Concept objects

    Examples:
      >>> for concept in app.concepts.get_by_page(2, 10):
      >>>   print concept.concept_id
    '''

    res = self.api.get_concepts(page, per_page)
    results = [self._to_obj(one) for one in res['concepts']]

    return results

  def get(self, concept_id):
    ''' Get a concept by id

    Args:
      concept_id: concept ID, the unique identifier of the concept

    Returns:
      If found, return the Concept object.
      Otherwise, return None

    Examples:
      >>> app.concepts.get('id')
    '''

    res = self.api.get_concept(concept_id)
    if res.get('concept'):
      concept = self._to_obj(res['concept'])
    else:
      concept = None

    return concept

  def search(self, term, lang=None):
    ''' search concepts by concept name with wildcards

    Args:
      term: search term with wildcards allowed
      lang: language to search, if none is specified the default for the application will be used

    Returns:
      a generator function with all concepts pertaining to the search term

    Examples:
      >>> app.concepts.search('cat')
      >>> # search for Chinese label name
      >>> app.concepts.search(u'狗*', lang='zh')
    '''

    page = 1
    per_page = 20

    while True:
      res = self.api.search_concepts(term, page, per_page, lang)

      if not res['concepts']:
        break

      for one in res['concepts']:
        yield self._to_obj(one)

      page += 1

  def update(self, concept_id, concept_name, action='overwrite'):
    ''' Patch concept

    Args:
      concept_id: id of the concept
      concept_name: the new name for the concept

    Returns:
      the new Concept object

    Examples:
      >>> app.concepts.update(concept_id='myid1', concept_name='new_concept_name2')
    '''

    c = Concept(concept_name=concept_name, concept_id=concept_id)
    res = self.api.patch_concepts(action=action, concepts=[c])

    return self._to_obj(res['concepts'][0])

  def bulk_update(self, concept_ids, concept_names, action='overwrite'):
    ''' Patch multiple concepts

    Args:
      concept_ids: a list of concept IDs, in sequence
      concept_names: a list of corresponding concept names, in the same sequence

    Returns:
      the new Concept object

    Examples:
      >>> app.concepts.bulk_update(concept_ids=['myid1', 'myid2'], concept_names=['name2', 'name3'])
    '''

    concepts = [Concept(concept_name=concept_name, concept_id=concept_id) for concept_name, concept_id in zip(concept_names, concept_ids)]
    res = self.api.patch_concepts(action=action, concepts=concepts)

    return [self._to_obj(c) for c in res['concepts']]

  def create(self, concept_id, concept_name=None):
    ''' Create a new concept

    Args:
      concept_id: concept ID, the unique identifier of the concept
      concept_name: name of the concept.
                    If name is not specified, it will be set to the same as the concept ID

    Returns:
      the new Concept object
    '''

    res = self.api.add_concepts([concept_id], [concept_name])
    concept = self._to_obj(res['concepts'][0])
    return concept

  def bulk_create(self, concept_ids, concept_names=None):
    ''' Bulk create concepts

    When the concept name is not set, it will be set as the same as concept ID.

    Args:
      concept_ids: a list of concept IDs, in sequence
      concept_names: a list of corresponding concept names, in the same sequence

    Returns:
      A list of Concept objects

    Examples:
      >>> app.concepts.bulk_create(['id1', 'id2'], ['cute cat', 'cute dog'])
    '''

    res = self.api.add_concepts(concept_ids, concept_names)
    concepts = [self._to_obj(one) for one in res['concepts']]
    return concepts

  def _to_obj(self, item):

    concept_id = item['id']
    concept_name = item['name']
    app_id = item['app_id']
    created_at = item['created_at']

    return Concept(concept_name=concept_name, concept_id=concept_id, app_id=app_id, created_at=created_at)


class Model(object):

  def __init__(self, api, item=None, model_id=None):
    self.api = api

    if model_id is not None:
      self.model_id = model_id
      self.model_version = None
    elif item:
      self.model_id = item['id']
      self.model_name = item['name']
      self.created_at = item['created_at']
      self.app_id = item['app_id']
      self.model_version = item['model_version']['id']
      self.model_status_code = item['model_version']['status']['code']

      self.output_info = item.get('output_info', {})
      self.concepts = []

      if self.output_info.get('output_config'):
        output_config = self.output_info['output_config']
        self.concepts_mutually_exclusive = output_config['concepts_mutually_exclusive']
        self.closed_environment = output_config['closed_environment']

        if output_config.get('hyper_parameters'):
          self.hyper_parameters = json.loads(output_config['hyper_parameters'])
      else:
        self.concepts_mutually_exclusive = False
        self.closed_environment = False

      if self.output_info.get('data', {}).get('concepts'):
        for concept in self.output_info['data']['concepts']:
          concept = Concept(concept_name=concept['name'],
                            concept_id=concept['id'],
                            app_id=concept['app_id'],
                            created_at=concept['created_at'])
          self.concepts.add(concept)

  def get_info(self, verbose=False):
    ''' get model info, with or without the concepts associated with the model.

    Args:
      verbose: default is False. True will yield output_info, with concepts of the model

    Returns:
      raw json of the response

    Examples:
      >>> # with basic model info
      >>> model.get_info()
      >>> # model info with concepts
      >>> model.get_info(verbose=True)
    '''

    if verbose is False:
      ret = self.api.get_model(self.model_id)
    else:
      ret = self.api.get_model_output_info(self.model_id)

    return ret

  def get_concept_ids(self):
    ''' get concepts IDs associated with the model

    Args:
      Void

    Returns:
      a list of concept IDs

    Examples:
      >>> ids = model.get_concept_ids()
    '''

    if self.concepts:
      concepts = [c.dict() for c in self.concepts]
    else:
      res = self.get_info(verbose=True)
      concepts = res['model']['output_info'].get('data', {}).get('concepts', [])

    return [c['id'] for c in concepts]

  def dict(self):

    data = {
             "model": {
               "name": self.model_name,
               "output_info": {
                 "output_config": {
                   "concepts_mutually_exclusive": self.concepts_mutually_exclusive,
                   "closed_environment": self.closed_environment
                 }
               }
             }
           }

    if self.model_id:
      data['model']['id'] = self.model_id

    if self.concepts:
      ids = [{"id": concept_id} for concept_id in self.concepts]
      data['model']['output_info']['data'] = { "concepts": ids }

    return data

  def train(self, sync=True, timeout=60):
    '''
    train the model in synchronous or asynchronous mode. Synchronous will block until the model is trained, async will not.

    Args:
      sync: indicating synchronous or asynchronous, default is True

    Returns:
      the Model object

    '''

    res = self.api.create_model_version(self.model_id)

    status = res['status']
    if status['code'] == 10000:
      model_id = res['model']['id']
      model_version = res['model']['model_version']['id']
      model_status_code = res['model']['model_version']['status']['code']
    else:
      return res

    if sync is False:
      return res

    # train in sync despite the RESTful api is always async
    # will loop until the model is trained
    # 21103: queued for training
    # 21101: being trained

    wait_interval = 1
    time_start = time.time()

    while model_status_code == 21103 or model_status_code == 21101:

      elapsed = time.time() - time_start
      if elapsed > timeout:
        break

      if elapsed < 10:
        wait_interval = 1
      elif elapsed < 60:
        wait_interval = 5
      elif elapsed < 120:
        wait_interval = 10
      elif elapsed < 180:
        wait_interval = 20
      elif elapsed < 240:
        wait_interval = 30
      else:
        wait_interval = 60

      time.sleep(wait_interval)
      res_ver = self.api.get_model_version(model_id, model_version)
      model_status_code = res_ver['model_version']['status']['code']

    res['model']['model_version'] = res_ver['model_version']

    model = self._to_obj(res['model'])
    return model

  def predict_by_url(self, url, lang=None, is_video=False,
                     min_value=None, max_concepts=None, select_concepts=None):
    ''' predict a model with url

    Args:
      url: publicly accessible url of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    '''

    url = url.strip()

    if is_video is True:
      input = Video(url=url)
    else:
      input = Image(url=url)

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang,
                                                                        min_value=min_value,
                                                                        max_concepts=max_concepts,
                                                                        select_concepts=select_concepts))

    res = self.predict([input], model_output_info)
    return res

  def predict_by_filename(self, filename, lang=None, is_video=False,
                          min_value=None, max_concepts=None, select_concepts=None):
    ''' predict a model with a local filename

    Args:
      filename: filename on local filesystem
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    '''

    fileio = open(filename, 'rb')

    if is_video is True:
      input = Video(file_obj=fileio)
    else:
      input = Image(file_obj=fileio)

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang,
                                                                        min_value=min_value,
                                                                        max_concepts=max_concepts,
                                                                        select_concepts=select_concepts))

    res = self.predict([input], model_output_info)
    return res

  def predict_by_bytes(self, raw_bytes, lang=None, is_video=False,
                       min_value=None, max_concepts=None, select_concepts=None):
    ''' predict a model with image raw bytes

    Args:
      raw_bytes: raw bytes of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    '''

    base64_bytes = base64.b64encode(raw_bytes)

    if is_video is True:
      input = Video(base64=base64_bytes)
    else:
      input = Image(base64=base64_bytes)

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang,
                                                                        min_value=min_value,
                                                                        max_concepts=max_concepts,
                                                                        select_concepts=select_concepts))

    res = self.predict([input], model_output_info)
    return res

  def predict_by_base64(self, base64_bytes, lang=None, is_video=False,
                        min_value=None, max_concepts=None, select_concepts=None):
    ''' predict a model with base64 encoded image bytes

    Args:
      base64_bytes: base64 encoded image bytes
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    '''

    if is_video is True:
      input = Video(base64=base64_bytes)
    else:
      input = Image(base64=base64_bytes)

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang,
                                                                        min_value=min_value,
                                                                        max_concepts=max_concepts,
                                                                        select_concepts=select_concepts))

    res = self.predict([input], model_output_info)
    return res

  def predict(self, inputs, model_output_info=None):
    ''' predict with multiple images

    Args:
      inputs: a list of Image objectsg

    Returns:
      the prediction of the model in JSON format
    '''

    res = self.api.predict_model(self.model_id, inputs, self.model_version, model_output_info)
    return res

  def merge_concepts(self, concept_ids, overwrite=False):
    ''' merge concepts in a model

    When overwrite is False, if the concept does not exist in the model it will be appended.
    Otherwise, the original one will be kept.

    Args:
      concept_ids: a list of concept id
      overwrite: True or False. If True, the existing concepts will be replaced

    Returns:
      the Model object
    '''

    if overwrite is True:
      action = 'overwrite'
    else:
      action = 'merge'

    model = self.update(action=action, concept_ids=concept_ids)
    return model

  def add_concepts(self, concept_ids):
    ''' merge concepts into a model

    This is just an alias of `merge_concepts`, for easier understanding of adding new concepts
    to the model without overwritting them

    Args:
      concept_ids: a list of concept IDs

    Returns:
      the Model object

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.add_concepts(['cat', 'dog'])
    '''

    return self.merge_concepts(concept_ids)

  def update(self, action='merge', model_name=None, concepts_mutually_exclusive=None, \
             closed_environment=None, concept_ids=None):
    '''
    Update the model attributes. The name of the model, list of concepts, and
    the attributes ``concepts_mutually_exclusive`` and ``closed_environment`` can
    be changed.
    Note this is a overwriting change. For a valid call, at least one or more attributes should be
    specified. Otherwise the call will be just skipped without error.

    Args:
      action: the way to patch the model: ['merge', 'remove', 'overwrite']
      model_name: name of the model
      concepts_mutually_exclusive: whether the concepts are mutually exclusive
      closed_environment: whether negative concepts should be taken into account during training
      concept_ids: a list of concept ids

    Returns:
      the Model object

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.update(model_name="new_model_name")
      >>> model.update(concepts_mutually_exclusive=False)
      >>> model.update(closed_environment=True)
      >>> model.update(concept_ids=["bird", "hurd"])
      >>> model.update(concepts_mutually_exclusive=True, concept_ids=["bird", "hurd"])
    '''

    args = [model_name, concepts_mutually_exclusive, closed_environment, concept_ids]
    if not any(map(lambda x: x is not None, args)):
      return self

    model = {"id": self.model_id,
             "output_info": {
               "output_config": {},
               "data": {}
             }
            }

    if model_name:
      model["name"] = model_name

    if concepts_mutually_exclusive is not None:
      model["output_info"]["output_config"]["concepts_mutually_exclusive"] = concepts_mutually_exclusive

    if closed_environment is not None:
      model["output_info"]["output_config"]["closed_environment"] = closed_environment

    if concept_ids is not None:
      model["output_info"]["data"]["concepts"] = [{"id": concept_id} for concept_id in concept_ids]

    res = self.api.patch_model(model, action)
    model = res['models'][0]
    return self._to_obj(model)

  def delete_concepts(self, concept_ids):
    ''' delete concepts from a model

    Args:
      concept_ids: a list of concept IDs to be removed

    Returns:
      the Model object

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.delete_concepts(['cat', 'dog'])
    '''

    model = self.update(action='remove', concept_ids=concept_ids)
    return model

  def list_versions(self):
    ''' list all model versions

    Args:
      void

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.list_versions()
    '''

    res = self.api.get_model_versions(self.model_id)
    return res

  def get_version(self, version_id):
    ''' get model version info for a particular version

    Args:
      version_id: version id of the model version

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.get_version('model_version_id')
    '''

    res = self.api.get_model_version(self.model_id, version_id)
    return res

  def delete_version(self, version_id):
    ''' delete model version by version_id

    Args:
      version_id: version id of the model version

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.delete_version('model_version_id')
    '''

    res = self.api.delete_model_version(self.model_id, version_id)
    return res

  def create_version(self):

    res = self.api.create_model_version(self.model_id)
    return res

  def get_inputs(self, version_id=None, page=1, per_page=20):
    '''
    Get all the inputs from the model or a specific model version.
    Without specifying a model version id, this will yield all inputs

    Args:
      version_id: model version id
      page: page number
      per_page: number of inputs to return for each page

    Returns:
      A list of Input objects
    '''

    res = self.api.get_model_inputs(self.model_id, version_id, \
                                    page, per_page)

    return res

  def send_concept_feedback(self, input_id, url, concepts=None, not_concepts=None, feedback_info=None):
    '''
    Send feedback for this model

    Args:
      input: input for the feedback

    Returns:
      None
    '''

    feedback_input = Image(url=url, image_id=input_id, concepts=concepts, not_concepts=not_concepts, feedback_info=feedback_info)
    res = self.api.send_model_feedback(self.model_id, self.model_version, feedback_input)

    return res

  def send_region_feedback(self, input_id, url, concepts=None, not_concepts=None, regions=None, feedback_info=None):
    '''
    Send feedback for this model

    Args:
      input: input for the feedback

    Returns:
      None
    '''

    feedback_input = Image(url=url, image_id=input_id, concepts=concepts, not_concepts=not_concepts,
                           regions=regions,
                           feedback_info=feedback_info)
    res = self.api.send_model_feedback(self.model_id, self.model_version, feedback_input)

    return res

  def _to_obj(self, item):
    ''' convert a model json object to Model object '''
    return Model(self.api, item)


class Concept(object):

  """ Clarifai Concept
  """

  def __init__(self, concept_name=None, concept_id=None, app_id=None, created_at=None, value=None):
    self.concept_name = concept_name
    self.concept_id = concept_id
    self.app_id = app_id
    self.created_at = created_at
    self.value = value

  def dict(self):

    data = {}

    if self.concept_name is not None:
      data['name'] = self.concept_name

    if self.concept_id is not None:
      data['id'] = self.concept_id

    if self.app_id is not None:
      data['app_id'] = self.app_id

    if self.created_at is not None:
      data['created_at'] = self.created_at

    if self.value is not None:
      data['value'] = self.value

    return data


class ApiClient(object):
  """ Handles auth and making requests for you.

  The constructor for API access. You must sign up at developer.clarifai.com first and create an
  application in order to generate your credentials for API access.

  Args:
    self: instance of ApiClient
    app_id: the app_id for an application you've created in your Clarifai account.
    app_secret: the app_secret for the same application.
    base_url: Base URL of the API endpoints.
    quiet: if True then silence debug prints.
  """

  patch_actions = ['merge', 'remove', 'overwrite']
  concepts_patch_actions = ['overwrite']

  def __init__(self, app_id=None, app_secret=None, base_url=None, api_key=None, quiet=True):

    if platform.system() == 'Windows':
      homedir = os.environ.get('HOMEPATH', '.')
    else:
      homedir = os.environ.get('HOME', '.')

    CONF_FILE=os.path.join(homedir, '.clarifai', 'config')

    if api_key is None:
      if os.environ.get('CLARIFAI_API_KEY'):
        logger.debug("Using env variables for api_key")
        api_key_str = os.environ['CLARIFAI_API_KEY']
      elif os.path.exists(CONF_FILE):
        parser = ConfigParser()
        parser.optionxform = str

        with open(CONF_FILE, 'r') as fdr:
          parser.readfp(fdr)

        if parser.has_option('clarifai', 'CLARIFAI_API_KEY'):
          api_key_str = parser.get('clarifai', 'CLARIFAI_API_KEY')
        else:
          api_key_str = ''
      else:
        api_key_str = ''

    if app_id is None:
      if os.environ.get('CLARIFAI_APP_ID') and os.environ.get('CLARIFAI_APP_SECRET'):
        logger.debug("Using env variables for id and secret")
        app_id_str = os.environ['CLARIFAI_APP_ID']
        app_secret_str = os.environ['CLARIFAI_APP_SECRET']
      elif os.path.exists(CONF_FILE):
        parser = ConfigParser()
        parser.optionxform = str

        with open(CONF_FILE, 'r') as fdr:
          parser.readfp(fdr)

        if parser.has_option('clarifai', 'CLARIFAI_APP_ID') and \
           parser.has_option('clarifai', 'CLARIFAI_APP_SECRET'):
          app_id_str = parser.get('clarifai', 'CLARIFAI_APP_ID')
          app_secret_str = parser.get('clarifai', 'CLARIFAI_APP_SECRET')
        else:
          app_id_str = app_secret_str = ''
      else:
        app_id_str = app_secret_str = ''

    if base_url is None:
      if os.environ.get('CLARIFAI_API_BASE'):
        base_url_str = os.environ.get('CLARIFAI_API_BASE')
      elif os.path.exists(CONF_FILE):
        parser = ConfigParser()
        parser.optionxform = str

        with open(CONF_FILE, 'r') as fdr:
          parser.readfp(fdr)

        if parser.has_option('clarifai', 'CLARIFAI_API_BASE'):
          base_url_str = parser.get('clarifai', 'CLARIFAI_API_BASE')
        else:
          base_url_str = 'api.clarifai.com'
      else:
        base_url_str = 'api.clarifai.com'
    else:
      base_url_str = base_url

    if app_id and app_secret:
      self.app_id = app_id
      self.app_secret = app_secret
      self.api_key = ''
    elif api_key:
      self.api_key = api_key
      self.app_id = self.app_secret = ''
    else:
      self.app_id = app_id_str
      self.app_secret = app_secret_str
      self.api_key = api_key_str

    if quiet:
      logger.setLevel(logging.INFO)
    else:
      logger.setLevel(logging.DEBUG)

    parsed = urlparse(base_url_str)
    scheme = 'https' if parsed.scheme == '' else parsed.scheme
    base_url_parsed = parsed.path if not parsed.netloc else parsed.netloc
    self.base_url = base_url_parsed
    self.scheme = scheme
    self.basev2 = urljoin(scheme + '://', base_url_parsed)
    logger.debug("Base url: %s", self.basev2)
    self.token = None
    self.headers = None

    # Make sure when you create a client, it's ready for requests.
    self.get_token()

  def get_token(self):
    ''' Get an access token using your app_id and app_secret.

    You shouldn't need to call this method yourself. If there is no access token yet, this method
    will be called when a request is made. If a token expires, this method will also automatically
    be called to renew the token.

    '''

    if self.api_key:
      self.token = None
      self.headers = {'Authorization': "Key %s" % self.api_key}
      return {}

    data = {'grant_type': 'client_credentials'}
    auth = (self.app_id, self.app_secret)
    logger.debug("get_token: %s data: %s", self.basev2 + '/v2/token', data)

    authurl = urljoin(self.scheme + '://', self.base_url, 'v2', 'token')
    res = requests.post(authurl, auth=auth, data=data)
    if res.status_code == 200:
      logger.debug("Got V2 token: %s", res.json())
      self.token = res.json()['access_token']
      self.headers = {'Authorization': "Bearer %s" % self.token}
    else:
      raise TokenError("Could not get a new token for v2: %s", str(res.json()))
    return res.json()

  def set_token(self, token):
    ''' manually set the token to this client

    You shouldn't need to call this unless you know what you are doing, because the client handles
    the token generation and refersh for you. This is only intended for debugging purpose when you
    want to verify the token got from somewhere else.
    '''
    self.token = token

  def delete_token(self):
    ''' manually reset the token to empty

    You shouldn't need to call this unless you know what you are doing, because the client handles
    the token generation and refersh for you. This is only intended for debugging purpose when you
    want to reset the token.
    '''
    self.token = None

  def _check_token(self):
    ''' set the token when it is empty

    This function is called at every API call to check if the token is set.
    If it is not set, a token call will be issued and the token will be
    refreshed.
    '''

    if self.token is None:
      self.get_token()

  def _requester(self, resource, params, method, version="v2"):
    ''' Obtains info and verifies user via Token Decorator

    Args:
      resource:
      params: parameters passed to the request
      version: v1 or v2
      method: GET or POST or DELETE or PATCH

    Returns:
      JSON from user request
    '''

    self._check_token()
    url = urljoin(self.basev2, version, resource)

    # only retry under when status_code is non-200, under max-tries
    # and under some circumstances
    status_code = 199
    retry = True
    max_attempts = attempts = 3
    headers = {}

    while status_code != 200 and attempts > 0 and retry is True:

      logger.debug("=" * 100)

      # mangle the base64 because it is too long
      if params and params.get('inputs') and len(params['inputs']) > 0:
        params_copy = copy.deepcopy(params)
        for data in params_copy['inputs']:
          data = data['data']
          if data.get('image') and data['image'].get('base64'):
            base64_bytes = data['image']['base64'][:10] + '......' + data['image']['base64'][-10:]
            data['image']['base64'] = base64_bytes
          if data.get('video') and data['video'].get('base64'):
            base64_bytes = data['video']['base64'][:10] + '......' + data['video']['base64'][-10:]
            data['video']['base64'] = base64_bytes
      elif params and params.get('query') and params['query'].get('ands'):
        params_copy = copy.deepcopy(params)

        queries = params_copy['query']['ands']

        for query in queries:
          if query.get('output') and query['output'].get('input') and \
                  query['output']['input'].get('data') and \
                  query['output']['input']['data'].get('image') and \
                  query['output']['input']['data']['image'].get('base64'):
            data = query['output']['input']['data']
            base64_bytes = data['image']['base64'][:10] + '......' + data['image']['base64'][-10:]
            data['image']['base64'] = base64_bytes
      else:
        params_copy = params
      # mangle the base64 because it is too long

      logger.debug("%s %s\nHEADERS:\n%s\nPAYLOAD:\n%s",
                   method, url, pformat(headers), pformat(params_copy))

      if method == 'GET':
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Python-Client': '%s:%s' % (OS_VER, PYTHON_VERSION),
                   'Authorization': self.headers['Authorization']}
        res = requests.get(url, params=params, headers=headers)
      elif method == "POST":
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Python-Client': '%s:%s' % (OS_VER, PYTHON_VERSION),
                   'Authorization': self.headers['Authorization']}
        res = requests.post(url, data=json.dumps(params), headers=headers)
      elif method == "DELETE":
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Python-Client': '%s:%s' % (OS_VER, PYTHON_VERSION),
                   'Authorization': self.headers['Authorization']}
        res = requests.delete(url, data=json.dumps(params), headers=headers)
      elif method == "PATCH":
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Python-Client': '%s:%s' % (OS_VER, PYTHON_VERSION),
                   'Authorization': self.headers['Authorization']}
        res = requests.patch(url, data=json.dumps(params), headers=headers)
      else:
        raise UserError("Unsupported request type: '%s'" % method)

      try:
        js = res.json()
      except Exception:
        logger.exception("Could not get valid JSON from server response.")
        logger.debug("\nRESULT:\n%s", pformat(res.content.decode('utf-8')))
        return res

      logger.debug("\nRESULT:\n%s", pformat(json.loads(res.content.decode('utf-8'))))

      status_code = res.status_code
      attempts -= 1

      # allow retry when token expires
      # normally, this should be solved in one retry
      if status_code == 401 and isinstance(js, dict) and js.get('status', {}).get('details', '') == "expired token":
        self.get_token()
        retry = True
        continue

      # handle Gateway Error, normally retry will solve the problem
      if status_code == 502 or status_code == 503 or status_code == 500:
        retry = True
        continue

      # handle throttling
      # back off with 2/4/8 seconds
      # normally, this will be settled in 1 or 2 retries
      if status_code == 429:
        retry = True
        time.sleep(pow(2, max_attempts - attempts - 1))
        continue

      # in other cases, error out without retrying
      retry = False

    if res.status_code != 200:
      logger.debug("\nRESULT:\n%s", pformat(json.loads(res.content.decode('utf-8'))))
      raise ApiError(resource, params, method, res)

    return res.json()

  def get(self, resource, params=None, version="v2"):
    ''' Authorized get from Clarifai's API. '''
    return self._requester(resource, params, 'GET', version)

  def post(self, resource, params=None, version="v2"):
    ''' Authorized post to Clarifai's API. '''
    return self._requester(resource, params, 'POST', version)

  def delete(self, resource, params=None, version="v2"):
    ''' Authorized get from Clarifai's API. '''
    return self._requester(resource, params, 'DELETE', version)

  def patch(self, resource, params=None, version="v2"):
    ''' Authorized patch from Clarifai's API '''
    return self._requester(resource, params, 'PATCH', version)

  def add_inputs(self, objs):
    ''' Add a list of Images or Videos to an application.

    Args:
      obj: A list of Image or Video objects.

    Returns:
      raw JSON response from the API server, with a list of inputs and corresponding import status
    '''
    if not isinstance(objs, list):
      raise UserError("objs must be a list")

    for obj in objs:
      if not isinstance(obj, (Image, Video)):
        raise UserError("Not valid type of content to add. Must be Image or Video")
      if obj.input_id is not None and not isinstance(obj.input_id, basestring):
        raise UserError("Not valid input ID. Must be a string or None")
      if obj.input_id is not None and '/' in obj.input_id:
        raise UserError("Not valid input ID. Cannot contain character: \"/\"")



    resource = "inputs"
    data = {"inputs": [obj.dict() for obj in objs]}
    res = self.post(resource, data)
    return res

  def search_inputs(self, query, page=1, per_page=20):
    ''' Search an application and get predictions (optional)

    Args:
      query: the JSON query object that complies with Clarifai RESTful API
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      raw JSON response from the API server, with a list of inputs and corresponding ranking scores
    '''

    resource = "searches/"

    # Similar image search and predictions
    d = {'pagination': pagination(page, per_page).dict(),
         'query': query
        }

    res = self.post(resource, d)
    return res

  def get_input(self, input_id):
    ''' Get a single image by it's id.

    Args:
      input_id: the id of the Image.

    Returns:
      raw JSON response from the API server

      HTTP code:
       200 for Found
       404 for Not Found
    '''

    resource = "inputs/%s" % input_id
    res = self.get(resource)
    return res

  def get_inputs(self, page=1, per_page=20):
    ''' List all images for the Application, with pagination

    Args:
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      raw JSON response from the API server, with paginated list of inputs and corresponding status
    '''

    resource = "inputs"
    d = {'page': page, 'per_page': per_page}
    res = self.get(resource, d)
    return res

  def get_inputs_status(self):
    ''' Get counts of inputs in the Application.

    Returns:
      counts of the inputs, including processed, processing, etc. in JSON format.
    '''

    resource = "inputs/status"
    res = self.get(resource)
    return res

  def delete_input(self, input_id):
    ''' Delete a single input by its id.

    Args:
      input_id: the id of the input

    Returns:
      status of the deletion, in JSON format.
    '''

    if not input_id:
      raise UserError('cannot delete with empty input_id. \
                       use delete_all_inputs if you want to delete all')

    resource = "inputs/%s" % input_id
    res = self.delete(resource)
    return res

  def delete_inputs(self, input_ids):
    ''' bulk delete inputs with a list of input IDs

    Args:
      input_ids: the ids of the input, in a list

    Returns:
      status of the bulk deletion, in JSON format.
    '''

    resource = "inputs"
    data = {"ids": [input_id for input_id in input_ids]}

    res = self.delete(resource, data)
    return res

  def delete_all_inputs(self):
    ''' delete all inputs from the application

    Returns:
      status of the deletion, in JSON format.
    '''

    resource = "inputs"
    data = {"delete_all":True}

    res = self.delete(resource, data)
    return res

  def patch_inputs(self, action, inputs):
    ''' bulk update inputs, to delete or modify concepts

    Args:
      action: "merge" or "remove" or "overwrite"
      input_ids: list of input IDs
      concept_ids_pairs: For "merge_concepts", this is a list of (concept_id, value) tuples
                           where value is either True or False
                         For "delete_concepts", this is a list of concept ids

    Returns:
      the update status, in JSON format

    '''

    if action not in self.patch_actions:
      raise UserError("action not supported.")

    resource = "inputs"
    data = {
             "action": action,
             "inputs": []
           }

    images = []
    for img in inputs:
      item = img.dict()
      if not item.get('data'):
        continue

      new_item = copy.deepcopy(item)
      for key in item['data'].keys():
        if key not in ['concepts', 'metadata']:
          del new_item['data'][key]

      images.append(new_item)

    data["inputs"] = images

    res = self.patch(resource, data)
    return res

  def get_outputs(self, input_id):
    ''' Get output predictions for an input

    Args:
      input_id: the unique identifier for an input

    Returns:
      the input with output predictions in a json object
    '''

    resource = "inputs/%s/outputs" % input_id

    res = self.get(resource)
    return res

  def patch_outputs(self, input_id, action, concept_ids):
    ''' Patch predictions

    Args:
      input_id: the unique identifier of the input
      action: 'remove' or 'merge'
      concept_ids: the list of concept ids that will be removed or merged

    Returns:
      the patched input
    '''

    resource = "inputs/%s/outputs" % input_id
    patch_value = 1 if action == 'merge' else 0

    data = {
             "outputs": [
              {
                "data": {
                  "concepts": [ { "id": cid, "value": patch_value } for cid in concept_ids]
                },
                "model": {
                  "id": "aa9ca48295b37401f8af92ad1af0d91d"
                }
              }
             ],
             "action": action
           }

    res = self.patch(resource, data)
    return res

  def get_concept(self, concept_id):
    ''' Get a single concept by it's id.

    Args:
      concept_id: unique id of the concept

    Returns:
      the concept in JSON format with HTTP 200 Status
      or HTTP 404 with concept not found
    '''

    resource = "concepts/%s" % concept_id
    res = self.get(resource)
    return res

  def get_concepts(self, page=1, per_page=20):
    ''' List all concepts for the Application.

    Args:
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      a list of concepts in JSON format
    '''

    resource = "concepts"
    d = {'page': page, 'per_page': per_page}
    res = self.get(resource, d)
    return res

  def add_concepts(self, concept_ids, concept_names):
    ''' Add a list of concepts

    Args:
      concept_ids: a list of concept id
      concept_names: a list of concept name

    Returns:
      a list of concepts in JSON format along with the status code
    '''

    if not isinstance(concept_ids, list) or \
       not isinstance(concept_names, list):
      raise UserError('concept_ids and concept_names should be both be list ')

    if len(concept_ids) != len(concept_names):
      raise UserError('length of concept id list should match length of the concept name list')

    resource = "concepts"
    d = {'concepts':[]}

    for cid, cname in zip(concept_ids, concept_names):
      if cname is None:
        concept = {'id':cid}
      else:
        concept = {'id':cid,'name':cname}

      d['concepts'].append(concept)

    res = self.post(resource, d)
    return res

  def search_concepts(self, term, page=1, per_page=20, language=None):
    ''' Search concepts

    Args:
      term: search term with wildcards
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page
      language: language to search for the translation

    Returns:
      a list of concepts in JSON format along with the status code

    '''

    resource = "concepts/searches/"

    # Similar image search and predictions
    d = {'pagination': pagination(page, per_page).dict()}

    d.update({
               "concept_query": {
                 "name":term
               }
             })

    if language is not None:
      d['concept_query']['language'] = language

    res = self.post(resource, d)
    return res

  def patch_concepts(self, action, concepts):
    ''' bulk update concepts, to delete or modify concepts

    Args:
      action: only "overwrite" is supported
      concepts: a list of Concept(concept_name='', concept_id='')

    Returns:
      the update status, in JSON format

    '''

    if action not in self.concepts_patch_actions:
      raise UserError("action not supported.")

    resource = "concepts"
    data = {
      "action": action,
      "concepts": []
    }

    concepts_items = []
    for concept in concepts:
      item = concept.dict()
      if not item.get('id') or not item.get('name'):
        continue

      new_item = copy.deepcopy(item)
      for key in item.keys():
        if key not in ['id', 'name']:
          del new_item[key]

      concepts_items.append(new_item)

    data["concepts"] = concepts_items

    res = self.patch(resource, data)
    return res

  def get_models(self, page=1, per_page=20):
    ''' get all models with pagination

    Args:
      page: page number
      per_page: number of models to return per page

    Returns:
      a list of models in JSON format
    '''

    resource = "models"
    params = {'page': page,
              'per_page': per_page
             }

    res = self.get(resource, params)
    return res

  def get_model(self, model_id=None):
    ''' get model basic info by model id

    Args:
      model_id: the unique identifier of the model

    Returns:
      the model info in JSON format
    '''

    resource = "models/%s" % _escape(model_id)
    res = self.get(resource)
    return res

  def get_model_output_info(self, model_id=None):
    ''' get model output info by model id

    Args:
      model_id: the unique identifier of the model

    Returns:
      the model info with output_info in JSON format
    '''

    resource = "models/%s/output_info" % _escape(model_id)

    res = self.get(resource)
    return res

  def get_model_versions(self, model_id, page=1, per_page=20):
    ''' get model versions

    Args:
      model_id: the unique identifier of the model
      page: page number
      per_page: the number of versions to return per page

    Returns:
      a list of model versions in JSON format
    '''

    resource = "models/%s/versions" % _escape(model_id)
    params = {'page': page,
              'per_page': per_page
             }

    res = self.get(resource, params)
    return res

  def get_model_version(self, model_id, version_id):
    ''' get model info for a specific model version

    Args:
      model_id: the unique identifier of a model
      version_id: the model version id
    '''

    resource = "models/%s/versions/%s" % (model_id, version_id)

    res = self.get(resource)
    return res

  def delete_model_version(self, model_id, model_version):
    ''' delete a model version '''

    resource = "models/%s/versions/%s" % (_escape(model_id), model_version)
    res = self.delete(resource)
    return res

  def delete_model(self, model_id):
    ''' delete a model '''

    resource = "models/%s" % _escape(model_id)
    res = self.delete(resource)
    return res

  def delete_all_models(self):
    ''' delete all models '''

    resource = "models"
    data = {"delete_all":True}

    res = self.delete(resource, data)
    return res

  def get_model_inputs(self, model_id, version_id=None, page=1, per_page=20):
    ''' get inputs for the latest model or a specific model version '''

    if not version_id:
      resource = "models/%s/inputs?page=%d&per_page=%d" % \
                 (_escape(model_id), page, per_page)
    else:
      resource = "models/%s/version/%s/inputs?page=%d&per_page=%d" % \
                 (_escape(model_id), version_id, page, per_page)

    res = self.get(resource)
    return res

  def search_models(self, name=None, model_type=None):
    ''' search model by name and type '''

    resource = "models/searches"

    if name is not None and model_type is not None:
      data = {"model_query": {
                "name": name,
                "type": model_type
                }
             }
    elif name is None and model_type is not None:
      data = {"model_query": {
                "type": model_type
                }
             }
    elif name is not None and model_type is None:
      data = {"model_query": {
                "name": name
                }
             }
    else:
      data = {}

    res = self.post(resource, data)
    return res

  def create_model(self, model_id, model_name=None, concepts=None, \
                   concepts_mutually_exclusive=False, \
                   closed_environment=False, \
                   hyper_parameters=None):
    ''' create custom model '''

    if not model_name:
      model_name = model_id

    resource = "models"

    data = {
             "model": {
               "id": model_id,
               "name": model_name,
               "output_info": {
                 "output_config": {
                   "concepts_mutually_exclusive": concepts_mutually_exclusive,
                   "closed_environment": closed_environment
                 }
               }
             }
           }

    if concepts:
      data['model']['output_info']['data'] = { "concepts":
                                                 [{"id": concept} for concept in concepts]
                                             }
    if hyper_parameters:
      try:
        data['model']['output_info']['output_config']['hyper_parameters'] = json.dumps(hyper_parameters)
      except ValueError:
        pass

    res = self.post(resource, data)
    return res

  def patch_model(self, model, action='merge'):

    if action not in self.patch_actions:
      raise UserError("action not supported.")

    resource = "models"
    data = {
             "action": action,
             "models": [model]
           }

    res = self.patch(resource, data)
    return res

  def create_model_version(self, model_id):
    ''' train for a model '''

    resource = "models/%s/versions" % _escape(model_id)

    res = self.post(resource)
    return res

  def predict_model(self, model_id, objs, version_id=None, model_output_info=None):

    if version_id is None:
      resource = "models/%s/outputs" % _escape(model_id)
    else:
      resource = "models/%s/versions/%s/outputs" % (_escape(model_id), version_id)

    if not isinstance(objs, list):
      raise UserError("objs must be a list")
    if not isinstance(objs[0], (Image, Video)):
      raise UserError("Not valid type of content to add. Must be Image or Video")

    data = {"inputs": [obj.dict() for obj in objs]}

    if model_output_info is not None:
      data.update({'model':model_output_info.dict()})

    res = self.post(resource, data)
    return res

  def send_model_feedback(self, model_id, version_id, obj):

    if version_id is None:
      resource = "models/%s/feedback" % _escape(model_id)
    else:
      resource = "models/%s/versions/%s/feedback" % (_escape(model_id), version_id)

    data = {"input": obj.dict()}

    res = self.post(resource, data)
    return res

  def send_search_feedback(self, obj):

    resource = "searches/feedback"

    data = {"input": obj.dict()}

    res = self.post(resource, data)
    return res

  def predict_concepts(self, objs, lang=None):

    models = self.search_models(name='general-v1.3', model_type='concept')
    model = models['models'][0]
    model_id = model['id']

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang))
    return self.predict_model(model_id, objs, model_output_info=model_output_info)

  def predict_colors(self, objs):

    models = self.search_models(name='color', model_type='color')
    model = models['models'][0]
    model_id = model['id']

    return self.predict_model(model_id, objs)

  def predict_embed(self, objs, model='general-v1.3'):

    models = self.search_models(name=model, model_type='embed')
    model = models['models'][0]
    model_id = model['id']

    return self.predict_model(model_id, objs)


class pagination(object):

  def __init__(self, page=1, per_page=20):
    self.page = page
    self.per_page = per_page
  def dict(self):
    return {'page': self.page, 'per_page': self.per_page}


class TokenError(Exception):
  pass


class ApiError(Exception):
  """ API Server error """

  def __init__(self, resource, params, method, response):
    self.resource = resource
    self.params = params
    self.method = method
    self.response = response

    self.error_code = response.json().get('status', {}).get('code', None)
    self.error_desc = response.json().get('status', {}).get('description', None)
    self.error_details = response.json().get('status', {}).get('details', None)

    current_ts_str = str(time.time())

    msg = """%(method)s /%(resource)s FAILED(%(time_ts)s). status_code: %(status_code)d, reason: %(reason)s, error_code: %(error_code)s, error_description: %(error_desc)s, error_details: %(error_details)s
 >> REQUEST(%(time_ts)s) %(request)s
 >> RESPONSE(%(time_ts)s) %(response)s""" % {
        'method': method,
        'resource': resource,
        'status_code': response.status_code,
        'reason': response.reason,
        'error_code': self.error_code,
        'error_desc': self.error_desc,
        'error_details': self.error_details,
        'request': json.dumps(params),
        'response': json.dumps(response.json()),
        'time_ts': current_ts_str
    }

    super(ApiError, self).__init__(msg)

  # def __str__(self):
  #   parent_str = super(ApiError, self).__str__()
  #   return parent_str + str(self.json)


class ApiClientError(Exception):
  """ API Client Error """
  pass


class UserError(Exception):
  """ User Error """
  pass


class ApiStatus(object):
  """ Clarifai API Status Code """

  def __init__(self, item):
    self.code = item['code']
    self.description = item['description']

  def dict(self):
    d = {'status': { 'code': self.code,
                     'description': self.description
                   }
        }

    return d


class ApiResponse(object):
  """ Clarifai API Response """

  def __init__(self):
    self.status = None


class InputCounts(object):
  """ input counts for upload status """

  def __init__(self, item):

    if not item.get('counts'):
      raise ApiClient('unable to initialize. need a dict with key=counts')

    counts = item['counts']

    self.processed = counts['processed']
    self.to_process = counts['to_process']
    self.errors = counts['errors']

  def dict(self):
    d = { 'counts': {
                    'processed': self.processed,
                    'to_process': self.to_process,
                    'errors': self.errors
                   }
        }
    return d


class ModelOutputInfo(object):

  def __init__(self, concepts=None, output_config=None):
    self.concepts = concepts
    self.output_config = output_config

  def dict(self):
    data = {'output_info':{}}

    if self.output_config:
      data['output_info'].update(self.output_config.dict())

    if self.concepts:
      data = {'data':{'concepts':[concept.dict() for concept in self.concepts]}}
      data['output_info'].update()

    return data


class ModelOutputConfig(object):

  def __init__(self, mutually_exclusive=False, closed_environment=False, language=None,
               min_value=None, max_concepts=None, select_concepts=None):
    self.concepts_mutually_exclusive = mutually_exclusive
    self.closed_environment = closed_environment
    self.language = language
    self.min_value = min_value
    self.max_concepts = max_concepts
    self.select_concepts = select_concepts

  def dict(self):
    data = {'output_config':{
      'concepts_mutually_exclusive':self.concepts_mutually_exclusive,
      'closed_environment':self.closed_environment
    }}

    if self.language is not None:
      data['output_config']['language'] = self.language

    if self.min_value is not None:
      data['output_config']['min_value'] = self.min_value

    if self.max_concepts is not None:
      data['output_config']['max_concepts'] = self.max_concepts

    if self.select_concepts is not None:
      data['output_config']['select_concepts'] = [c.dict() for c in self.select_concepts]

    return data


class BoundingBox(object):

  def __init__(self, top_row, left_col, bottom_row, right_col):
    self.top_row = top_row
    self.left_col = left_col
    self.bottom_row = bottom_row
    self.right_col = right_col

  def dict(self):

    data = {'bounding_box': {
      'top_row': self.top_row,
      'left_col': self.left_col,
      'bottom_row': self.bottom_row,
      'right_col': self.right_col
    }}

    return data


class RegionInfo(object):

  def __init__(self, bbox=None, feedback_type=None):
    self.bbox = bbox
    self.feedback_type = feedback_type

  def dict(self):

    data = {"region_info": {}}

    if self.bbox:
      data['region_info'].update(self.bbox.dict())

    if self.feedback_type:
      if isinstance(self.feedback_type, FeedbackType):
        data['region_info']['feedback'] = self.feedback_type.name
      else:
        data['region_info']['feedback'] = self.feedback_type

    return data


class Region(object):

  def __init__(self, region_info, concepts=None, face=None):

    self.region_info = region_info
    self.concepts = concepts
    self.face = face

  def dict(self):

    data = {}
    data.update(self.region_info.dict())

    if self.concepts:
      data['data'] = {'concepts':[c.dict() for c in self.concepts]}

    if self.face:
      data['data'] = self.face.dict()

    return data


class Face(object):

  def __init__(self, identity=None, age_appearance=None, gender_appearance=None, multicultural_appearance=None):

    self.identity = identity
    self.age_appearance = age_appearance
    self.gender_appearance = gender_appearance
    self.multicultural_appearance = multicultural_appearance

  def dict(self):

    data = {'face':{}}

    if self.identity:
      data['face'].update(self.identity.dict())

    if self.age_appearance:
      data['face'].update(self.age_appearance.dict())

    if self.gender_appearance:
      data['face'].update(self.gender_appearance.dict())

    if self.multicultural_appearance:
      data['face'].update(self.multicultural_appearance.dict())

    return data

class FaceIdentity(object):

  def __init__(self, concepts):
    self.concepts = concepts

  def dict(self):
    data = {'identity': {
      'concepts': [c.dict() for c in self.concepts]
    }}
    return data


class FaceAgeAppearance(object):

  def __init__(self, concepts):
    self.concepts = concepts

  def dict(self):
    data = {'age_appearance': {
      'concepts': [c.dict() for c in self.concepts]
    }}
    return data


class FaceGenderAppearance(object):

  def __init__(self, concepts):
    self.concepts = concepts

  def dict(self):
    data = {'gender_appearance': {
      'concepts': [c.dict() for c in self.concepts]
    }}
    return data


class FaceMulticulturalAppearance(object):

  def __init__(self, concepts):
    self.concepts = concepts

  def dict(self):
    data = {'multicultural_appearance': {
      'concepts': [c.dict() for c in self.concepts]
    }}
    return data

