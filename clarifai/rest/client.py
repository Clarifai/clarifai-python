# -*- coding: utf-8 -*-

"""
Clarifai API Python Client
"""
 
import base64
import time
import json
import logging
import os
import requests
from configparser import ConfigParser
from io import BytesIO
from posixpath import join as urljoin
from future.moves.urllib.parse import urlparse

logger = logging.getLogger('clarifai')
logger.handlers = []
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

logging.getLogger("requests").setLevel(logging.WARNING)

CLIENT_VERSION = '2.0.6'


class ClarifaiApp(object):

  """ Clarifai Application Object

      The is the entry point of the Clarifai Client API
      With authentication to an application, you can access
      all the models, concepts, inputs in this application through
      the attributes of this class.

      To access the models: use app.models
      To access the inputs: use app.inputs
      To access the concepts: use app.concepts

  """

  def __init__(self, app_id=None, app_secret=None, base_url=None, quiet=False):

    self.api = ApiClient(app_id, app_secret, base_url)
    self.auth = Auth(self.api)

    self.concepts = Concepts(self.api)
    self.inputs = Inputs(self.api)
    self.models = Models(self.api)


class Auth(object):

  """ Clarifai Authentication

      This class is initialized as an attirbute of the clarifai application object
      with app.auth
  """

  def __init__(self, api):
    self.api = api

  def get_token(self):
    ''' get token string

    Returns:
      The token as a string
    '''

    res = self.api.get_token()
    token = res['access_token']
    return token


class Input(object):

  """ The Clarifai Input object
  """

  def __init__(self, input_id=None, concepts=None, not_concepts=None):
    ''' Construct an Image/Video object. it must have one of url or file_obj set.
    Args:
      input_id: unique id to set for the image. If None then the server will create and return one for
    you.
      concepts: a list of concepts this asset associate with
      not_concepts: a list of concepts this asset does not associate with
    '''

    self.input_id = input_id

    if not isinstance(concepts, (list, tuple)) and concepts is not None:
      raise UserError('concepts should be a list or tuple')

    if not isinstance(not_concepts, (list, tuple)) and not_concepts is not None:
      raise UserError('not_concepts should be a list or tuple')

    self.concepts = concepts
    self.not_concepts = not_concepts

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

    return data


class Image(Input):

  def __init__(self, url=None, file_obj=None, base64=None, crop=None, \
               image_id=None, concepts=None, not_concepts=None, \
               allow_dup_url=False):
    '''
      url: the url to a publically accessible image.
      file_obj: a file-like object in which read() will give you the bytes.
      crop: a list of float in the range 0-1.0 in the order [top, left, bottom, right] to crop out
            the asset before use.
    '''

    super(Image, self).__init__(image_id, concepts, not_concepts)

    if ((url is not None and file_obj is not None) or
        (url is not None and base64 is not None) or
        (file_obj is not None and base64 is not None) or
        (url is None and file_obj is None and base64 is None)):
      raise UserError("You must only set one of: [url, file_obj, base64] argumets.")

    if crop is not None and (not isinstance(crop, list) or len(crop) != 4):
      raise UserError("crop arg must be list of 4 floats or None")

    self.url = url
    self.file_obj = file_obj
    self.base64 = base64
    self.crop = crop
    self.allow_dup_url = allow_dup_url

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
      if hasattr(self.file_obj, 'getvalue'):
        base64_imgstr = base64.b64encode(self.file_obj.getvalue()).decode('UTF-8')
      elif hasattr(self.file_obj, 'read'):
        base64_imgstr = base64.b64encode(self.file_obj.read()).decode('UTF-8')
      else:
        raise UserError("Not sure how to read your file_obj")

      image['image']['base64'] = base64_imgstr
    elif self.base64 is not None:
      image['image']['base64'] = self.base64
    else:
      image['image']['url'] = self.url

    if self.crop is not None:
      image['image']['crop'] = self.crop

    image['image']['allow_duplicate_url'] = self.allow_dup_url

    data['data'].update(image)
    return data


class Video(Input):

  def __init__(self):
    raise Exception('Not supported yet.')

  def dict(self):
    pass


class SearchTerm(object):
  """
  Clarifai search term interface
  the base class for InputSearchTerm and OutputSearchTerm

  It is used to build SearchQueryBuilder
  """

  def __init__(self): 
    pass

  def dict(self):
    pass


class InputSearchTerm(SearchTerm):
  """
  Clarifai Image Search Input search term
  For input search, you can specify search term for url string match, input_id string match,
  concept string match, and concept_id string match
  value indicates whether the concept search is a NOT search

  Examples:
    >>> # search for url, string match
    >>> InputSearchTerm(url='http://blabla')
    >>> # search for input ID, string match
    >>> InputSearchTerm(input_id='site1_bla')
    >>> # search for annotated concept
    >>> InputSearchTerm(concept='tag1')
    >>> # search for not the annotated concept
    >>> InputSearchTerm(concept='tag1', value=False)
  """

  def __init__(self, url=None, input_id=None, concept=None, concept_id=None, value=True):
    self.url = url
    self.input_id = input_id
    self.concept = concept
    self.concept_id = concept_id
    self.value = value

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

    return obj

class OutputSearchTerm(SearchTerm):
  """
  Clarifai Image Search Output search term
  For output search, you can specify search term for url, base64, and input_id for
  visual search, 
  or specify concept and concept_id for string match
  value indicates whether the concept search is a NOT search

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

  def __init__(self, url=None, base64=None, input_id=None, concept=None, concept_id=None, value=True, crop=None):
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
  you should use the existing function search_by_annotated_concepts, search_by_predicted_concepts,
  or search_by_image and search_by_metadata

  Currently the query builder only supports a list of query terms with AND
  InputSearchTerm and OutputSearchTerm are the only term supported by the query builder

  Examples:
    >>> qb = SearchQueryBuilder()
    >>> qb.add_term(term1)
    >>> qb.add_term(term2)
    >>>
    >>> app.inputs.search(qb)

  """

  def __init__(self):
    self.terms = []

  def add_term(self, term):
    ''' add search term to the query
        this could be search by input or output
        construct the search_term with InputSearchTerm()
        and OutputSearchTerm()
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

    return query


class Models(object):

  def __init__(self, api):
    self.api = api

    # the cache of the model name -> model id mapping
    # to avoid an extra model query on every prediction by model name 
    self.model_id_cache = {}

  def clear_model_cache(self):
    ''' clear model_name -> model_id cache

        WARNING: This is an internal function, user should not call this

        We cache model_name to model_id mapping for API efficiency
        At the first time you call a models.get() by name, the name to ID
        mapping is saved so next time there is no query. Then user does not
        have to query the model ID every time when they want to work on it.

        Returns:
          There is no return result for this call
    '''

    self.model_id_cache = {}

  def create(self, model_id, model_name=None, concepts=None, \
             concepts_mutually_exclusive=False, \
             closed_environment=False):

    ''' create a new model

    Args:
      model_id: ID of the model
      model_name: optional name of the model
      concepts: optional concepts to associated with this model
      concepts_mutually_exclusive: True or False, whether concepts are mutually exclusive
      closed_environment: True or False, whether use negatives for prediction

    Returns:
      Model object

    Examples:
      >>> # create a model with no concepts
      >>> app.models.create('my_model1')
      >>> # create a model with a few concepts
      >>> app.models.create('my_model2', concepts=['bird', 'fish'])
    '''
    if not model_name:
      model_name = model_id

    res = self.api.create_model(model_id, model_name, concepts, \
                                concepts_mutually_exclusive, closed_environment)

    if res.get('model'):
      model = self._to_obj(res['model'])
    elif res.get('status'):
      status = res['status']
      raise UserError('code: %d, desc: %s, details: %s' % (status['code'], status['description'], status['details']))

    return model

  def get_all(self):
    ''' get all models in the application

    Returns:
      a generator that yields Model object

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
        yield model

      page += 1

  def get_by_page(self, page=1, per_page=20):
    ''' get paginated models from the application

        When the number of models get high, you may want to get
        the paginated results from all the models

        Args:
          page: page number
          per_page: number of models returned in one page

        Returns:
          a list of Model objects

        Examples:
          >>> models = app.models.get_by_page(2, 20)
    '''

    res = self.api.get_models(page, per_page)
    results = [self._to_obj(one) for one in res['models']]

    return results

  def delete(self, model_id, version_id=None):
    ''' delete the model, or a specific version of the model

        Without model version id specified, it is to delete a
        model. Then all the versions associated with this model
        will be deleted as well.

        With model version id specified, it is to delete a
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
    ''' delete all models and the versions associated with each one

        After this operation, you will have no model in the
        application

        Returns:
          the raw JSON response from the server

        Examples:
          >>> app.models.delete_all()
    '''

    res = self.api.delete_all_models()
    return res

  def get(self, model_id, model_type='concept'):
    ''' get a model, by ID or name

    Args:
      model_id: unique identifier of a model
      model_type: type of the model

    Returns:
      the Model object
 
    Examples:
      >>> # get general-v1.3 model
      >>> app.models.get('general-v1.3')
    '''

    if self.model_id_cache.get(model_id): 
      model_id = self.model_id_cache[model_id]

    try:
      res = self.api.get_model(model_id)
      model = self._to_obj(res['model'])
    except ApiError as e:
      model_name = model_id
      if e.response.status_code == 404:
        res = self.search(model_name, model_type)
        if res is None or len(res) > 1:
          raise e
        else:
          model = res[0]
          model_id = model.model_id
          self.model_id_cache.update({model_name:model_id})

    return model

  def search(self, model_name, model_type='concept'):
    ''' search model by name and type

        search the model by name, default is to search concept model
        only. All the custom model trained are concept model.

        Args:
          model_name: name of the model. name is not unique.
          model_type: default to 'concept'

        Returns:
          a list of Model objects or None

        Examples:
          >>> # search for general-v1.3 concept model
          >>> app.models.search('general-v1.3')
          >>>
          >>> # search for color model
          >>> app.models.search('color', model_type='color')
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


class Inputs(object):

  def __init__(self, api):
    self.api = api

  def create_image(self, image):
    ''' create an image from Image object

    Args:
      image: a Clarifai Image object

    Returns:
      the image object just got created and uploaded

    Examples::
      >>> app.inputs.create_image(Image(url='https://samples.clarifai.com/metro-north.jpg'))
    '''

    ret = self.api.add_inputs([image])

    img = self._to_obj(ret['inputs'][0])
    return img

  def create_image_from_url(self, url, image_id=None, concepts=None, not_concepts=None, crop=None, allow_duplicate_url=False):
    ''' create an image from Image url

    Args:
      url: image url
      image_id: ID of the image
      concepts: a list of concepts
      not_concepts: a list of concepts
      crop: crop information, with four corner coordinates
      allow_duplicate_url: True of False, the flag to allow duplicate url to be imported

    Returns:
      the image object just got created and uploaded

    Examples::
      >>> app.inputs.create_image_url(url='https://samples.clarifai.com/metro-north.jpg')
    '''

    image = Image(url=url, image_id=image_id, concepts=concepts, not_concepts=not_concepts, crop=crop, allow_dup_url=allow_duplicate_url)

    return self.create_image(image)

  def create_image_from_filename(self, filename, image_id=None, concepts=None, not_concepts=None, crop=None, allow_duplicate_url=False):
    ''' create an image by local filename

    Args:
      filename: local filename
      image_id: ID of the image
      concepts: a list of concepts
      not_concepts: a list of concepts
      crop: crop information, with four corner coordinates

    Returns:
      the image object just got created and uploaded

    Examples::
      >>> app.inputs.create_image_filename(filename="a.jpeg")
    '''

    fileio = open(filename, 'rb')
    image = Image(file_obj=fileio, image_id=image_id, concepts=concepts, not_concepts=not_concepts, crop=crop, allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_bytes(self, img_bytes, image_id=None, concepts=None, not_concepts=None, crop=None, allow_duplicate_url=False):
    ''' create an image by image bytes

    Args:
      img_bytes: raw bytes of an image
      image_id: ID of the image
      concepts: a list of concepts
      not_concepts: a list of concepts
      crop: crop information, with four corner coordinates

    Returns:
      the image object just got created and uploaded

    Examples::
      >>> app.inputs.create_image_bytes(img_bytes="raw image bytes...")
    '''

    fileio = BytesIO(img_bytes)
    image = Image(file_obj=fileio, image_id=image_id, concepts=concepts, not_concepts=not_concepts, crop=crop, allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_base64(self, base64_bytes, image_id=None, concepts=None, not_concepts=None, crop=None, allow_duplicate_url=False):
    ''' create an image by base64 bytes

    Args:
      base64_bytes: base64 encoded image bytes
      image_id: ID of the image
      concepts: a list of concepts
      not_concepts: a list of concepts
      crop: crop information, with four corner coordinates

    Returns:
      the image object just got created and uploaded

    Examples::
      >>> app.inputs.create_image_bytes(base64_bytes="base64 encoded image bytes...")
    '''

    image = Image(base64=base64_bytes, image_id=image_id, concepts=concepts, not_concepts=not_concepts, crop=crop, allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def bulk_create_images(self, images):
    ''' bulk create images

    Args:
      images: a list of Image object

    Returns:
      a list of Image object just got created

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

  def get_all(self):
    ''' get all inputs in a generator

    Args:
      Void

    Returns:
      a generator that yields Input object

    Examples:
      >>> for image in app.inputs.get_all():
      >>>   print image.input_id
    '''

    page = 1
    per_page = 20

    while True:
      res = self.api.get_inputs(page, per_page)

      if not res['inputs']:
        break

      for one in res['inputs']:
        yield self._to_obj(one)

      page += 1

  def get_by_page(self, page=1, per_page=20):
    ''' get input with pagination

    Args:
      page: page number
      per_page: number of inputs to retrieve per page

    Returns:
      a list of Input object

    Examples:
      >>> for image in app.inputs.get_by_page(2, 10): 
      >>>   print image.input_id
    '''

    res = self.api.get_inputs(page, per_page)
    results = [self._to_obj(one) for one in res['inputs']]

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

  def search(self, qb, page=1, per_page=20):
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

    Returns:
      a list of Input/Image object
    '''

    res = self.api.search_inputs(qb.dict(), page, per_page)
    hits = [self._to_obj(one['input']) for one in res['hits']]
    return hits

  def search_by_image(self, image_id=None, \
                            image=None, url=None, \
                            imgbytes=None, base64bytes=None, \
                            fileobj=None, filename=None, \
                            crop=None, page=1, per_page=20):
    ''' search for visually similar images

    By passing image_id, raw image bytes, base64 encoded bytes, image file io stream,
    image filename, or Clarifai Image object, you can use the visual search power of
    the Clarifai API.

    Also you can specify crop of the image to search over

    Args:
      image_id: unique ID of the image for search
      image: Image object for search
      imgbytes: raw image bytes for search
      base64bytes: base63 encoded image bytes
      fileobj: file io stream, like open(file)
      filename: filename on local filesystem
      crop: crop of the image
      page: page number
      per_page: number of images returned per page

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

    not_nones = filter(lambda x: x, [image_id, image, url, imgbytes, base64bytes, fileobj, filename])
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
        term = OutputSearchTerm(base64=image.base64)
      elif image.file_obj:
        imgbytes = image.file_obj.read()
        base64_bytes = base64.b64encode(imgbytes)
        term = OutputSearchTerm(base64=base64_bytes)

      qb.add_term(term)

      res = self.search(qb, page, per_page)
    elif url:
      img = Image(url=url, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page)
    elif fileobj:
      img = Image(file_obj=fileobj, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page)
    elif imgbytes:
      fileio = BytesIO(imgbytes)
      img = Image(file_obj=fileio, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page)
    elif filename:
      fileio = open(filename, 'rb')
      img = Image(file_obj=fileio, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page)
    elif base64:
      img = Image(base64=base64bytes, crop=crop)
      res = self.search_by_image(image=img, page=page, per_page=per_page)

    return res

  def search_by_metadata(self, meta, page=1, per_page=20):
    ''' search by other meta data of the image rather than concept
        currently only the url string match is supported

    Args:
      meta: is a dictionary for meta data search, with key the search key and the value for string match
      page: page number
      per_page: the number of images to return per page

    Returns:
      a list of Image object

    Examples:
      >>> app.inputs.search_by_metadata(meta={'url':'bla'})
    '''

    if meta.get('url'):
      qb = SearchQueryBuilder()

      term = InputSearchTerm(url=meta['url'])
      qb.add_term(term)
      res = self.search(qb, page, per_page)
    else:
      raise UserError('Metadata query type not supported. Please double check.')

    return res

  def search_by_annotated_concepts(self, concept=None, concepts=None, \
                                   value=True, values=None, \
                                   concept_id=None, concept_ids=None, \
                                   page=1, per_page=20):
    ''' search over the user annotated concepts

    Args:
      concept: concept name to search
      concepts: a list of concept name to search
      concept_id: concept id to search
      concept_ids: a list of concept id to search
      value: whether the concept should exist or NOT
      values: the list of values corresponding to the concepts
      page: page number
      per_page: number of images to return per page

    Returns:
      a list of Image object

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

    return self.search(qb, page, per_page)

  def search_by_predicted_concepts(self, concept=None, concepts=None, \
                                         value=True, values=None,\
                                         concept_id=None, concept_ids=None, \
                                         page=1, per_page=20):
    ''' search over the predicted concepts

    Args:
      concept: concept name to search
      concepts: a list of concept name to search
      concept_id: concept id to search
      concept_ids: a list of concept id to search
      value: whether the concept should exist or NOT
      values: the list of values corresponding to the concepts
      page: page number
      per_page: number of images to return per page

    Returns:
      a list of Image object

    Examples:
      >>> app.inputs.search_by_predicted_concepts(concept='cat')
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

      qb = SearchQueryBuilder()

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

    return self.search(qb, page, per_page)

  def delete_concepts(self, input_id, concepts):
    ''' delete concepts from an input/image

    Args:
      input_id: unique ID of the input
      concepts: a list of concept name

    Returns:
      an Image object
    '''

    ret = self.api.update_input(input_id, 'delete_concepts', concepts)
    one = ret['input']
    return self._to_obj(one)

  def bulk_merge_concepts(self, input_ids, concept_lists):
    ''' bulk merge concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      label_lists: a list of concept list

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_merge_concepts('id', [[('cat',True), ('dog',False)]])
    '''

    ret = self.api.update_inputs('merge_concepts', input_ids, label_lists)
    return self._to_obj(ret['input'])

  def bulk_delete_concepts(self, input_ids, label_lists):
    ''' bulk delete concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      label_lists: a list of concept list

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_delete_concepts(['id'], [['cat', 'dog']])
    '''

    ret = self.api.update_inputs('delete_concepts', input_ids, label_lists)
    return self._to_obj(ret['input'])

  def merge_concepts(self, input_id, concepts, not_concepts):
    ''' merge concepts for one input

    Args:
      input_id: the unique ID of the input
      concepts: the list of concepts
      not_concepts: the list of negative concepts

    Returns:
      an Input object

    Examples:
      >>> app.inputs.merge_concepts('id', ['cat', 'kitty'], ['dog'])
    '''

    ret = self.api.update_input(input_id, 'merge_concepts', concepts, not_concepts)
    one = ret['input']
    return self._to_obj(one)

  def add_concepts(self, input_id, concepts, not_concepts):
    ''' add concepts for one input

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

    input_id = one['id']
    if one['data'].get('image'):
      if one['data']['image'].get('url'):
        one_input = Image(image_id=input_id, url=one['data']['image']['url'], concepts=concepts, not_concepts=not_concepts)
      elif one['data']['image'].get('base64'):
        one_input = Image(image_id=input_id, base64=one['data']['image']['base64'], concepts=concepts, not_concepts=not_concepts)
    elif one['data'].get('video'):
      raise UserError('Not supported yet')
    else:
      raise UserError('Unknown input type')

    return one_input


class Concepts(object):

  def __init__(self, api):
    self.api = api

  def get_all(self):
    ''' get all concepts in a generator

    Args:
      void

    Returns:
      all concepts in a generator
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

  def get(self, concept_id):
    ''' get a concept by id

    Args:
      concept_id: concept ID, the unique identifier of the concept

    Returns:
      If found, return the Concept object
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

  def search(self, term):
    ''' search concepts by concept name with wildcards

    Args:
      term: search term with wildcards

    Returns:
      a list concept in a generator

    Examples:
      >>> app.concepts.search('cat')
    '''

    page = 1
    per_page = 20

    while True:
      res = self.api.search_concepts(term, page, per_page)

      if not res['concepts']:
        break

      for one in res['concepts']:
        yield self._to_obj(one)

      page += 1

  def create(self, concept_id, concept_name=None):
    ''' create a new concept

    Args:
      concept_id: concept ID, the unique identifier of the concept
      concept_name: name of the concept
                    If name is not specified, it will be set to the same as concept ID

    Returns:
      the new Concept object
    '''

    res = self.api.add_concepts([concept_id], [concept_name])
    concept = self._to_obj(res['concepts'][0])
    return concept

  def bulk_create(self, concept_ids, concept_names=None):
    ''' bulk create concepts

        When the concept name is not set, it will be set as the same as concept ID.

    Args:
      concept_ids: a list of concept IDs
      concept_names: a list of concept name

    Returns:
      A list of Concept() object

    Examples::
      >>> bulk_create(['id1', 'id2'], ['cute cat', 'cute dog'])
    '''

    res = self.api.add_concepts(concept_ids, concept_names)
    concepts = [self._to_obj(one) for one in res['concepts']]
    return concepts

  def _to_obj(self, item):

    concept_id = item['id']
    concept_name = item['name']
    app_id = item['app_id']
    created_at = item['created_at']
    updated_at = item['updated_at']

    return Concept(concept_name, concept_id, app_id, \
                   created_at, updated_at)


class Model(object):

  def __init__(self, api, item=None):
    self.api = api

    if item:
      self.model_id = item['id']
      self.model_name = item['name']
      self.created_at = item['created_at']
      self.app_id = item['app_id']
      self.model_version = item['model_version']['id']

      self.output_info = item.get('output_info', {})

      if self.output_info.get('output_config'):
        self.concepts_mutually_exclusive = self.output_info['output_config']['concepts_mutually_exclusive']
        self.closed_environment = self.output_info['output_config']['closed_environment']
      else:
        self.concepts_mutually_exclusive = False
        self.closed_environment = False

  def get_info(self, verbose=False):
    ''' get model info '''

    if verbose is False:
      ret = self.api.get_model(self.model_id)
    else:
      ret = self.api.get_model_output_info(self.model_id)

    return ret

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

    if self.concept_ids:
      data['model']['output_info']['data'] = { "concepts": [{"id": concept_id} for concept_id in self.concept_ids] }

    return data

  def train(self, sync=True):
    ''' train a model

    train the model in synchronous or asynchronous mode

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
    while model_status_code == 21103 or model_status_code == 21101:
      time.sleep(2)
      res_ver = self.api.get_model_version(model_id, model_version)
      model_status_code = res_ver['model_version']['status']['code']

    res['model']['model_version'] = res_ver['model_version']

    model = self._to_obj(res['model'])
    return model

  def predict_by_url(self, url):
    ''' predict a model with url

    Args:
      url: url of an image

    Returns:
      the prediction of the model in JSON format
    '''

    image = Image(url=url)
    res = self.predict([image])
    return res

  def predict_by_filename(self, filename):
    ''' predict a model with a local filename

    Args:
      filename: filename on local filesystem

    Returns:
      the prediction of the model in JSON format
    '''

    fileio = open(filename, 'rb')
    image = Image(file_obj=fileio)
    res = self.predict([image])
    return res

  def predict_by_bytes(self, raw_bytes):
    ''' predict a model with image raw bytes

    Args:
      raw_bytes: raw bytes of an image

    Returns:
      the prediction of the model in JSON format
    '''

    base64_bytes = base64.b64encode(raw_bytes)
    image = Image(base64=base64_bytes)
    res = self.predict([image])
    return res

  def predict_by_base64(self, base64_bytes):
    ''' predict a model with base64 encoded image bytes

    Args:
      base64_bytes: base64 encoded image bytes

    Returns:
      the prediction of the model in JSON format
    '''

    image = Image(base64=base64_bytes)
    res = self.predict([image])
    return res

  def predict(self, inputs):
    ''' predict with multiple images

    Args:
      inputs: a list of Image object 

    Returns:
      the prediction of the model in JSON format 
    '''

    res = self.api.predict_model(self.model_id, inputs, self.model_version)
    return res

  def merge_concepts(self, concept_ids):
    ''' merge concepts in a model

    If the concept does not exist in the model, it will be appended,
    otherwise, the original one will be kept

    Args:
      concept_ids: a list of concept id

    Returns:
      the Model object
    '''

    res = self.api.update_model(self.model_id, 'merge_concepts', concept_ids)
    return self._to_obj(res['model'])

  def add_concepts(self, concept_ids):
    ''' merge concepts in a model

    This is just an alias of `merge_concepts`, for easier understanding of adding new concepts
    to the model without overwritting them

    Args:
      concept_ids: a list of concept id

    Returns:
      the Model object
    '''

    return self.merge_concepts(concept_ids)

  def delete_concepts(self, concept_ids):
    ''' delete concepts from a model

    Args:
      concept_ids: a list of concept id

    Returns:
      the Model object
    '''
    res = self.api.update_model(self.model_id, 'delete_concepts', concept_ids)
    return self._to_obj(res['model'])

  def list_versions(self):
    res = self.api.get_model_versions(self.model_id)
    return res

  def get_version(self, version_id):

    res = self.api.get_model_version(self.model_id, version_id)
    return res

  def delete_version(self, version_id):
    ''' delete model version by version_id '''

    res = self.api.delete_model_version(self.model_id, version_id)
    return res

  def create_version(self):

    res = self.api.create_model_version(self.model_id)
    return res

  def get_inputs(self, version_id=None, page=1, per_page=20):
    ''' get all the inputs from the model or a specific model version
        Without specifying model version id, this will yield the inputs

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

  def _to_obj(self, item):
    ''' convert a model json object to Model object '''
    return Model(self.api, item)


class Concept(object):

  """ Clarifai Concept
  """

  def __init__(self, concept_name, concept_id=None, app_id=None, created_at=None, updated_at=None):
    self.concept_name = concept_name
    self.concept_id = concept_id
    self.app_id = app_id
    self.created_at = created_at
    self.updated_at = updated_at


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

  update_model_action_options = ['merge_concepts', 'delete_concepts']

  def __init__(self, app_id=None, app_secret=None, base_url=None, quiet=True):

    CONF_FILE=os.path.join(os.environ['HOME'], '.clarifai', 'config')

    if app_id is None:
      if os.environ.get('CLARIFAI_APP_ID') and os.environ.get('CLARIFAI_APP_SECRET'):
        logger.debug("Using env variables for id and secret")
        app_id = os.environ['CLARIFAI_APP_ID']
        app_secret = os.environ['CLARIFAI_APP_SECRET']
      elif os.path.exists(CONF_FILE):
        parser = ConfigParser()
        parser.optionxform = str

        with open(CONF_FILE, 'r') as fdr:
          parser.readfp(fdr)

        if parser.has_option('clarifai', 'CLARIFAI_APP_ID') and parser.has_option('clarifai', 'CLARIFAI_APP_SECRET'):
          app_id = parser.get('clarifai', 'CLARIFAI_APP_ID')
          app_secret = parser.get('clarifai', 'CLARIFAI_APP_SECRET')
        else:
          app_id = api_secret = ''
      else:
        app_id = api_secret = ''

    if base_url is None:
      if os.environ.get('CLARIFAI_API_BASE'):
        base_url = os.environ.get('CLARIFAI_API_BASE')
      elif os.path.exists(CONF_FILE):
        parser = ConfigParser()
        parser.optionxform = str

        with open(CONF_FILE, 'r') as fdr:
          parser.readfp(fdr)

        if parser.has_option('clarifai', 'CLARIFAI_API_BASE'):
          base_url = parser.get('clarifai', 'CLARIFAI_API_BASE')
        else:
          base_url = 'api.clarifai.com'
      else:
        base_url = 'api.clarifai.com'

    self.app_id = app_id
    self.app_secret = app_secret

    if quiet:
      logger.setLevel(logging.INFO)
    else:
      logger.setLevel(logging.DEBUG)

    parsed = urlparse(base_url)
    scheme = 'https' if parsed.scheme == '' else parsed.scheme
    base_url = parsed.path if not parsed.netloc else parsed.netloc
    self.base_url = base_url
    self.scheme = scheme
    self.basev2 = urljoin(scheme + '://', base_url)
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

  def _requester(self, resource, params, method, version="v2", files=None):
    ''' Obtains info and verifies user via Token Decorator

    Args:
      resource:
      params: parameters passed to the request
      version: v1 or v2
      method: GET or POST

    Returns:
      JSON from user request
    '''
    self._check_token()
    url = urljoin(self.basev2, version, resource)
    # If 200 return, otherwise see if we need another token then retry.
    status_code = 199
    attempts = 1
    headers = {}
    while status_code != 200 and attempts > 0:
      if method == 'GET':
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Authorization': self.headers['Authorization']}
        res = requests.get(url, params=params, headers=headers)
      elif method == "POST":
        if files:
          headers = {'Authorization': self.headers['Authorization'],
                     'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                    }
          # Seek back to the start.
          for f in files.itervalues():
            f.seek(0)
          res = requests.post(url, data=params, headers=headers, files=files)
        else:
          headers = {'Content-Type': 'application/json',
                     'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                     'Authorization': self.headers['Authorization']}
          res = requests.post(url, data=json.dumps(params), headers=headers)
      elif method == "DELETE":
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Authorization': self.headers['Authorization']}
        res = requests.delete(url, data=json.dumps(params), headers=headers)
      elif method == "PATCH":
        headers = {'Content-Type': 'application/json',
                   'X-Clarifai-Client': 'python:%s' % CLIENT_VERSION,
                   'Authorization': self.headers['Authorization']}
        res = requests.patch(url, data=json.dumps(params), headers=headers)
      else:
        raise UserError("Unsupported request type: '%s'" % method)
      logger.debug("\n%s:\n url: %s\n headers: %s\n data: %s",
                   method, url, str(headers), str(params))
      logger.debug("\nRESULT:\n%s", str(res.content))
      try:
        js = res.json()
      except Exception:
        logger.exception("Could not get json from non-200 response.")
        return res
      if isinstance(js, dict) and js.get('status_code', None) == "TOKEN_EXPIRED":
        self.get_token()
      status_code = res.status_code
      attempts -= 1
    if res.status_code != 200:
      raise ApiError(resource, params, method, res)
    return res

  def get(self, resource, params=None, version="v2"):
    ''' Authorized get from Clarifai's API. '''
    return self._requester(resource, params, 'GET', version)

  def post(self, resource, params=None, version="v2"):
    ''' Authorized post to Clarifai's API. '''
    return self._requester(resource, params, 'POST', version)

  def post_form(self, resource, params=None, version="v2"):
    ''' Authorized post to Clarifai's API. '''
    return self._requester(resource, params=None, method='POST', version=version, files=params)

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
    if not isinstance(objs[0], (Image, Video)):
      raise UserError("Not valid type of content to add. Must be Image or Video")

    resource = "inputs"
    data = {"inputs": [obj.dict() for obj in objs]}
    res = self.post(resource, data)
    return res.json()

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
    return res.json()

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
    return res.json()

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
    return res.json()

  def get_inputs_status(self):
    ''' Get counts of inputs in the Application.

    Returns:
      counts of the inputs, including processed, processing, etc. in JSON format.
    '''

    resource = "inputs/status"
    res = self.get(resource)
    return res.json()

  def delete_input(self, input_id):
    ''' Delete a single input by its id.

    Args:
      input_id: the id of the input

    Returns:
      status of the deletion, in JSON format.
    '''

    if not input_id:
      raise UserError('cannot delete with empty input_id. use delete_all_inputs if you want to delete all')

    resource = "inputs/%s" % input_id
    res = self.delete(resource)
    return res.json()

  def delete_inputs(self, input_ids):
    ''' bulk delete inputs with a list of input IDs

    Args:
      input_ids: the ids of the input, in a list

    Returns:
      status of the bulk deletion, in JSON format.
    '''

    resource = "inputs"
    data = {"inputs": [{"id":input_id} for input_id in input_ids],
            "action":"delete_inputs"
           }

    res = self.patch(resource, data)
    return res.json()

  def delete_all_inputs(self):
    ''' delete all inputs from the application

    Returns:
      status of the deletion, in JSON format.
    '''

    resource = "inputs"
    res = self.delete(resource)
    return res.json()

  def update_input(self, input_id, action="merge_concepts", concepts=None, not_concepts=None):
    ''' update the concepts for a given image

    Args:
      input_id: unique identifier of the input
      action: "merge_concepts" or "delete_concepts"
      concepts: a list of concept names indicating the input is associated with
      not_concepts: a list of concept names indicating the input is not associated with

    Returns:
      status of the deletion, in JSON format.
    '''

    # query by AND and NOT terms
    if concepts is not None:
      pos_terms = [(term, True) for term in concepts]
    else:
      pos_terms = []

    if not_concepts is not None:
      neg_terms = [(term, False) for term in not_concepts]
    else:
      neg_terms = []

    terms = pos_terms + neg_terms

    resource = "inputs/%s/data/concepts" % input_id

    data = {
             'concepts': [{'id': name, 'value': val} for name, val in terms],
             'action': action
           }

    res = self.patch(resource, data)
    return res.json()

  def update_inputs(self, action, input_ids, concept_ids_pairs):
    ''' bulk update inputs, to delete or modify concepts

    Args:
      action: "merge_concepts" or "delete_concepts"
      input_ids: list of input IDs
      concept_ids_pairs: For "merge_concepts", this is a list of (concept_id, value) tuples
                           where value is either True or False
                         For "delete_concepts", this is a list of concept ids

    Returns:
      the update status, in JSON format

    '''

    resource = "inputs"
    data = {"inputs":[]}

    assert(len(input_ids) == len(concept_ids_pairs))

    if action == 'merge_concepts':
      for idx, input_id in enumerate(input_ids):
        entry = {"id":input_id,
                 "data":{"concepts":[{"id":concept_id, "value":val} for concept_id, val in concept_ids_pairs[idx]]
                        }
                }
        data["inputs"].append(entry)
    elif action == 'delete_concepts':
      for idx, input_id in enumerate(input_ids):
        entry = {"id":input_id,
                 "data":{"concepts":[{"id":concept_id} for concept_id in concept_ids_pairs[idx]]
                        }
                }
        data["inputs"].append(entry)

    res = self.patch(resource, data)
    return res.json()

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
    return res.json()

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
    return res.json()

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
    return res.json()

  def search_concepts(self, term, page=1, per_page=20):
    ''' Search concepts

    Args:
      term: search term with wildcards
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

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

    res = self.post(resource, d)
    return res.json()

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
    return res.json()

  def get_model(self, model_id=None):
    ''' get model basic info by model id

    Args:
      model_id: the unique identifier of the model

    Returns:
      the model info in JSON format
    '''

    resource = "models/%s" % model_id

    res = self.get(resource)
    return res.json()

  def get_model_output_info(self, model_id=None):
    ''' get model output info by model id

    Args:
      model_id: the unique identifier of the model

    Returns:
      the model info with output_info in JSON format
    '''

    resource = "models/%s/output_info" % model_id

    res = self.get(resource)
    return res.json()

  def get_model_versions(self, model_id, page=1, per_page=20):
    ''' get model vesions

    Args:
      model_id: the unique identifier of the model
      page: page number
      per_page: the number of versions to return per page

    Returns:
      a list of model versions in JSON format
    '''

    resource = "models/%s/versions" % model_id
    params = {'page': page,
              'per_page': per_page
             }

    res = self.get(resource, params)
    return res.json()

  def get_model_version(self, model_id, version_id):
    ''' get model info for a specific model version

    Args:
      model_id: the unique identifier of a model
      version_id: the model version id
    '''

    resource = "models/%s/versions/%s" % (model_id, version_id)

    res = self.get(resource)
    return res.json()

  def delete_model_version(self, model_id, model_version):
    ''' delete a model version '''

    resource = "models/%s/versions/%s" % (model_id, model_version)
    res = self.delete(resource)
    return res.json()

  def delete_model(self, model_id):
    ''' delete a model '''

    resource = "models/%s" % model_id
    res = self.delete(resource)
    return res.json()

  def delete_all_models(self):
    ''' delete all models '''

    resource = "models"
    res = self.delete(resource)
    return res.json()

  def get_model_inputs(self, model_id, version_id=None, page=1, per_page=20):
    ''' get inputs for the latest model or a specific model version '''

    if not version_id:
      resource = "models/%s/inputs?page=%d&per_page=%d" % \
                 (model_id, page, per_page)
    else:
      resource = "models/%s/version/%s/inputs?page=%d&per_page=%d" % \
                 (model_id, version_id, page, per_page)

    res = self.get(resource)
    return res.json()

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
    return res.json()

  def create_model(self, model_id, model_name=None, concepts=None, \
                   concepts_mutually_exclusive=False, \
                   closed_environment=False):
    ''' create custom model '''

    if not model_name:
      model_name = model_id

    resource = "models"

    data = {
             "model": {
               "name": model_name,
               "output_info": {
                 "output_config": {
                   "concepts_mutually_exclusive": concepts_mutually_exclusive,
                   "closed_environment": closed_environment
                 }
               }
             }
           }

    if model_id:
      data['model']['id'] = model_id

    if concepts:
      data['model']['output_info']['data'] = { "concepts": [{"id": concept} for concept in concepts] }

    res = self.post(resource, data)
    return res.json()

  def update_model(self, model_id, action, concept_ids):

    if not model_id:
      raise UserError('model_id could not be empty')

    if action not in self.update_model_action_options:
      raise UserError('action not allowed')

    resource = "models/%s/output_info/data/concepts" % model_id
    data = {
             "concepts": [{"id": concept_id} for concept_id in concept_ids],
             "action": action
           }

    res = self.patch(resource, data)
    return res.json()

  def create_model_version(self, model_id):
    ''' train for a model '''

    resource = "models/%s/versions" % model_id

    res = self.post(resource)

    return res.json()

  def predict_model(self, model_id, objs, version_id=None):

    if version_id is None:
      resource = "models/%s/outputs" % model_id
    else:
      resource = "models/%s/versions/%s/outputs" % (model_id, version_id)

    if not isinstance(objs, list):
      raise UserError("objs must be a list")
    if not isinstance(objs[0], (Image, Video)):
      raise UserError("Not valid type of content to add. Must be Image or Video")

    data = {"inputs": [obj.dict() for obj in objs]}
    res = self.post(resource, data)
    return res.json()

  def predict_concepts(self, objs):

    models = self.search_models(name='general-v1.3', model_type='concept')
    model = models['models'][0]
    model_id = model['id']

    return self.predict_model(model_id, objs)

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
    msg = "%s %s FAILED. code: %d, reason: %s, response:%s" % (
      method, resource, response.status_code, response.reason,
      str(response.json()))
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

