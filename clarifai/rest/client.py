# -*- coding: utf-8 -*-
"""
Clarifai API Python Client
"""

import base64 as base64_lib
import copy
import logging
import os
import platform
import time
import typing  # noqa
import warnings

from configparser import ConfigParser
from enum import Enum
from io import BytesIO
from posixpath import join as urljoin
from pprint import pformat

import requests
from future.moves.urllib.parse import urlparse
from google.protobuf.struct_pb2 import Struct
from jsonschema import validate
from past.builtins import basestring

from clarifai.errors import ApiClientError, ApiError, TokenError, UserError  # noqa
from clarifai.rest.geo import Geo, GeoBox, GeoLimit, GeoPoint
from clarifai.rest.grpc.grpc_json_channel import GRPCJSONChannel, dict_to_protobuf, protobuf_to_dict
from clarifai.rest.grpc.proto.clarifai.api.concept_pb2 import Concept as ConceptPB
from clarifai.rest.grpc.proto.clarifai.api.concept_pb2 import (
    ConceptQuery, GetConceptRequest, ListConceptsRequest, PatchConceptsRequest,
    PostConceptsRequest, PostConceptsSearchesRequest)
from clarifai.rest.grpc.proto.clarifai.api.data_pb2 import Data as DataPB
from clarifai.rest.grpc.proto.clarifai.api.endpoint_pb2 import _V2
from clarifai.rest.grpc.proto.clarifai.api.endpoint_pb2_grpc import V2Stub
from clarifai.rest.grpc.proto.clarifai.api.input_pb2 import (DeleteInputRequest,
                                                             DeleteInputsRequest,
                                                             GetInputCountRequest, GetInputRequest)
from clarifai.rest.grpc.proto.clarifai.api.input_pb2 import Input as InputPB
from clarifai.rest.grpc.proto.clarifai.api.input_pb2 import (
    ListInputsRequest, ListModelInputsRequest, PatchInputsRequest, PostInputsRequest,
    PostModelFeedbackRequest, PostModelOutputsRequest)
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import (DeleteModelRequest,
                                                             DeleteModelsRequest, GetModelRequest,
                                                             ListModelsRequest)
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import Model as ModelPB
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import ModelQuery
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import OutputConfig as OutputConfigPB
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import OutputInfo as OutputInfoPB
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import (PatchModelsRequest, PostModelsRequest,
                                                             PostModelsSearchesRequest)
from clarifai.rest.grpc.proto.clarifai.api.model_version_pb2 import (
    DeleteModelVersionRequest, GetModelVersionRequest, ListModelVersionsRequest,
    PostModelVersionMetricsRequest, PostModelVersionsRequest)
from clarifai.rest.grpc.proto.clarifai.api.search_pb2 import (PostSearchesRequest,
                                                              PostSearchFeedbackRequest, Query)
from clarifai.rest.grpc.proto.clarifai.api.workflow_pb2 import (
    GetWorkflowRequest, ListPublicWorkflowsRequest, ListWorkflowsRequest,
    PostWorkflowResultsRequest)
from clarifai.rest.grpc.proto.clarifai.utils.pagination.pagination_pb2 import Pagination
from clarifai.rest.solutions.solutions import Solutions
# Versions are imported here to avoid breaking existing client code.
from clarifai.versions import CLIENT_VERSION, OS_VER, PYTHON_VERSION  # noqa

logger = logging.getLogger('clarifai')
logger.handlers = []
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)

logging.getLogger("requests").setLevel(logging.WARNING)

GITHUB_TAG_ENDPOINT = 'https://api.github.com/repos/clarifai/clarifai-python/git/refs/tags'

DEFAULT_TAG_MODEL = 'general-v1.3'

RETRIES = 2  # if connections fail retry a couple times.
CONNECTIONS = 20  # number of connections to maintain in pool.

TOKENS_DEPRECATED_MESSAGE = (
    "App ID/secret are deprecated, please switch to API keys. See here how: "
    "http://help.clarifai.com/api/account-related/all-about-api-keys")


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

  def __init__(
      self,  # type: ClarifaiApp
      app_id=None,  # type: typing.Optional[str]
      app_secret=None,  # type: typing.Optional[str]
      base_url=None,  # type: typing.Optional[str]
      api_key=None,  # type: typing.Optional[str]
      quiet=True,  # type: bool
      log_level=None  # type: typing.Optional[int]
  ):
    # type: (...) -> None

    self.api = ApiClient(
        app_id=app_id,
        app_secret=app_secret,
        base_url=base_url,
        api_key=api_key,
        quiet=quiet,
        log_level=log_level)  # type: ApiClient
    self.solutions = Solutions(api_key)  # type: Solutions

    self.public_models = PublicModels(self.api)  # type: PublicModels

    self.concepts = Concepts(self.api)  # type: Concepts
    self.inputs = Inputs(self.api)  # type: Inputs
    self.models = Models(self.api, self.solutions)  # type: Models
    self.workflows = Workflows(self.api)  # type: Workflows

  """
  Below are the shortcut functions for a more smooth transition of the v1 users
  Also they are convenient functions for the tag only users so they do not have
  to know the extra concepts of Inputs, Models, etc.
  """

  def tag_urls(self, urls, model_name=DEFAULT_TAG_MODEL, model_id=None):
    # type: (typing.Union[typing.List[str], str], str, typing.Optional[str]) -> dict
    warnings.warn('tag_* methods are deprecated. Please switch to using model.predict_* methods.',
                  DeprecationWarning)

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
    # type: (typing.Union[typing.List[str], str], str, typing.Optional[str]) -> dict
    warnings.warn('tag_* methods are deprecated. Please switch to using model.predict_* methods.',
                  DeprecationWarning)

    # validate input
    if not isinstance(files, list) or (len(files) > 1 and not isinstance(files[0], basestring)):
      raise UserError('files must be a list of string file names')

    if len(files) > 128:
      raise UserError('max batch size is 128')

    images = [Image(filename=filename) for filename in files]

    if model_id is not None:
      model = Model(self.api, model_id=model_id)
    else:
      model = self.models.get(model_name)

    res = model.predict(images)
    return res

  def wait_until_inputs_delete_finish(self):  # type: () -> None
    """ Block until a current inputs deletion operation finishes

    The criteria for unblocking is 0 inputs returned from GET /inputs

    Returns:
      None
    """

    inputs = self.inputs.get_by_page()

    while len(inputs) > 0:
      time.sleep(0.2)
      inputs = self.inputs.get_by_page()

  def wait_until_inputs_upload_finish(self, max_wait=666666):  # type: (int) -> None
    """ Block until the inputs upload finishes

    The criteria for unblocking is 0 "to_process" inputs
    from GET /inputs/status

    Returns:
      None
    """
    to_process = 1
    elapsed = 0.0
    time_start = time.time()

    while to_process != 0 and elapsed > max_wait:
      status = self.inputs.check_status()
      to_process = status.to_process
      elapsed = time.time() - time_start
      time.sleep(1)

  def wait_until_models_delete_finish(self):  # type: () -> None
    """ Block until the inputs deletion finishes

    The criteria for unblocking is 0 models returned from GET /models

    Returns:
      None
    """

    private_models = list(self.models.get_all(private_only=True))

    while len(private_models) > 0:
      time.sleep(0.2)
      private_models = list(self.models.get_all(private_only=True))


class Input(object):
  """ The Clarifai Input object
  """

  def __init__(
      self,  # type: Input
      input_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      metadata=None,  # type: typing.Optional[dict]
      geo=None,  # type: typing.Optional[Geo]
      regions=None,  # type: typing.Optional[typing.List[Region]]
      feedback_info=None  # type: typing.Optional[FeedbackInfo]
  ):
    # type: (...) -> None
    """ Construct an Image/Video object. it must have one of url or file_obj set.
    Args:
      input_id: unique id to set for the image. If None then the server will create and return
      one for you.
      concepts: a list of concept names this asset is associated with
      not_concepts: a list of concept names this asset does not associate with
      metadata: metadata as a JSON object to associate arbitrary info with the input
      geo: geographical info for the input, as a Geo() object
      regions: regions of Region object
      feedback_info: FeedbackInfo object
    """

    self.input_id = input_id

    if concepts and not isinstance(concepts, (list, tuple)) and concepts:
      raise UserError('concepts should be a list or tuple')

    if not_concepts and not isinstance(not_concepts, (list, tuple)):
      raise UserError('not_concepts should be a list or tuple')

    if metadata and not isinstance(metadata, dict):
      raise UserError('metadata should be a dictionary')

    # validate geo
    if geo and not isinstance(geo, Geo):
      raise UserError('geo should be a Geo object')

    # validate more
    if (regions and not isinstance(regions, list) and
        not all(isinstance(r, Region) for r in regions)):
      raise UserError('regions should be a list of Region')

    if feedback_info and not isinstance(feedback_info, FeedbackInfo):
      raise UserError('feedback_info should be a FeedbackInfo object')

    self.concepts = concepts
    self.not_concepts = not_concepts
    self.metadata = metadata
    self.geo = geo
    self.feedback_info = feedback_info
    self.regions = regions
    self.score = 0  # type: int
    self.status = None  # type: ApiStatus

  def dict(self):  # type: () -> dict
    """ Return the data of the Input as a dict ready to be input to json.dumps. """
    data = {}
    positive_concepts = [(name, True) for name in (self.concepts or [])]
    negative_concepts = [(name, False) for name in (self.not_concepts or [])]
    concepts = positive_concepts + negative_concepts
    if concepts:
      data['concepts'] = [{'id': name, 'value': value} for name, value in concepts]
    if self.metadata:
      data['metadata'] = self.metadata
    if self.geo:
      data.update(self.geo.dict())
    if self.regions:
      data['regions'] = [r.dict() for r in self.regions]

    input_ = {}
    if self.input_id:
      input_['id'] = self.input_id
    if self.feedback_info:
      input_.update(self.feedback_info.dict())
    if data:
      input_['data'] = data

    return input_


class Image(Input):

  def __init__(
      self,  # type: Image
      url=None,  # type: typing.Optional[str]
      file_obj=None,  # type: typing.Optional[typing.Any]
      base64=None,  # type: typing.Optional[typing.Union[str, bytes]]
      filename=None,  # type: typing.Optional[str]
      crop=None,  # type: typing.Optional[BoundingBox]
      image_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      regions=None,  # type: typing.Optional[typing.List[Region]]
      metadata=None,  # type: typing.Optional[dict]
      geo=None,  # type: typing.Optional[Geo]
      feedback_info=None,  # type: typing.Optional[FeedbackInfo]
      allow_dup_url=False  # type: bool
  ):
    # type: (...) -> None
    """ construct an image

    Args:
      url: the url to a publically accessible image.
      file_obj: a file-like object in which read() will give you the bytes.
      crop: a list of float in the range 0-1.0 in the order [top, left, bottom, right] to crop out
            the asset before use.
      image_id: the image ID
      concepts: the concepts associated with the image
      not_concepts: the concepts not associated with the image
      regions: regions of an image
      metadata: the metadata attached to the image
      geo: geographical information about the image
      feedback_info: feedback information
      allow_dup_url: whether to allow duplicate URLs
    """

    super(Image, self).__init__(
        image_id,
        concepts,
        not_concepts,
        metadata=metadata,
        geo=geo,
        regions=regions,
        feedback_info=feedback_info)

    if crop and (not isinstance(crop, list) or len(crop) != 4):
      raise UserError("crop arg must be list of 4 floats or None")

    self.url = url.strip() if url else url
    self.file_obj = file_obj
    self.filename = filename
    self.base64 = base64
    self.crop = crop
    self.allow_dup_url = allow_dup_url

    we_opened_file = False

    # override the filename with the fileobj as fileobj
    if self.filename is not None:
      if not os.path.exists(self.filename):
        raise UserError("Invalid file path %s. Please check!" % self.filename)
      elif not os.path.isfile(self.filename):
        raise UserError("Not a regular file %s. Please check!" % self.filename)

      self.file_obj = open(self.filename, 'rb')
      self.filename = None

      we_opened_file = True

    if self.file_obj:
      if hasattr(self.file_obj, 'mode') and self.file_obj.mode != 'rb':
        raise UserError(
            ("If you're using open(), then you need to read bytes using the 'rb' mode. "
             "For example: open(filename, 'rb')"))

      # DO NOT put 'read' as first condition
      # as io.BytesIO() has both read() and getvalue() and read() gives you an empty buffer...
      if hasattr(self.file_obj, 'getvalue'):
        self.file_obj.seek(0)
        self.base64 = base64_lib.b64encode(file_obj.getvalue())
      elif hasattr(self.file_obj, 'read'):
        self.file_obj.seek(0)
        self.base64 = base64_lib.b64encode(self.file_obj.read())
      else:
        raise UserError("Not sure how to read your file_obj")

    # Only close the file if we opened it. The users are responsible for closing
    # their open files.
    if we_opened_file:
      self.file_obj.close()

  def dict(self):  # type: () -> dict

    data = super(Image, self).dict()

    image = {}

    if self.base64:
      image['base64'] = self.base64.decode('UTF-8')
    if self.url:
      image['url'] = self.url
    if self.crop:
      image['crop'] = self.crop
    if self.allow_dup_url:
      image['allow_duplicate_url'] = self.allow_dup_url

    if image:
      image_data = {'image': image}
      if 'data' in data:
        data['data'].update(image_data)
      else:
        data['data'] = image_data
    return data


class Video(Input):

  def __init__(
      self,  # type: Video
      url=None,  # type: typing.Optional[str]
      file_obj=None,  # type: typing.Optional[typing.Any]
      base64=None,  # type: typing.Optional[typing.Union[str, bytes]]
      filename=None,  # type: typing.Optional[str]
      video_id=None  # type: typing.Optional[str]
  ):
    # type: (...) -> None
    """
      url: the url to a publicly accessible video.
      file_obj: a file-like object in which read() will give you the bytes.
      base64: base64 encoded string for the video
      filename: a local file name
      video_id: user-defined identifier of this video
    """

    super(Video, self).__init__(input_id=video_id)

    self.url = url.strip() if url else url
    self.file_obj = file_obj
    self.filename = filename
    self.base64 = base64

    we_opened_file = False

    # override the filename with the fileobj as fileobj
    if self.filename is not None:
      if not os.path.exists(self.filename):
        raise UserError("Invalid file path %s. Please check!")
      elif not os.path.isfile(self.filename):
        raise UserError("Not a regular file %s. Please check!")

      self.file_obj = open(self.filename, 'rb')
      self.filename = None

      we_opened_file = True

    if self.file_obj is not None:
      if hasattr(self.file_obj, 'mode') and self.file_obj.mode != 'rb':
        raise UserError(
            ("If you're using open(), then you need to read bytes using the 'rb' mode. "
             "For example: open(filename, 'rb')"))

      # DO NOT put 'read' as first condition
      # as io.BytesIO() has both read() and getvalue() and read() gives you an empty buffer...
      if hasattr(self.file_obj, 'getvalue'):
        self.file_obj.seek(0)
        self.base64 = base64_lib.b64encode(self.file_obj.getvalue())
      elif hasattr(self.file_obj, 'read'):
        self.file_obj.seek(0)
        self.base64 = base64_lib.b64encode(self.file_obj.read())
      else:
        raise UserError("Not sure how to read your file_obj")

    # Only close the file if we opened it. The users are responsible for closing
    # their open files.
    if we_opened_file:
      self.file_obj.close()

  def dict(self):  # type: () -> dict

    data = super(Video, self).dict()

    video = {'video': {}}

    if self.base64 is not None:
      video['video']['base64'] = self.base64.decode('UTF-8')
    else:
      video['video']['url'] = self.url

    if 'data' in data:
      data['data'].update(video)
    else:
      data['data'] = video
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

  def __init__(
      self,  # type: FeedbackInfo
      end_user_id=None,  # type: typing.Optional[str]
      session_id=None,  # type: typing.Optional[str]
      event_type=None,  # type: typing.Optional[str]
      output_id=None,  # type: typing.Optional[str]
      search_id=None  # type: typing.Optional[str]
  ):
    # type: (...) -> None

    self.end_user_id = end_user_id
    self.session_id = session_id
    self.event_type = event_type
    self.output_id = output_id
    self.search_id = search_id

  def dict(self):  # type: () -> dict

    data = {
        "feedback_info": {
            "end_user_id": self.end_user_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
        }
    }

    if self.output_id:
      data['feedback_info']['output_id'] = self.output_id

    if self.search_id:
      data['feedback_info']['search_id'] = self.search_id

    return data


class SearchTerm(object):
  """
  Clarifai search term interface. This is the base class for InputSearchTerm and OutputSearchTerm

  It is used to build SearchQueryBuilder
  """

  def __init__(self):  # type: () -> None
    pass  # if changed, please also change the type hint for this function

  def dict(self):  # type: () -> None
    pass  # if changed, please also change the type hint for this function


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
    >>> InputSearchTerm(geo=Geo(geo_point=GeoPoint(-40, 30),
    >>>                 geo_limit=GeoLimit('withinMiles', 10)))
  """

  def __init__(
      self,  # type: InputSearchTerm
      url=None,  # type: typing.Optional[str]
      input_id=None,  # type: typing.Optional[str]
      concept=None,  # type: typing.Optional[str]
      concept_id=None,  # type: typing.Optional[str]
      value=True,  # type: typing.Optional[typing.Union[bool, float]]
      metadata=None,  # type: typing.Optional[dict]
      geo=None  # type: typing.Optional[Geo]
  ):
    self.url = url
    self.input_id = input_id
    self.concept = concept
    self.concept_id = concept_id
    self.value = value
    self.metadata = metadata
    self.geo = geo

  def dict(self):  # type: () -> dict
    if self.url:
      obj = {"input": {"data": {"image": {"url": self.url}}}}
    elif self.input_id:
      obj = {"input": {"id": self.input_id, "data": {"image": {}}}}
    elif self.concept:
      obj = {"input": {"data": {"concepts": [{"name": self.concept, "value": self.value}]}}}
    elif self.concept_id:
      obj = {"input": {"data": {"concepts": [{"id": self.concept_id, "value": self.value}]}}}
    elif self.metadata:
      obj = {"input": {"data": {"metadata": self.metadata}}}
    elif self.geo:
      obj = {"input": {"data": {}}}
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

  def __init__(
      self,  # type: OutputSearchTerm
      url=None,  # type: typing.Optional[str]
      base64=None,  # type: typing.Optional[typing.Union[str, bytes]]
      input_id=None,  # type: typing.Optional[str]
      concept=None,  # type: typing.Optional[str]
      concept_id=None,  # type: typing.Optional[str]
      value=True,  # type: typing.Optional[typing.Union[bool, float]]
      crop=None  # type: typing.Optional[BoundingBox]
  ):
    self.url = url
    self.base64 = base64
    self.input_id = input_id
    self.concept = concept
    self.concept_id = concept_id
    self.value = value
    self.crop = crop

  def dict(self):  # type: () -> dict
    if self.url:
      obj = {"output": {"input": {"data": {"image": {"url": self.url}}}}}

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    if self.base64:
      obj = {"output": {"input": {"data": {"image": {"base64": self.base64}}}}}

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    elif self.input_id:
      obj = {"output": {"input": {"id": self.input_id, "data": {"image": {}}}}}

      # add crop as needed
      if self.crop:
        obj['output']['input']['data']['image']['crop'] = self.crop

    elif self.concept:
      obj = {"output": {"data": {"concepts": [{"name": self.concept, "value": self.value}]}}}

    elif self.concept_id:
      obj = {"output": {"data": {"concepts": [{"id": self.concept_id, "value": self.value}]}}}

    return obj


class SearchQueryBuilder(object):
  """
  Clarifai Image Search Query Builder

  This builder is for advanced search use ONLY.

  If you are looking for simple concept search, or simple image similarity search,
  you should use one of the existing functions ``search_by_annotated_concepts``,
  ``search_by_predicted_concepts``,
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

  def __init__(self, language=None):  # type: (typing.Optional[str]) -> None
    self.terms = [
    ]  # type: typing.List[typing.Optional[typing.Union[InputSearchTerm, OutputSearchTerm]]]
    self.language = language

  def add_term(self, term):
    # type: (typing.Optional[typing.Union[InputSearchTerm, OutputSearchTerm]]) -> None
    """ add a search term to the query.
        This can search by input or by output.
        Construct the term argument with an InputSearchTerm
        or OutputSearchTerm object.
    """
    if not isinstance(term, InputSearchTerm) and not isinstance(term, OutputSearchTerm):
      raise UserError('first level search term could be only InputSearchTerm, OutputSearchTerm')

    self.terms.append(term)

  def dict(self):  # type: () -> dict
    """ construct the raw query for the RESTful API """

    query = {"ands": [term.dict() for term in self.terms]}

    if self.language is not None:
      query.update({'language': self.language})

    return query


class Workflow(object):
  """ the workflow class
      has the workflow attributes and a list of models associated with it
  """

  api = None  # type: ApiClient

  def __init__(self, api, workflow=None, workflow_id=None):
    # type: (ApiClient, dict, str) -> None

    self.api = api

    if workflow is not None:
      self.wf_id = workflow['id']  # type: str
      if workflow.get('nodes'):
        self.nodes = [WorkflowNode(node) for node in workflow['nodes']]
      else:
        self.nodes = []
    elif workflow_id is not None:
      self.wf_id = workflow_id  # type: str
      self.nodes = []

  def dict(self):  # type: () -> dict
    obj = {
        'id': self.wf_id,
    }

    if self.nodes:
      obj['nodes'] = [node.dict() for node in self.nodes]

    return obj

  def predict_by_url(
      self,  # type: Workflow
      url,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: typing.Optional[bool]
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None  # type: typing.Optional[typing.List[Concept]]
  ):
    # type: (...) -> dict
    """ predict a model with url

    Args:
      url: publicly accessible url of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    """

    url = url.strip()

    if is_video is True:
      input_ = Video(url=url)
    else:
      input_ = Image(url=url)

    output_config = ModelOutputConfig(
        language=lang,
        min_value=min_value,
        max_concepts=max_concepts,
        select_concepts=select_concepts)

    res = self.predict([input_], output_config)
    return res

  def predict_by_filename(
      self,  # type: Workflow
      filename,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: typing.Optional[bool]
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None  # type: typing.Optional[typing.List[Concept]]
  ):
    # type: (...) -> dict
    """ predict a model with a local filename

    Args:
      filename: filename on local filesystem
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    """

    fileio = open(filename, 'rb')

    if is_video is True:
      input_ = Video(file_obj=fileio)
    else:
      input_ = Image(file_obj=fileio)

    output_config = ModelOutputConfig(
        language=lang,
        min_value=min_value,
        max_concepts=max_concepts,
        select_concepts=select_concepts)

    res = self.predict([input_], output_config)
    return res

  def predict_by_bytes(
      self,  # type: Workflow
      raw_bytes,  # type: bytes
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: typing.Optional[bool]
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None  # type: typing.Optional[typing.List[Concept]]
  ):
    # type: (...) -> dict
    """ predict a model with image raw bytes

    Args:
      raw_bytes: raw bytes of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    """

    base64_bytes = base64_lib.b64encode(raw_bytes)

    if is_video is True:
      input_ = Video(base64=base64_bytes)
    else:
      input_ = Image(base64=base64_bytes)

    output_config = ModelOutputConfig(
        language=lang,
        min_value=min_value,
        max_concepts=max_concepts,
        select_concepts=select_concepts)

    res = self.predict([input_], output_config)
    return res

  def predict_by_base64(
      self,  # type: Workflow
      base64_bytes,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: typing.Optional[bool]
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None  # type: typing.Optional[typing.List[Concept]]
  ):
    # type: (...) -> dict
    """ predict a model with base64 encoded image bytes

    Args:
      base64_bytes: base64 encoded image bytes
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed

    Returns:
      the prediction of the model in JSON format
    """

    if is_video is True:
      input_ = Video(base64=base64_bytes)
    else:
      input_ = Image(base64=base64_bytes)

    model_output_config = ModelOutputConfig(
        language=lang,
        min_value=min_value,
        max_concepts=max_concepts,
        select_concepts=select_concepts)

    res = self.predict([input_], model_output_config)
    return res

  def predict(self, inputs, output_config=None):
    # type: (typing.List[typing.Union[Input]], ModelOutputConfig) -> dict
    """ predict with multiple images

    Args:
      inputs: a list of Image objectsg
      output_config: output_config for more prediction parameters

    Returns:
      the prediction of the model in JSON format
    """

    res = self.api.predict_workflow(self.wf_id, inputs, output_config)
    return res


class WorkflowNode(object):
  """ the node in the workflow
  """

  def __init__(self, wf_node):  # type: (dict) -> None
    self.node_id = wf_node['id']  # type: str
    self.model_id = wf_node['model']['id']  # type: str
    self.model_version_id = wf_node['model']['model_version']['id']  # type: str

  def dict(self):  # type: () -> dict
    node = {
        'id': self.node_id,
        'model': {
            'id': self.model_id,
            'model_version': {
                'id': self.model_version_id
            }
        }
    }
    return node


class Workflows(object):

  def __init__(self, api):  # type: (ApiClient) -> None
    self.api = api  # type: ApiClient

  def get_all(self, public_only=False):
    # type: (typing.Optional[bool]) -> typing.Generator[Workflow, None, None]
    """ get all workflows in the application

    Args:
      public_only: whether to get public workflow

    Returns:
      a generator that yields Workflow object

    Examples:
      >>> for workflow in app.workflows.get_all():
      >>>   print(workflow.id)
    """

    res = self.api.get_workflows(public_only)

    # FIXME(robert): hack to correct the empty workflow
    if not res.get('workflows'):
      res['workflows'] = []

    if not res['workflows']:
      return

    for one in res['workflows']:
      workflow = Workflow(self.api, one)
      yield workflow

  def get_by_page(self, public_only=False, page=1, per_page=20):
    # type: (bool, int, int) -> typing.List[Workflow]
    """ get paginated workflows from the application

        When the number of workflows get high, you may want to get
        the paginated results from all the models

    Args:
      public_only: whether to get public workflow
      page: page number
      per_page: number of models returned in one page

    Returns:
      a list of Workflow objects

    Examples:
      >>> workflows = app.workflows.get_by_page(2, 20)
    """

    res = self.api.get_workflows(public_only)
    results = [Workflow(self.api, one) for one in res['workflows']]

    return results

  def get(self, workflow_id):  # type: (str) -> Workflow
    """ get workflow by id

    Args:
      workflow_id: ID of the workflow

    Returns:
      A Workflow object or None

    Examples:
      >>> workflow = app.workflows.get('General')
    """

    res = self.api.get_workflow(workflow_id)
    workflow = Workflow(self.api, res['workflow'])
    return workflow


class Models(object):

  def __init__(self, api, solutions):  # type: (ApiClient, Solutions) -> None
    self.api = api  # type: ApiClient
    self.solutions = solutions  # type: Solutions

    # the cache of the model name -> model id mapping
    # to avoid an extra model query on every prediction by model name
    self.model_id_cache = self.init_model_cache()

  def init_model_cache(self):
    # type: () -> typing.Dict[typing.Tuple[typing.Optional[str], typing.Optional[str]], str]
    """ Initialize the model cache for the public models

        This will go through all public models and cache them

        Returns:
          JSON object containing the name, type, and id of all cached models
    """

    model_cache = {}

    # this is a generator, will NOT raise Exception
    models = self.get_all(public_only=True)

    try:
      for m in models:
        model_name = m.model_name
        model_type = m.output_info['type']
        model_id = m.model_id
        model_cache.update({(model_name, model_type): model_id})

        # for general-v1.3 concept model, make an extra cache entry
        if model_name == 'general-v1.3' and model_type == 'concept':
          model_cache.update({(model_name, None): model_id})
    except ApiError as e:
      if e.error_code == 11007:
        logger.debug("not authorized to call GET /models. Unable to cache models")
      else:
        raise e

    return model_cache

  def clear_model_cache(self):  # type: () -> None
    """ clear model_name -> model_id cache

        WARNING: This is an internal function, user should not call this

        We cache model_name to model_id mapping for API efficiency.
        The first time you call a models.get() by name, the name to ID
        mapping is saved so next time there is no query. Then user does not
        have to query the model ID every time when they want to work on it.
    """

    self.model_id_cache = {}

  def create(
      self,  # type: Models
      model_id,  # type: str
      model_name=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      concepts_mutually_exclusive=False,  # type: bool
      closed_environment=False,  # type: bool
      hyper_parameters=None  # type: typing.Optional[dict]
  ):
    # type (...) -> Model
    """ Create a new model

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
    """
    if not model_name:
      model_name = model_id

    res = self.api.create_model(model_id, model_name, concepts, concepts_mutually_exclusive,
                                closed_environment, hyper_parameters)

    if res.get('model'):
      model = self._to_obj(res['model'])
    elif res.get('status'):
      status = res['status']
      raise UserError('code: %d, desc: %s, details: %s' % (status['code'], status['description'],
                                                           status['details']))
    else:
      raise NotImplementedError('The response returned no model and no status, unable to handle'
                                'such response in the client')

    return model

  def _is_public(self, model):  # type: (Model) -> bool
    """ use app_id to determine whether it is a public model

        For public model, the app_id is either '' or 'main'
        For private model, the app_id is not empty but not 'main'
    """
    return model.app_id == '' or model.app_id == 'main'

  def get_all(self, public_only=False, private_only=False):
    # type: (bool, bool) -> typing.Generator[Model, None, None]
    """ Get all models in the application

    Args:
      public_only: only yield public models
      private_only: only yield private models that tie to your own account

    Returns:
      a generator function that yields Model objects

    Examples:
      >>> for model in app.models.get_all():
      >>>     print(model.model_name)
    """

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
    # type: (bool, bool, int, int) -> typing.List[Model]
    """ get paginated models from the application

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
    """

    res = self.api.get_models(page, per_page)
    results = [self._to_obj(one) for one in res['models']]

    if public_only:
      results = filter(lambda m: self._is_public(m), results)
    elif private_only:
      results = filter(lambda m: not self._is_public(m), results)

    return results

  def delete(self, model_id, version_id=None):  # type: (str, typing.Optional[str]) -> dict
    """ delete the model, or a specific version of the model

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
    """

    if not version_id:
      res = self.api.delete_model(model_id)
    else:
      res = self.api.delete_model_version(model_id, version_id)

    return res

  def bulk_delete(self, model_ids):  # type: (typing.List[str]) -> dict
    """ Delete multiple models.

        Args:
          model_ids: a list of unique IDs of the models to delete

        Returns:
          the raw JSON response from the server

        Examples:
          >>> app.models.delete_models(['model_id1', 'model_id2'])
    """

    res = self.api.delete_models(model_ids)
    return res

  def delete_all(self):  # type: () -> dict
    """ Delete all models and the versions associated with each one

        After this operation, you will have no models in the
        application

        Returns:
          the raw JSON response from the server

        Examples:
          >>> app.models.delete_all()
    """

    res = self.api.delete_all_models()
    return res

  def get(
      self,  # type: Models
      model_name=None,  # type: typing.Optional[str]
      model_id=None,  # type: typing.Optional[str]
      model_type=None  # type:typing.Optional[str]
  ):
    # type: (...) -> Model
    """ Get a model, by ID or name

    Args:
      model_name: name of the model
      model_id: unique identifier of the model
      model_type: type of the model

    Returns:
      the Model object

    Examples:
      >>> # get general-v1.3 model
      >>> app.models.get('general-v1.3')
    """

    # if the model ID is specified, just make the Model
    if model_id:
      model = Model(self.api, model_id=model_id, solutions=self.solutions)
      return model

    # search for the model_name together with the model_type
    if self.model_id_cache.get((model_name, model_type)):
      model_id = self.model_id_cache[(model_name, model_type)]
      model = Model(self.api, model_id=model_id, solutions=self.solutions)
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

        # exclude embed and cluster model when it's not explicitly searched for
        if not model_type:
          res = [
              found_model for found_model in res
              if found_model.output_info['type'] not in (u'embed', u'cluster')
          ]
        if len(res) > 1:
          logging.error('A model by the name of %s or a single similarly-named model could not be '
                        'found' % model_name)
          return None

        # TODO(Rok) HIGH: This sets the return value to a dict, but previous return values are
        #                 Model objects.
        model = res[0]
        self.model_id_cache.update({(model_name, model_type): model.model_id})
      else:
        model = None

    return model

  def search(self, model_name, model_type=None):
    # type: (typing.Optional[str], typing.Optional[str]) -> typing.List[Model]
    """
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
          >>> app.models.search('face', model_type='facedetect')
    """

    res = self.api.search_models(model_name, model_type)
    if res.get('models'):
      results = [self._to_obj(one) for one in res['models']]
    else:
      results = None

    return results

  def _to_obj(self, item):  # type: (dict) -> Model
    """ convert a model json object to Model object """
    return Model(self.api, item, solutions=self.solutions)


def _escape(param):  # type: (str) -> str
  return param.replace('/', '%2F')


class Inputs(object):

  def __init__(self, api):  # type: (ApiClient) -> None
    self.api = api  # type: ApiClient

  def create_image(self, image):  # type: (Image) -> Image
    """ create an image from Image object

    Args:
      image: a Clarifai Image object

    Returns:
      the Image object that just got created and uploaded

    Examples:
      >>> app.inputs.create_image(Image(url='https://samples.clarifai.com/metro-north.jpg'))
    """

    ret = self.api.add_inputs([image])

    img = self._to_obj(ret['inputs'][0])
    return img

  def create_image_from_url(
      self,  # type: Inputs
      url,  # type: str
      image_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      crop=None,  # type: typing.Optional[BoundingBox]
      metadata=None,  # type: typing.Optional[dict]
      geo=None,  # type: typing.Optional[Geo]
      allow_duplicate_url=False  # type: bool
  ):
    # type: (...) -> Image
    """ create an image from Image url

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
      >>>                                  geo=Geo(geo_point=GeoPoint(22.22, 44.44))
    """

    url = url.strip() if url else url

    image = Image(
        url=url,
        image_id=image_id,
        concepts=concepts,
        not_concepts=not_concepts,
        crop=crop,
        metadata=metadata,
        geo=geo,
        allow_dup_url=allow_duplicate_url)

    return self.create_image(image)

  def create_image_from_filename(
      self,  # type: Inputs
      filename,  # type: str
      image_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      crop=None,  # type: typing.Optional[BoundingBox]
      metadata=None,  # type: typing.Optional[dict]
      geo=None,  # type: typing.Optional[Geo]
      allow_duplicate_url=False  # type: bool
  ):
    # type: (...) -> Image
    """ create an image by local filename

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
    """

    with open(filename, 'rb') as fileio:
      image = Image(
          file_obj=fileio,
          image_id=image_id,
          concepts=concepts,
          not_concepts=not_concepts,
          crop=crop,
          metadata=metadata,
          geo=geo,
          allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_bytes(
      self,  # type: Inputs
      img_bytes,  # type: bytes
      image_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      crop=None,  # type: typing.Optional[BoundingBox]
      metadata=None,  # type: typing.Optional[str]
      geo=None,  # type: typing.Optional[Geo]
      allow_duplicate_url=False  # type: bool
  ):
    # type: (...) -> Image
    """ create an image by image bytes

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
    """

    fileio = BytesIO(img_bytes)
    image = Image(
        file_obj=fileio,
        image_id=image_id,
        concepts=concepts,
        not_concepts=not_concepts,
        crop=crop,
        metadata=metadata,
        geo=geo,
        allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def create_image_from_base64(
      self,  # type: Inputs
      base64_bytes,  # type: str
      image_id=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      crop=None,  # type: typing.Optional[BoundingBox]
      metadata=None,  # type: typing.Optional[dict]
      geo=None,  # type: typing.Optional[Geo]
      allow_duplicate_url=False  # type: bool
  ):
    # type: (...) -> Image
    """ create an image by base64 bytes

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
    """

    image = Image(
        base64=base64_bytes,
        image_id=image_id,
        concepts=concepts,
        not_concepts=not_concepts,
        crop=crop,
        metadata=metadata,
        geo=geo,
        allow_dup_url=allow_duplicate_url)
    return self.create_image(image)

  def bulk_create_images(self, images):  # type: (typing.List[Image]) -> typing.List[Image]
    """ Create images in bulk

    Args:
      images: a list of Image objects

    Returns:
      a list of the Image objects that were just created

    Examples:
      >>> img1 = Image(url="", concepts=['cat', 'kitty'])
      >>> img2 = Image(url="", concepts=['dog'], not_concepts=['cat'])
      >>> app.inputs.bulk_create_images([img1, img2])
    """

    lens = len(images)
    if lens > 128:
      raise UserError('the maximum number of inputs in a batch is 128')

    res = self.api.add_inputs(images)
    images = [self._to_obj(one) for one in res['inputs']]
    return images

  def check_status(self):  # type: () -> InputCounts
    """ check the input upload status

    Returns:
      InputCounts object

    Examples:
      >>> status = app.inputs.check_status()
      >>> print(status.code)
      >>> print(status.description)
    """

    ret = self.api.get_inputs_status()
    counts = InputCounts(ret)
    return counts

  def get_all(self, ignore_error=False):  # type: (bool) -> typing.Generator[Input, None, None]
    """ Get all inputs


    Args:
      ignore_error: ignore errored inputs. For example some images may fail to be imported
                    due to bad url

    Returns:
      a generator function that yields Input objects

    Examples:
      >>> for image in app.inputs.get_all():
      >>>     print(image.input_id)
    """

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
        input_ = self._to_obj(one)

        if ignore_error is True and input_.status.code != 30000:
          continue

        yield input_

      page += 1

  def get_by_page(self, page=1, per_page=20, ignore_error=False):
    # type: (int, int, bool) -> typing.List[Input]
    """ Get inputs with pagination

    Args:
      page: page number
      per_page: number of inputs to retrieve per page
      ignore_error: ignore errored inputs. For example some images may fail to be imported
                    due to bad url

    Returns:
      a list of Input objects

    Examples:
      >>> for image in app.inputs.get_by_page(2, 10):
      >>>     print(image.input_id)
    """

    try:
      res = self.api.get_inputs(page, per_page)
    except ApiError as e:
      if e.response.status_code == 207 and e.error_code == 10010:
        res = e.response.json()
      else:
        raise e

    results = []
    for one in res['inputs']:
      input_ = self._to_obj(one)

      if ignore_error is True and input_.status.code != 30000:
        continue

      results.append(input_)

    return results

  def delete(self, input_id):  # type: (str) -> ApiStatus
    """ delete an input with input ID

    Args:
      input_id: the unique input ID

    Returns:
      ApiStatus object

    Examples:
      >>> ret = app.inputs.delete('id1')
      >>> print(ret.code)
    """

    if isinstance(input_id, list):
      res = self.api.delete_inputs(input_id)
    else:
      res = self.api.delete_input(input_id)

    return ApiStatus(res['status'])

  def delete_all(self):  # type: () -> ApiStatus
    """ delete all inputs from the application
    """
    res = self.api.delete_all_inputs()
    return ApiStatus(res['status'])

  def get(self, input_id):  # type: (str) -> Image
    """ get an Input object by input ID

    Args:
      input_id: the unique identifier of the input

    Returns:
      an Image/Input object

    Examples:
      >>> image = app.inputs.get('id1')
      >>> print(image.input_id)

    """

    res = self.api.get_input(input_id)
    one = res['input']
    return self._to_obj(one)

  def search(self, qb, page=1, per_page=20, raw=False):
    # type: (SearchQueryBuilder, int, int, bool) -> typing.List[Image]
    """ search with a clarifai image query builder

        WARNING: this is the advanced search function. You will need to build a query builder
        in order to use this.

      There are a few simple search functions:
          search_by_annotated_concepts()
          search_by_predicted_concepts()
          search_by_image()
          search_by_metadata()

    Args:
      qb: clarifai query builder
      page: the results page
      per_page: results per page
      raw: whether to return the original JSON object instead of a list of Image objects

    Returns:
      a list of Input/Image object
    """

    res = self.api.search_inputs(qb.dict(), page, per_page)

    # output raw result when the flag is set
    if raw:
      return res

    hits = [self._to_search_obj(one) for one in res['hits']]
    return hits

  def search_by_image(
      self,  # type: Inputs
      image_id=None,  # type: typing.Optional[str]
      image=None,  # type: typing.Optional[Image]
      url=None,  # type: typing.Optional[str]
      imgbytes=None,  # type: typing.Optional[bytes]
      base64bytes=None,  # type: typing.Optional[str]
      fileobj=None,  # type: typing.Optional[typing.Any]
      filename=None,  # type: typing.Optional[str]
      crop=None,  # type: typing.Optional[BoundingBox]
      page=1,  # type: int
      per_page=20,  # type: int
      raw=False  # type: bool
  ):
    # type: (...) -> typing.List[Image]
    """ Search for visually similar images

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
    """

    not_nones = [
        x for x in [image_id, image, url, imgbytes, base64bytes, fileobj, filename]
        if x is not None
    ]
    if len(not_nones) != 1:
      raise UserError('Unable to construct an image')

    if image_id:
      qb = SearchQueryBuilder()
      term = OutputSearchTerm(input_id=image_id, crop=crop)
      qb.add_term(term)

      return self.search(qb, page, per_page, raw)
    elif image:
      qb = SearchQueryBuilder()

      if image.url:
        term = OutputSearchTerm(url=image.url, crop=crop)
      elif image.base64:
        term = OutputSearchTerm(base64=image.base64.decode('UTF-8'), crop=crop)
      elif image.file_obj:
        if hasattr(image.file_obj, 'getvalue'):
          base64_bytes = base64_lib.b64encode(image.file_obj.getvalue()).decode('UTF-8')
        elif hasattr(image.file_obj, 'read'):
          base64_bytes = base64_lib.b64encode(image.file_obj.read()).decode('UTF-8')
        else:
          raise UserError("Not sure how to read your file_obj")

        term = OutputSearchTerm(base64=base64_bytes, crop=crop)
      else:
        raise UserError('Unrecognized image object')

      qb.add_term(term)

      return self.search(qb, page, per_page, raw)

    if url:
      img = Image(url=url)
    elif fileobj:
      img = Image(file_obj=fileobj)
    elif imgbytes:
      fileio = BytesIO(imgbytes)
      img = Image(file_obj=fileio)
    elif filename:
      fileio = open(filename, 'rb')
      img = Image(file_obj=fileio)
    elif base64bytes:
      img = Image(base64=base64bytes)
    else:
      raise UserError('None of the arguments was passed in')

    return self.search_by_image(image=img, page=page, per_page=per_page, raw=raw, crop=crop)

  def search_by_original_url(self, url, page=1, per_page=20, raw=False):
    # type: (str, int, int, bool) -> typing.List[Image]
    """ search by the original url of the uploaded images

    Args:
      url: url of the image
      page: page number
      per_page: the number of images to return per page
      raw: raw result indicator

    Returns:
      a list of Image objects

    Examples:
      >>> app.inputs.search_by_original_url(url='http://bla')
    """

    qb = SearchQueryBuilder()

    term = InputSearchTerm(url=url)
    qb.add_term(term)
    res = self.search(qb, page, per_page, raw)

    return res

  def search_by_metadata(self, metadata, page=1, per_page=20, raw=False):
    # type: (dict, int, int, bool) -> typing.List[Image]
    """ search by meta data of the image rather than concept

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
    """

    if isinstance(metadata, dict):
      qb = SearchQueryBuilder()

      term = InputSearchTerm(metadata=metadata)
      qb.add_term(term)
      res = self.search(qb, page, per_page, raw)
    else:
      raise UserError('Metadata must be a valid dictionary. Please double check.')

    return res

  def search_by_annotated_concepts(
      self,  # type: Inputs
      concept=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      value=True,  # type: bool
      values=None,  # type: typing.Optional[typing.List[bool]]
      concept_id=None,  # type: typing.Optional[str]
      concept_ids=None,  # type: typing.Optional[typing.List[str]]
      page=1,  # type: int
      per_page=20,  # type: int
      raw=False  # type: bool
  ):
    # type: (...) -> typing.List[Image]
    """ search using the concepts the user has manually specified

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
    """

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

  def search_by_geo(
      self,  # type: Inputs
      geo_point=None,  # type: typing.Optional[GeoPoint]
      geo_limit=None,  # type: typing.Optional[GeoLimit]
      geo_box=None,  # type: typing.Optional[GeoBox]
      page=1,  # type: int
      per_page=20,  # type: int
      raw=False  # type: bool
  ):
    # type: (...) -> typing.List[Image]
    """ search by geo point and geo limit

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
    """
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

  def search_by_predicted_concepts(
      self,  # type: Inputs
      concept=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.Optional[str]]
      value=True,  # type: bool
      values=None,  # type: typing.Optional[typing.List[bool]]
      concept_id=None,  # type: typing.Optional[str]
      concept_ids=None,  # type: typing.Optional[typing.List[str]]
      page=1,  # type: int
      per_page=20,  # type: int
      lang=None,  # type: typing.Optional[str]
      raw=False  # type: bool
  ):
    # type: (...) -> typing.List[Image]
    """ search over the predicted concepts

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
    """
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
    # type: (str, typing.Optional[FeedbackInfo]) -> dict
    """
    Send feedback for search

    Args:
      input_id: unique identifier for the input
      feedback_info: the feedback information

    Returns:
      None
    """

    feedback_input = Image(image_id=input_id, feedback_info=feedback_info)
    res = self.api.send_search_feedback(feedback_input)

    return res

  def update(self, image, action='merge'):  # type: (Image, str) -> Image
    """
    Update the information of an input/image

    Args:
      image: an Image object that has concepts, metadata, etc.
      action: one of ['merge', 'overwrite']

              'merge' is to append the info onto the existing info, for either concept or
              metadata

              'overwrite' is to overwrite the existing metadata and concepts with the
              existing ones

    Returns:
      an Image object

    Examples:
      >>> new_img = Image(image_id="abc", concepts=['c1', 'c2'], not_concepts=['c3'],
      >>>                 metadata={'key':'val'})
      >>> app.inputs.update(new_img, action='overwrite')
    """
    res = self.api.patch_inputs(action=action, inputs=[image])

    one = res['inputs'][0]
    return self._to_obj(one)

  # TODO(Rok) MEDIUM: Unconsistent name. Should be bulk_update_image. Deprecate this method
  #                   and create a new one.
  def bulk_update(self, images, action='merge'):
    # type: (typing.List[typing.Union[Input]], str) -> typing.List[Image]
    """ Update the input
    update the information of an input/image

    Args:
      images: a list of Image objects that have concepts, metadata, etc.
      action: one of ['merge', 'overwrite']

              'merge' is to append the info onto the existing info, for either concept or
              metadata

              'overwrite' is to overwrite the existing metadata and concepts with the
              existing ones

    Returns:
      an Image object

    Examples:
      >>> new_img1 = Image(image_id="abc1", concepts=['c1', 'c2'], not_concepts=['c3'],
      >>>                  metadata={'key':'val'})
      >>> new_img2 = Image(image_id="abc2", concepts=['c1', 'c2'], not_concepts=['c3'],
      >>>                  metadata={'key':'val'})
      >>> app.inputs.update([new_img1, new_img2], action='overwrite')
    """
    ret = self.api.patch_inputs(action=action, inputs=images)
    objs = [self._to_obj(item) for item in ret['inputs']]
    return objs

  def delete_concepts(self, input_id, concepts):  # type: (str, typing.List[str]) -> Image
    """ delete concepts from an input/image

    Args:
      input_id: unique ID of the input
      concepts: a list of concept names

    Returns:
      an Image object
    """

    res = self.update(Image(image_id=input_id, concepts=concepts), action='remove')
    return res

  def bulk_merge_concepts(self, input_ids, concept_lists):
    # type: (typing.List[str], typing.List[typing.List[str]]) -> typing.List[Image]
    """ bulk merge concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      concept_lists: a list of concept lists, each one corresponding to a listed input ID and
      filled with concepts to be added to that input

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_merge_concepts('id', [[('cat',True), ('dog',False)]])
    """

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
    # type: (typing.List[str], typing.List[typing.List[str]]) -> typing.List[Image]
    """ bulk delete concepts from a list of input ids

    Args:
      input_ids: a list of input IDs
      concept_lists: a list of concept lists, each one corresponding to a listed input ID and
      filled with concepts to be deleted from that input

    Returns:
      an Input object

    Examples:
      >>> app.inputs.bulk_delete_concepts(['id'], [['cat', 'dog']])
    """

    # the reason list comprehension is not used is it breaks the 100 chars width
    inputs = []
    for input_id, concepts in zip(input_ids, concept_lists):
      one_input = Image(image_id=input_id, concepts=concepts)
      inputs.append(one_input)

    res = self.bulk_update(inputs, action='remove')
    return res

  def merge_concepts(
      self,  # type: Inputs
      input_id,  # type: str
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type:  typing.Optional[typing.List[str]]
      overwrite=False  # type: bool
  ):
    # type: (...) -> Image
    """ Merge concepts for one input

    Args:
      input_id: the unique ID of the input
      concepts: the list of concepts
      not_concepts: the list of negative concepts
      overwrite: if True, this operation will replace the previous concepts. If False,
      it will append them.


    Returns:
      an Input object

    Examples:
      >>> app.inputs.merge_concepts('id', ['cat', 'kitty'], ['dog'])
    """

    image = Image(image_id=input_id, concepts=concepts, not_concepts=not_concepts)

    if overwrite is True:
      action = 'overwrite'
    else:
      action = 'merge'

    res = self.update(image, action=action)
    return res

  def add_concepts(self, input_id, concepts=None, not_concepts=None):
    # type: (str, typing.Optional[typing.List[str]], typing.Optional[typing.List[str]]) -> Image
    """ Add concepts for one input

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
    """
    return self.merge_concepts(input_id, concepts, not_concepts)

  def merge_metadata(self, input_id, metadata):  # type: (str, dict) -> Image
    """ merge metadata for the image

    This is to merge/update the metadata of the given image

    Args:
      input_id: the unique ID of the input
      metadata: the metadata dictionary

    Examples:
      >>> # merge the metadata
      >>> # metadata will be appended to the existing key/value pairs
      >>> app.inputs.merge_metadata('id', {'key1':'value1', 'key2':'value2'})
    """
    image = Image(image_id=input_id, metadata=metadata)

    action = 'merge'
    res = self.update(image, action=action)
    return res

  def _to_search_obj(self, one):  # type: (dict) -> Image
    """ convert the search candidate to input object """
    score = one['score']
    one_input = self._to_obj(one['input'])
    one_input.score = score
    return one_input

  def _to_obj(self, one):  # type: (dict) -> Image

    # get concepts
    concepts = []
    not_concepts = []
    for concept in one['data'].get('concepts', []):
      if concept.get('value', 1) == 1:
        concepts.append(concept.get('name') or concept['id'])
      else:
        not_concepts.append(concept.get('name') or concept['id'])

    if not concepts:
      concepts = None

    if not not_concepts:
      not_concepts = None

    # get metadata
    metadata = one['data'].get('metadata')

    # get geo
    geo = geo_json = one['data'].get('geo')

    if geo_json:
      geo_schema = {
          'additionalProperties': False,
          'type': 'object',
          'properties': {
              'geo_point': {
                  'type': 'object',
                  'properties': {
                      'longitude': {
                          'type': 'number'
                      },
                      'latitude': {
                          'type': 'number'
                      }
                  }
              }
          }
      }

      validate(geo_json, geo_schema)
      geo = Geo(GeoPoint(geo_json['geo_point']['longitude'], geo_json['geo_point']['latitude']))

    # get regions
    regions = None
    regions_json = one['data'].get('regions')
    if regions_json:
      regions = [
          Region(
              region_id=r['id'],
              region_info=RegionInfo(
                  bbox=BoundingBox(
                      top_row=r['region_info']['bounding_box']['top_row'],
                      left_col=r['region_info']['bounding_box']['left_col'],
                      bottom_row=r['region_info']['bounding_box']['bottom_row'],
                      right_col=r['region_info']['bounding_box']['right_col'])),
              face=Face(FaceIdentity([c for c in r['data']['face']['identity']['concepts']]))
              if r.get('data', {}).get('face') else None) for r in regions_json
      ]

    input_id = one['id']
    if one['data'].get('image'):
      allow_dup_url = one['data']['image'].get('allow_duplicate_url', False)

      if one['data']['image'].get('url'):
        if one['data']['image'].get('crop'):
          crop = one['data']['image']['crop']
          one_input = Image(
              image_id=input_id,
              url=one['data']['image']['url'],
              concepts=concepts,
              not_concepts=not_concepts,
              crop=crop,
              metadata=metadata,
              geo=geo,
              regions=regions,
              allow_dup_url=allow_dup_url)
        else:
          one_input = Image(
              image_id=input_id,
              url=one['data']['image']['url'],
              concepts=concepts,
              not_concepts=not_concepts,
              metadata=metadata,
              geo=geo,
              regions=regions,
              allow_dup_url=allow_dup_url)
      elif one['data']['image'].get('base64'):
        if one['data']['image'].get('crop'):
          crop = one['data']['image']['crop']
          one_input = Image(
              image_id=input_id,
              base64=one['data']['image']['base64'],
              concepts=concepts,
              not_concepts=not_concepts,
              crop=crop,
              metadata=metadata,
              geo=geo,
              regions=regions,
              allow_dup_url=allow_dup_url)
        else:
          one_input = Image(
              image_id=input_id,
              base64=one['data']['image']['base64'],
              concepts=concepts,
              not_concepts=not_concepts,
              metadata=metadata,
              geo=geo,
              regions=regions,
              allow_dup_url=allow_dup_url)
      else:
        raise UserError('Unknown input type')
    elif one['data'].get('video'):
      raise UserError('Not supported yet')
    else:
      raise UserError('Unknown input type')

    if one.get('status'):
      one_input.status = ApiStatus(one['status'])

    return one_input


class Concepts(object):

  def __init__(self, api):  # type: (ApiClient) -> None
    self.api = api  # type: ApiClient

  def get_all(self):  # type: () -> typing.Generator[Concept, None, None]
    """ Get all concepts associated with the application

    Returns:
      all concepts in a generator function
    """

    page = 1
    per_page = 20

    while True:
      res = self.api.get_concepts(page, per_page)

      if not res['concepts']:
        break

      for one in res['concepts']:
        yield self._to_obj(one)

      page += 1

  def get_by_page(self, page=1, per_page=20):  # type: (int, int) -> typing.List[Concept]
    """ get concept with pagination

    Args:
      page: page number
      per_page: number of concepts to retrieve per page

    Returns:
      a list of Concept objects

    Examples:
      >>> for concept in app.concepts.get_by_page(2, 10):
      >>>     print(concept.concept_id)
    """

    res = self.api.get_concepts(page, per_page)
    results = [self._to_obj(one) for one in res.get('concepts', [])]

    return results

  def get(self, concept_id):  # type: (str) -> Concept
    """ Get a concept by id

    Args:
      concept_id: concept ID, the unique identifier of the concept

    Returns:
      If found, return the Concept object.
      Otherwise, return None

    Examples:
      >>> app.concepts.get('id')
    """

    res = self.api.get_concept(concept_id)
    if res.get('concept'):
      concept = self._to_obj(res['concept'])
      return concept
    else:
      return None

  def search(self, term, lang=None):
    # type: (str, typing.Optional[str]) -> typing.Generator[Concept, None, None]
    """ search concepts by concept name with wildcards

    Args:
      term: search term with wildcards allowed
      lang: language to search, if none is specified the default for the application will be
            used

    Returns:
      a generator function with all concepts pertaining to the search term

    Examples:
      >>> app.concepts.search('cat')
      >>> # search for Chinese label name
      >>> app.concepts.search(u'狗*', lang='zh')
    """

    page = 1
    per_page = 20

    while True:
      res = self.api.search_concepts(term, page, per_page, lang)

      if not res.get('concepts'):
        break

      for one in res['concepts']:
        yield self._to_obj(one)

      page += 1

  def update(self, concept_id, concept_name, action='overwrite'):
    # type: (str, str, str) -> Concept
    """ Patch concept

    Args:
      concept_id: id of the concept
      concept_name: the new name for the concept
      action: the action

    Returns:
      the new Concept object

    Examples:
      >>> app.concepts.update(concept_id='myid1', concept_name='new_concept_name2')
    """

    c = Concept(concept_name=concept_name, concept_id=concept_id)
    res = self.api.patch_concepts(action=action, concepts=[c])

    return self._to_obj(res['concepts'][0])

  def bulk_update(self, concept_ids, concept_names, action='overwrite'):
    # type: (typing.List[str], typing.List[str], str) -> typing.List[Concept]
    """ Patch multiple concepts

    Args:
      concept_ids: a list of concept IDs, in sequence
      concept_names: a list of corresponding concept names, in the same sequence
      action: the type of update

    Returns:
      the new Concept object

    Examples:
      >>> app.concepts.bulk_update(concept_ids=['myid1', 'myid2'],
      >>>                          concept_names=['name2', 'name3'])
    """

    concepts = [
        Concept(concept_name=concept_name, concept_id=concept_id)
        for concept_name, concept_id in zip(concept_names, concept_ids)
    ]
    res = self.api.patch_concepts(action=action, concepts=concepts)

    return [self._to_obj(c) for c in res['concepts']]

  def create(self, concept_id, concept_name=None):
    # type: (str, typing.Optional[str]) -> Concept
    """ Create a new concept

    Args:
      concept_id: concept ID, the unique identifier of the concept
      concept_name: name of the concept.
                    If name is not specified, it will be set to the same as the concept ID

    Returns:
      the new Concept object
    """

    res = self.api.add_concepts([concept_id], [concept_name])
    concept = self._to_obj(res['concepts'][0])
    return concept

  def bulk_create(self, concept_ids, concept_names=None):
    # type: (typing.List[str], typing.Optional[typing.List[str]]) -> typing.List[Concept]
    """ Bulk create concepts

    When the concept name is not set, it will be set as the same as concept ID.

    Args:
      concept_ids: a list of concept IDs, in sequence
      concept_names: a list of corresponding concept names, in the same sequence

    Returns:
      A list of Concept objects

    Examples:
      >>> app.concepts.bulk_create(['id1', 'id2'], ['cute cat', 'cute dog'])
    """

    res = self.api.add_concepts(concept_ids, concept_names)
    concepts = [self._to_obj(one) for one in res['concepts']]
    return concepts

  def _to_obj(self, item):  # type: (dict) -> Concept

    concept_id = item['id']
    concept_name = item['name']
    app_id = item['app_id']
    created_at = item['created_at']

    return Concept(
        concept_name=concept_name, concept_id=concept_id, app_id=app_id, created_at=created_at)


class Model(object):

  def __init__(
      self,  # type: Model
      api,  # type: ApiClient
      item=None,  # type: typing.Optional[dict]
      model_id=None,  # type: typing.Optional[str]
      solutions=None  # type: typing.Optional[Solutions]
  ):
    # type: (...) -> None

    self.api = api  # type: ApiClient
    self.solutions = ModelSolutions(solutions, self)  # type: ModelSolutions

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

      output_config = self.output_info.get('output_config', {})
      self.concepts_mutually_exclusive = output_config.get('concepts_mutually_exclusive',
                                                           False)  # type: bool
      self.closed_environment = output_config.get('closed_environment', False)  # type: bool

      self.hyper_parameters = output_config.get('hyper_params')  # type: typing.Optional[dict]

      self.concepts = []  # type: typing.List[Concept]
      if self.output_info.get('data', {}).get('concepts'):
        for concept in self.output_info['data']['concepts']:
          concept = Concept(
              concept_name=concept['name'],
              concept_id=concept['id'],
              app_id=concept['app_id'],
              created_at=concept['created_at'])
          self.concepts.append(concept)

  def get_info(self, verbose=False):  # type: (bool) -> dict
    """ get model info, with or without the concepts associated with the model.

    Args:
      verbose: default is False. True will yield output_info, with concepts of the model

    Returns:
      raw json of the response

    Examples:
      >>> # with basic model info
      >>> model.get_info()
      >>> # model info with concepts
      >>> model.get_info(verbose=True)
    """

    if not verbose:
      ret = self.api.get_model(self.model_id, self.model_version)
    else:
      ret = self.api.get_model_output_info(self.model_id, self.model_version)

    return ret

  def get_concept_ids(self):  # type: () -> typing.List[Concept]
    """ get concepts IDs associated with the model

    Returns:
      a list of concept IDs

    Examples:
      >>> ids = model.get_concept_ids()
    """

    if self.concepts:
      concepts = [c.dict() for c in self.concepts]
    else:
      res = self.get_info(verbose=True)
      concepts = res['model']['output_info'].get('data', {}).get('concepts', [])

    return [c['id'] for c in concepts]

  def dict(self):  # type: () -> dict

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
      data['model']['output_info']['data'] = {"concepts": ids}

    return data

  def train(self, sync=True, timeout=60):  # type: (bool, int) -> typing.Union[Model, dict]
    """
    train the model in synchronous or asynchronous mode. Synchronous will block until the
    model is trained, async will not.

    Args:
      sync: indicating synchronous or asynchronous, default is True
      timeout: Used when sync=True. Num. of seconds we should wait for training to complete.

    Returns:
      the Model object

    """

    res = self.api.create_model_version(self.model_id)

    status = res['status']
    if status['code'] == 10000:
      model_id = res['model']['id']
      model_version = res['model']['model_version']['id']
      model_status_code = res['model']['model_version']['status']['code']
    else:
      # TODO: This should probably be converted to a Model object.
      return res

    if sync is False:
      # TODO: This should probably be converted to a Model object.
      return res

    # train in sync despite the RESTful api is always async
    # will loop until the model is trained
    # 21103: queued for training
    # 21101: being trained

    time_start = time.time()

    res_ver = None
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

    if res_ver:
      res['model']['model_version'] = res_ver['model_version']

    return self._to_obj(res['model'])

  def predict_by_url(
      self,  # type: Model
      url,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: bool
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None,  # type: typing.Optional[typing.List[Concept]]
      sample_ms=None  # type: typing.Optional[int]
  ):
    # type: (...) -> dict
    """ predict a model with url

    Args:
      url: publicly accessible url of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed
      sample_ms: video frame prediction every sample_ms milliseconds

    Returns:
      the prediction of the model in JSON format
    """

    url = url.strip()

    if is_video is True:
      input_ = Video(url=url)  # type: Input
    else:
      input_ = Image(url=url)

    model_output_info = ModelOutputInfo(
        output_config=ModelOutputConfig(
            language=lang,
            min_value=min_value,
            max_concepts=max_concepts,
            select_concepts=select_concepts,
            sample_ms=sample_ms))

    res = self.predict([input_], model_output_info)
    return res

  def predict_by_filename(
      self,  # type: Model
      filename,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: bool
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None,  # type: typing.Optional[typing.List[Concept]]
      sample_ms=None  # type: typing.Optional[int]
  ):
    # type: (...) -> dict
    """ predict a model with a local filename

    Args:
      filename: filename on local filesystem
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed
      sample_ms: video frame prediction every sample_ms milliseconds

    Returns:
      the prediction of the model in JSON format
    """

    with open(filename, 'rb') as fileio:
      if is_video is True:
        input_ = Video(file_obj=fileio)  # type: Input
      else:
        input_ = Image(file_obj=fileio)

    model_output_info = ModelOutputInfo(
        output_config=ModelOutputConfig(
            language=lang,
            min_value=min_value,
            max_concepts=max_concepts,
            select_concepts=select_concepts,
            sample_ms=sample_ms))

    res = self.predict([input_], model_output_info)
    return res

  def predict_by_bytes(
      self,  # type: Model
      raw_bytes,  # type: bytes
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: bool
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None,  # type: typing.Optional[typing.List[Concept]]
      sample_ms=None  # type: typing.Optional[int]
  ):
    # type: (...) -> dict
    """ predict a model with image raw bytes

    Args:
      raw_bytes: raw bytes of an image
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed
      sample_ms: video frame prediction every sample_ms milliseconds

    Returns:
      the prediction of the model in JSON format
    """

    base64_bytes = base64_lib.b64encode(raw_bytes)

    if is_video is True:
      input_ = Video(base64=base64_bytes)  # type: Input
    else:
      input_ = Image(base64=base64_bytes)

    model_output_info = ModelOutputInfo(
        output_config=ModelOutputConfig(
            language=lang,
            min_value=min_value,
            max_concepts=max_concepts,
            select_concepts=select_concepts,
            sample_ms=sample_ms))

    res = self.predict([input_], model_output_info)
    return res

  def predict_by_base64(
      self,  # type: Model
      base64_bytes,  # type: str
      lang=None,  # type: typing.Optional[str]
      is_video=False,  # type: bool
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None,  # type: typing.Optional[typing.List[Concept]]
      sample_ms=None  # type: typing.Optional[int]
  ):
    # type: (...) -> dict
    """ predict a model with base64 encoded image bytes

    Args:
      base64_bytes: base64 encoded image bytes
      lang: language to predict, if the translation is available
      is_video: whether this is a video
      min_value: threshold to cut the predictions, 0-1.0
      max_concepts: max concepts to keep in the predictions, 0-200
      select_concepts: a list of concepts that are selected to be exposed
      sample_ms: video frame prediction every sample_ms milliseconds

    Returns:
      the prediction of the model in JSON format
    """

    if is_video is True:
      input_ = Video(base64=base64_bytes)  # type: Input
    else:
      input_ = Image(base64=base64_bytes)

    model_output_info = ModelOutputInfo(
        output_config=ModelOutputConfig(
            language=lang,
            min_value=min_value,
            max_concepts=max_concepts,
            select_concepts=select_concepts,
            sample_ms=sample_ms))

    res = self.predict([input_], model_output_info)
    return res

  # TODO(Rok) MEDIUM: Add bulk_predict methods.
  def predict(self, inputs, model_output_info=None):
    # type: (typing.List[Input], typing.Optional[ModelOutputInfo]) -> dict
    """ predict with multiple images

    Args:
      inputs: a list of Image objects
      model_output_info: the model output info

    Returns:
      the prediction of the model in JSON format
    """

    res = self.api.predict_model(self.model_id, inputs, self.model_version, model_output_info)
    return res

  def merge_concepts(self, concept_ids, overwrite=False):
    # type: (typing.List[str], bool) -> Model
    """ merge concepts in a model

    When overwrite is False, if the concept does not exist in the model it will be appended.
    Otherwise, the original one will be kept.

    Args:
      concept_ids: a list of concept id
      overwrite: True or False. If True, the existing concepts will be replaced

    Returns:
      the Model object
    """

    if overwrite is True:
      action = 'overwrite'
    else:
      action = 'merge'

    model = self.update(action=action, concept_ids=concept_ids)
    return model

  def add_concepts(self, concept_ids):  # type: (typing.List[str]) -> Model
    """ merge concepts into a model

    This is just an alias of `merge_concepts`, for easier understanding of adding new concepts
    to the model without overwritting them.

    Args:
      concept_ids: a list of concept IDs

    Returns:
      the Model object

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.add_concepts(['cat', 'dog'])
    """

    return self.merge_concepts(concept_ids)

  def update(
      self,  # type: Model
      action='merge',  # type: str
      model_name=None,  # type: typing.Optional[str]
      concepts_mutually_exclusive=None,  # type: typing.Optional[bool]
      closed_environment=None,  # type: typing.Optional[bool]
      concept_ids=None  # type: typing.Optional[typing.List[str]]
  ):
    # type: (...) -> Model
    """
    Update the model attributes. The name of the model, list of concepts, and
    the attributes ``concepts_mutually_exclusive`` and ``closed_environment`` can
    be changed. Note this is a overwriting change. For a valid call, at least one or
    more attributes should be specified. Otherwise the call will be just skipped without error.

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
    """

    args = [model_name, concepts_mutually_exclusive, closed_environment, concept_ids]
    if not any(map(lambda x: x is not None, args)):
      return self

    model = {
        "id": self.model_id,
    }  # type: typing.Dict[str, typing.Any]

    if model_name:
      model["name"] = model_name

    output_config = {}
    if concepts_mutually_exclusive is not None:
      # model["output_info"]["output_config"][
      #   "concepts_mutually_exclusive"] = concepts_mutually_exclusive
      output_config["concepts_mutually_exclusive"] = concepts_mutually_exclusive

    if closed_environment is not None:
      # model["output_info"]["output_config"]["closed_environment"] = closed_environment
      output_config["closed_environment"] = closed_environment

    data = {}
    if concept_ids is not None:
      # model["output_info"]["data"]["concepts"] = [{"id": concept_id} for concept_id in concept_ids]
      data["concepts"] = [{"id": concept_id} for concept_id in concept_ids]

    output_info = {}
    if output_config:
      output_info["output_config"] = output_config
    if data:
      output_info["data"] = data

    if output_info:
      model["output_info"] = output_info

    res = self.api.patch_model(model, action)
    model = res['models'][0]
    return self._to_obj(model)

  def delete_concepts(self, concept_ids):  # type: (typing.List[str]) -> Model
    """ delete concepts from a model

    Args:
      concept_ids: a list of concept IDs to be removed

    Returns:
      the Model object

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.delete_concepts(['cat', 'dog'])
    """

    model = self.update(action='remove', concept_ids=concept_ids)
    return model

  def list_versions(self):  # type: () -> dict
    """ list all model versions

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.list_versions()
    """

    res = self.api.get_model_versions(self.model_id)
    return res

  def get_version(self, version_id):  # type: (str) -> dict
    """ get model version info for a particular version

    Args:
      version_id: version id of the model version

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.get_version('model_version_id')
    """

    res = self.api.get_model_version(self.model_id, version_id)
    return res

  def delete_version(self, version_id):  # type: (str) -> dict
    """ delete model version by version_id

    Args:
      version_id: version id of the model version

    Returns:
      the JSON response

    Examples:
      >>> model = self.app.models.get('model_id')
      >>> model.delete_version('model_version_id')
    """

    res = self.api.delete_model_version(self.model_id, version_id)
    return res

  def create_version(self):  # type: () -> dict

    res = self.api.create_model_version(self.model_id)
    return res

  def get_inputs(self, version_id=None, page=1, per_page=20):
    # type: (typing.Optional[str], int, int) -> dict
    """
    Get all the inputs from the model or a specific model version.
    Without specifying a model version id, this will yield all inputs

    Args:
      version_id: model version id
      page: page number
      per_page: number of inputs to return for each page

    Returns:
      A list of Input objects
    """

    res = self.api.get_model_inputs(self.model_id, version_id, page, per_page)

    return res

  def send_concept_feedback(
      self,  # type: Model
      input_id,  # type: str
      url,  # type: str
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      feedback_info=None  # type: typing.Optional[FeedbackInfo]
  ):
    # type: (...) -> dict
    """
    Send feedback for this model

    Args:
      input_id: input id for the feedback
      url: the url of the input
      concepts: concepts that are present
      not_concepts: concepts that aren't present
      feedback_info: feedback info

    Returns:
      None
    """

    feedback_input = Image(
        url=url,
        image_id=input_id,
        concepts=concepts,
        not_concepts=not_concepts,
        feedback_info=feedback_info)
    res = self.api.send_model_feedback(self.model_id, self.model_version, feedback_input)

    return res

  def send_region_feedback(
      self,  # type: Model
      input_id,  # type: str
      url,  # type: str
      concepts=None,  # type: typing.Optional[typing.List[str]]
      not_concepts=None,  # type: typing.Optional[typing.List[str]]
      regions=None,  # type: typing.Optional[typing.List[Region]]
      feedback_info=None  # type: typing.Optional[FeedbackInfo]
  ):
    # type: (...) -> dict
    """
    Send feedback for this model

    Args:
      input_id: input id for the feedback
      url: the input url

    Returns:
      None
    """

    feedback_input = Image(
        url=url,
        image_id=input_id,
        concepts=concepts,
        not_concepts=not_concepts,
        regions=regions,
        feedback_info=feedback_info)
    res = self.api.send_model_feedback(self.model_id, self.model_version, feedback_input)

    return res

  def _to_obj(self, item):  # type: (dict) -> Model
    """ convert a model json object to Model object """
    return Model(self.api, item)

  def evaluate(self):  # type: () -> dict
    """ run model evaluation

    Returns:
      the model version data with evaluation metrics in JSON format
    """

    if self.model_version is None:
      raise UserError('To run model evaluation, please set the model_version field')

    res = self.api.run_model_evaluation(self.model_id, self.model_version)
    return res


class ModelSolutions(object):

  def __init__(self, solutions, model):  # type: (Solutions, Model) -> None
    self.moderation = ModelSolutionsModeration(solutions, model)  # type: ModelSolutionsModeration


class ModelSolutionsModeration(object):

  def __init__(self, solutions, model):  # type: (Solutions, Model) -> None
    self.solutions = solutions  # type: Solutions
    self.model = model  # type: Model

  def predict_by_url(self, url):  # type: (str) -> dict
    return self.solutions.moderation.predict_model(self.model.model_id, url)


class Concept(object):
  """ Clarifai Concept
  """

  def __init__(
      self,  # type: Concept
      concept_name=None,  # type: typing.Optional[str]
      concept_id=None,  # type: typing.Optional[str]
      app_id=None,  # type: typing.Optional[str]
      created_at=None,  # type: typing.Optional[str]
      value=None  # type: typing.Optional[float]
  ):
    self.concept_name = concept_name
    self.concept_id = concept_id
    self.app_id = app_id
    self.created_at = created_at
    self.value = value

  def dict(self):  # type: () -> dict

    data = {}  # type: typing.Dict[str, typing.Any]

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


class PublicModels(object):
  """
  A collection of already existing models provided by the API for immediate use.
  """

  # TODO(Rok) HIGH: Construct these with the solution object.
  def __init__(self, api):  # type: (ApiClient) -> None
    """ Ctor. """
    """ Apparel model recognizes clothing, accessories, and other fashion-related items. """
    self.apparel_model = Model(api, model_id='e0be3b9d6a454f0493ac3a30784001ff')  # type: Model
    """ Celebrity model identifies celebrities that closely resemble detected faces. """
    self.celebrity_model = Model(api, model_id='e466caa0619f444ab97497640cefc4dc')  # type: Model
    """ Color model recognizes dominant colors on an input. """
    self.color_model = Model(api, model_id='eeed0b6733a644cea07cf4c60f87ebb7')  # type: Model
    """ Demographics model predicts the age, gender, and cultural appearance. """
    self.demographics_model = Model(
        api, model_id='c0c0ac362b03416da06ab3fa36fb58e3')  # type: Model
    """ Face detection model detects the presence and location of human faces. """
    self.face_detection_model = Model(
        api, model_id='a403429f2ddf4b49b307e318f00e528b')  # type: Model
    """ 
    Face embedding model computes numerical embedding vectors using our Face detection model.
    """
    self.face_embedding_model = Model(
        api, model_id='d02b4508df58432fbb84e800597b8959')  # type: Model
    """ Focus model returns overall focus and identifies in-focus regions. """
    self.focus_model = Model(api, model_id='c2cf7cecd8a6427da375b9f35fcd2381')  # type: Model
    """ Food model recognizes food items and dishes, down to the ingredient level. """
    self.food_model = Model(api, model_id='bd367be194cf45149e75f01d59f77ba7')  # type: Model
    """ General embedding model computes numerical embedding vectors using our General model. """
    self.general_embedding_model = Model(
        api, model_id='bbb5f41425b8468d9b7a554ff10f8581')  # type: Model
    """ General model predicts most generally. """
    self.general_model = Model(api, model_id='aaa03c23b3724a16a56b629203edc62c')  # type: Model
    """ Landscape quality model predicts the quality of a landscape image. """
    self.landscape_quality_model = Model(
        api, model_id='bec14810deb94c40a05f1f0eb3c91403')  # type: Model
    """ Logo model detects and identifies brand logos. """
    self.logo_model = Model(api, model_id='c443119bf2ed4da98487520d01a0b1e3')  # type: Model
    """ Moderation model predicts inputs such as safety, gore, nudity, etc. """
    self.moderation_model = Model(api, model_id='d16f390eb32cad478c7ae150069bd2c6')  # type: Model
    """ NSFW model identifies different levels of nudity. """
    self.nsfw_model = Model(api, model_id='e9576d86d2004ed1a38ba0cf39ecb4b1')  # type: Model
    """ Portrait quality model predicts the quality of a portrait image. """
    self.portrait_quality_model = Model(
        api, model_id='de9bd05cfdbf4534af151beb2a5d0953')  # type: Model
    """ Textures & Patterns model predicts textures and patterns on an image. """
    self.textures_and_patterns_model = Model(
        api, model_id='fbefb47f9fdb410e8ce14f24f54b47ff')  # type: Model
    """ Travel model recognizes travel and hospitality-related concepts. """
    self.travel_model = Model(api, model_id='eee28c313d69466f836ab83287a54ed9')  # type: Model
    """ Wedding model recognizes wedding-related concepts bride, groom, flowers, and more. """
    self.wedding_model = Model(api, model_id='c386b7a870114f4a87477c0824499348')  # type: Model


class ApiClient(object):
  """ Handles auth and making requests for you.

  The constructor for API access. You must sign up at developer.clarifai.com first and create an
  application in order to generate your credentials for API access.

  Args:
    self: instance of ApiClient
    app_id: (DEPRECATED) the app_id for an application you've created in your Clarifai account.
    app_secret: (DEPRECATED) the app_secret for the same application.
    base_url: base URL of the API endpoints.
    api_key: the API key, used for authentication.
    quiet: if True then silence debug prints.
    log_level: log level from logging module
  """

  patch_actions = ['merge', 'remove', 'overwrite']
  concepts_patch_actions = ['overwrite']

  def __init__(
      self,  # type: ApiClient
      app_id=None,  # type: typing.Optional[str]
      app_secret=None,  # type: typing.Optional[str]
      base_url=None,  # type: typing.Optional[str]
      api_key=None,  # type: typing.Optional[str]
      quiet=True,  # type: bool
      log_level=None  # type: typing.Optional[int]
  ):

    if app_id or app_secret:
      warnings.warn('Tokens deprecated', DeprecationWarning)
      raise DeprecationWarning(TOKENS_DEPRECATED_MESSAGE)

    if not log_level:
      if quiet:
        log_level = logging.ERROR
      else:
        log_level = logging.DEBUG
    logger.setLevel(log_level)

    if not api_key:
      api_key = self._read_key_from_env_or_os()
    self.api_key = api_key

    if not base_url:
      base_url = self._read_base_from_env_or_os()
    parsed = urlparse(base_url)
    scheme = 'https' if parsed.scheme == '' else parsed.scheme
    base_url_parsed = parsed.path if not parsed.netloc else parsed.netloc
    self.base_url = base_url_parsed
    self.scheme = scheme  # type: typing.Optional[str]
    self.basev2 = urljoin(scheme + '://', base_url_parsed)  # type: str
    logger.debug("Base url: %s", self.basev2)
    self.token = None
    self.headers = None

    self.session = self._make_requests_session()  # type: requests.Session

  def _make_requests_session(self):  # type: () -> requests.Session
    http_adapter = requests.adapters.HTTPAdapter(
        max_retries=RETRIES, pool_connections=CONNECTIONS, pool_maxsize=CONNECTIONS)

    session = requests.Session()
    session.mount('http://', http_adapter)
    session.mount('https://', http_adapter)
    return session

  def _read_key_from_env_or_os(self):  # type: () -> typing.Optional[str]
    conf_file = self._config_file_path()
    env_api_key = os.environ.get('CLARIFAI_API_KEY')
    if env_api_key:
      logger.debug("Using env. variable CLARIFAI_API_KEY for API key")
      return env_api_key
    elif os.path.exists(conf_file):
      parser = ConfigParser()
      parser.optionxform = str  # type: ignore

      with open(conf_file, 'r') as fdr:
        parser.readfp(fdr)

      if parser.has_option('clarifai', 'CLARIFAI_API_KEY'):
        return parser.get('clarifai', 'CLARIFAI_API_KEY')
    return None

  def _read_base_from_env_or_os(self):  # type: () -> str
    conf_file = self._config_file_path()
    env_clarifai_api_base = os.environ.get('CLARIFAI_API_BASE')
    if env_clarifai_api_base:
      base_url = env_clarifai_api_base
    elif os.path.exists(conf_file):
      parser = ConfigParser()
      parser.optionxform = str  # type: ignore

      with open(conf_file, 'r') as fdr:
        parser.readfp(fdr)

      if parser.has_option('clarifai', 'CLARIFAI_API_BASE'):
        base_url = parser.get('clarifai', 'CLARIFAI_API_BASE')
      else:
        base_url = 'api.clarifai.com'
    else:
      base_url = 'api.clarifai.com'
    return base_url

  def _config_file_path(self):  # type: () -> str
    if platform.system() == 'Windows':
      home_dir = os.environ.get('HOMEPATH', '.')
    else:
      home_dir = os.environ.get('HOME', '.')
    conf_file = os.path.join(home_dir, '.clarifai', 'config')
    return conf_file

  def get_token(self):  # type: () -> None
    """
    Tokens are deprecated, please switch to API keys. See here how:
   "http://help.clarifai.com/api/account-related/all-about-api-keys"
    """
    warnings.warn('Tokens deprecated', DeprecationWarning)
    raise DeprecationWarning(TOKENS_DEPRECATED_MESSAGE)

  def set_token(self, token):  # type: (str) -> None
    """
    Tokens are deprecated, please switch to API keys. See here how:
   "http://help.clarifai.com/api/account-related/all-about-api-keys"
    """
    warnings.warn('Tokens deprecated', DeprecationWarning)
    raise DeprecationWarning(TOKENS_DEPRECATED_MESSAGE)

  def delete_token(self):  # type: () -> None
    """
    Tokens are deprecated, please switch to API keys. See here how:
   "http://help.clarifai.com/api/account-related/all-about-api-keys"
    """
    warnings.warn('Tokens deprecated', DeprecationWarning)
    raise DeprecationWarning(TOKENS_DEPRECATED_MESSAGE)

  def _grpc_stub(self):  # type: () -> V2Stub
    return V2Stub(  # type: ignore
        GRPCJSONChannel(
            session=self.session, key=self.api_key, base_url=self.basev2, service_descriptor=_V2))

  def _grpc_request(self, method, argument):  # type: (typing.Callable, typing.Any) -> dict

    # only retry under when status_code is non-200, under max-tries
    # and under some circumstances
    max_attempts = 3
    for attempt_num in range(1, max_attempts + 1):
      try:
        res = method(argument)

        dict_res = protobuf_to_dict(res)
        logger.debug("\nRESULT:\n%s", pformat(dict_res))
        return dict_res
      except ApiError as ex:
        if ex.response is not None:
          status_code = ex.response.status_code

          if attempt_num == max_attempts:
            logger.debug("Failed after %d retries" % max_attempts)
            raise

          # handle Gateway Error, normally retry will solve the problem
          if int(status_code / 100) == 5:
            continue

          # handle throttling
          # back off with 2/4/8 seconds
          # normally, this will be settled in 1 or 2 retries
          if status_code == 429:
            time.sleep(2**attempt_num)
            continue

        # in other cases, error out without retrying
        raise

    # The for loop above either returns or raises.
    raise Exception('This code is never reached')

  def add_inputs(self, objs):  # type: (typing.List[Input]) -> dict
    """ Add a list of Images or Videos to an application.

    Args:
      objs: A list of Image or Video objects.

    Returns:
      raw JSON response from the API server, with a list of inputs and corresponding import
      status
    """
    if not isinstance(objs, list):
      raise UserError("objs must be a list")

    inputs_pb = []
    for obj in objs:
      if not isinstance(obj, (Image, Video)):
        raise UserError("Not valid type of content to add. Must be Image or Video")
      if obj.input_id:
        if not isinstance(obj.input_id, basestring):
          raise UserError("Not valid input ID. Must be a string or None")
        if '/' in obj.input_id:
          raise UserError('Not valid input ID. Cannot contain character: "/"')

      resulting_protobuf = dict_to_protobuf(InputPB, obj.dict())

      inputs_pb.append(resulting_protobuf)

    return self._grpc_request(self._grpc_stub().PostInputs, PostInputsRequest(inputs=inputs_pb))

  def search_inputs(self, query, page=1, per_page=20):  # type: (dict, int, int) -> dict
    """ Search an application and get predictions (optional)

    Args:
      query: the JSON query object that complies with Clarifai RESTful API
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      raw JSON response from the API server, with a list of inputs and corresponding ranking
      scores
    """

    q = dict_to_protobuf(Query, query)

    return self._grpc_request(
        self._grpc_stub().PostSearches,
        PostSearchesRequest(query=q, pagination=Pagination(page=page, per_page=per_page)))

  def get_input(self, input_id):  # type: (str) -> dict
    """ Get a single image by it's id.

    Args:
      input_id: the id of the Image.

    Returns:
      raw JSON response from the API server

      HTTP code:
       200 for Found
       404 for Not Found
    """

    return self._grpc_request(self._grpc_stub().GetInput, GetInputRequest(input_id=input_id))

  def get_inputs(self, page=1, per_page=20):  # type: (int, int) -> dict
    """ List all images for the Application, with pagination

    Args:
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      raw JSON response from the API server, with paginated list of inputs and corresponding
      status
    """

    return self._grpc_request(self._grpc_stub().ListInputs,
                              ListInputsRequest(page=page, per_page=per_page))

  def get_inputs_status(self):  # type: () -> dict
    """ Get counts of inputs in the Application.

    Returns:
      counts of the inputs, including processed, processing, etc. in JSON format.
    """

    return self._grpc_request(self._grpc_stub().GetInputCount, GetInputCountRequest())

  def delete_input(self, input_id):  # type: (str) -> dict
    """ Delete a single input by its id.

    Args:
      input_id: the id of the input

    Returns:
      status of the deletion, in JSON format.
    """

    return self._grpc_request(self._grpc_stub().DeleteInput, DeleteInputRequest(input_id=input_id))

  def delete_inputs(self, input_ids):  # type: (typing.List[str]) -> dict
    """ bulk delete inputs with a list of input IDs

    Args:
      input_ids: the ids of the input, in a list

    Returns:
      status of the bulk deletion, in JSON format.
    """

    return self._grpc_request(self._grpc_stub().DeleteInputs, DeleteInputsRequest(ids=input_ids))

  def delete_all_inputs(self):  # type: () -> dict
    """ delete all inputs from the application

    Returns:
      status of the deletion, in JSON format.
    """

    return self._grpc_request(self._grpc_stub().DeleteInputs, DeleteInputsRequest(delete_all=True))

  def patch_inputs(self, action, inputs):
    # type: (str, typing.List[typing.Union[Input]]) -> dict
    """ bulk update inputs, to delete or modify concepts

    Args:
      action: "merge" or "remove" or "overwrite"
      inputs: list of inputs

    Returns:
      the update status, in JSON format

    """

    if action not in self.patch_actions:
      raise UserError("action not supported.")

    inputs_pb = []
    for input_ in inputs:
      input_dict = input_.dict()

      if 'data' not in input_dict:
        continue

      reduced_input_dict = copy.deepcopy(input_dict)
      for key in input_dict['data'].keys():
        if key not in ['concepts', 'metadata', 'regions']:
          del reduced_input_dict['data'][key]

      resulting_protobuf = dict_to_protobuf(InputPB, reduced_input_dict)

      inputs_pb.append(resulting_protobuf)

    return self._grpc_request(self._grpc_stub().PatchInputs,
                              PatchInputsRequest(action=action, inputs=inputs_pb))

  def get_concept(self, concept_id):  # type: (str) -> dict
    """ Get a single concept by it's id.

    Args:
      concept_id: unique id of the concept

    Returns:
      the concept in JSON format with HTTP 200 Status
      or HTTP 404 with concept not found
    """
    return self._grpc_request(
        self._grpc_stub().GetConcept, GetConceptRequest(concept_id=concept_id))

  def get_concepts(self, page=1, per_page=20):  # type: (int, int) -> dict
    """ List all concepts for the Application.

    Args:
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page

    Returns:
      a list of concepts in JSON format
    """

    return self._grpc_request(self._grpc_stub().ListConcepts,
                              ListConceptsRequest(page=page, per_page=per_page))

  # TODO(Rok) MEDIUM: Allow skipping concept_names.
  def add_concepts(self, concept_ids, concept_names):
    # type: (typing.List[str], typing.List[typing.Optional[str]]) -> dict
    """ Add a list of concepts

    Args:
      concept_ids: a list of concept id
      concept_names: a list of concept name

    Returns:
      a list of concepts in JSON format along with the status code
    """

    if not isinstance(concept_ids, list) or \
        not isinstance(concept_names, list):
      raise UserError('concept_ids and concept_names should be both be list ')

    if len(concept_ids) != len(concept_names):
      raise UserError('length of concept id list should match length of the concept name list')

    concepts = []
    for cid, cname in zip(concept_ids, concept_names):
      if cname is None:
        concept = {'id': cid}
      else:
        concept = {'id': cid, 'name': cname}
      concepts.append(dict_to_protobuf(ConceptPB, concept))

    return self._grpc_request(
        self._grpc_stub().PostConcepts, PostConceptsRequest(concepts=concepts))

  def search_concepts(self, term, page=1, per_page=20, language=None):
    # type: (str, int, int, typing.Optional[str]) -> dict
    """ Search concepts

    Args:
      term: search term with wildcards
      page: the page of results to get, starts at 1.
      per_page: number of results returned per page
      language: language to search for the translation

    Returns:
      a list of concepts in JSON format along with the status code

    """

    return self._grpc_request(
        self._grpc_stub().PostConceptsSearches,
        PostConceptsSearchesRequest(
            concept_query=ConceptQuery(name=term, language=language),
            pagination=Pagination(page=page, per_page=per_page)))

  def patch_concepts(self, action, concepts):  # type: (str, typing.List[Concept]) -> dict
    """ bulk update concepts, to delete or modify concepts

    Args:
      action: only "overwrite" is supported
      concepts: a list of Concept(concept_name='', concept_id='')

    Returns:
      the update status, in JSON format

    """

    if action not in self.concepts_patch_actions:
      raise UserError("action not supported.")

    concepts_pb = [dict_to_protobuf(ConceptPB, c.dict()) for c in concepts]
    return self._grpc_request(self._grpc_stub().PatchConcepts,
                              PatchConceptsRequest(action=action, concepts=concepts_pb))

  def get_models(self, page=1, per_page=20):  # type: (int, int) -> dict
    """ get all models with pagination

    Args:
      page: page number
      per_page: number of models to return per page

    Returns:
      a list of models in JSON format
    """

    response = self._grpc_request(self._grpc_stub().ListModels,
                                  ListModelsRequest(page=page, per_page=per_page))
    return response

  def get_model(self, model_id,
                model_version_id=None):  # type: (str, typing.Optional[str]) -> dict
    """ get model basic info by model id

    Args:
      model_id: the unique identifier of the model
      model_version_id: the unique identifier of the model version

    Returns:
      the model info in JSON format
    """

    return self._grpc_request(
        self._grpc_stub().GetModel,
        GetModelRequest(model_id=_escape(model_id), version_id=model_version_id))

  def get_model_output_info(self, model_id,
                            model_version_id=None):  # type: (str, typing.Optional[str]) -> dict
    """ get model output info by model id

    Args:
      model_id: the unique identifier of the model
      model_version_id: the unique identifier of the model version

    Returns:
      the model info with output_info in JSON format
    """
    return self._grpc_request(
        self._grpc_stub().GetModelOutputInfo,
        GetModelRequest(model_id=_escape(model_id), version_id=model_version_id))

  def get_model_versions(self, model_id, page=1, per_page=20):  # type: (str, int, int) -> dict
    """ get model versions

    Args:
      model_id: the unique identifier of the model
      page: page number
      per_page: the number of versions to return per page

    Returns:
      a list of model versions in JSON format
    """

    return self._grpc_request(
        self._grpc_stub().ListModelVersions,
        ListModelVersionsRequest(model_id=model_id, page=page, per_page=per_page))

  def get_model_version(self, model_id, version_id):  # type: (str, str) -> dict
    """ get model info for a specific model version

    Args:
      model_id: the unique identifier of a model
      version_id: the model version id
    """

    return self._grpc_request(self._grpc_stub().GetModelVersion,
                              GetModelVersionRequest(model_id=model_id, version_id=version_id))

  def delete_model_version(self, model_id, model_version):  # type: (str, str) -> dict
    """ delete a model version """

    return self._grpc_request(
        self._grpc_stub().DeleteModelVersion,
        DeleteModelVersionRequest(model_id=_escape(model_id), version_id=model_version))

  def delete_model(self, model_id):  # type: (str) -> dict
    """ delete a model """

    return self._grpc_request(
        self._grpc_stub().DeleteModel, DeleteModelRequest(model_id=_escape(model_id)))

  def delete_models(self, model_ids):  # type: (typing.List[str]) -> dict
    """ delete the models """

    return self._grpc_request(
        self._grpc_stub().DeleteModels,
        DeleteModelsRequest(ids=[_escape(id_) for id_ in model_ids]))

  def delete_all_models(self):  # type: () -> dict
    """ delete all models """

    return self._grpc_request(self._grpc_stub().DeleteModels, DeleteModelsRequest(delete_all=True))

  def get_model_inputs(self, model_id, version_id=None, page=1, per_page=20):
    # type: (str, typing.Optional[str], int, int) -> dict
    """ get inputs for the latest model or a specific model version """

    if version_id:
      version_id = _escape(version_id)

    return self._grpc_request(
        self._grpc_stub().ListModelInputs,
        ListModelInputsRequest(
            model_id=_escape(model_id), version_id=version_id, page=page, per_page=per_page))

  def search_models(self, name=None, model_type=None):
    # type: (typing.Optional[str], typing.Optional[str]) -> dict
    """ search model by name and type """

    return self._grpc_request(
        self._grpc_stub().PostModelsSearches,
        PostModelsSearchesRequest(model_query=ModelQuery(name=name, type=model_type)))

  def create_model(
      self,  # type: ApiClient
      model_id,  # type: str
      model_name=None,  # type: typing.Optional[str]
      concepts=None,  # type: typing.Optional[typing.List[str]]
      concepts_mutually_exclusive=False,  # type: bool
      closed_environment=False,  # type: bool
      hyper_parameters=None  # type: typing.Optional[dict]
  ):
    # type: (...) -> dict
    """
    Create a new model.

    Args:
      model_id: The model ID
      model_name:  The model name
      concepts: A list of concept IDs that this model will use. A better name here would be
                concept_ids
      concepts_mutually_exclusive: Whether the concepts are mutually exclusive
      closed_environment: Whether the concept environment is closed
      hyper_parameters: The hyper parameters

    Returns:
      A model dictionary.
    """

    if not model_name:
      model_name = model_id

    data = None
    if concepts:
      data = dict_to_protobuf(DataPB,
                              {'concepts': [{
                                  'id': concept_id
                              } for concept_id in concepts]})

    hyper_parameters_pb = None
    if hyper_parameters:
      hyper_parameters_pb = dict_to_protobuf(Struct, hyper_parameters)

    output_info = OutputInfoPB(
        data=data,
        output_config=OutputConfigPB(
            concepts_mutually_exclusive=concepts_mutually_exclusive,
            closed_environment=closed_environment,
            hyper_params=hyper_parameters_pb))

    model = ModelPB(id=model_id, name=model_name, output_info=output_info)

    return self._grpc_request(self._grpc_stub().PostModels, PostModelsRequest(model=model))

  def patch_model(self, model, action='merge'):  # type: (dict, str) -> dict
    """
    Args:
      model: the model dictionary
      action: the patch action

    Returns:
      the model object
    """

    if action not in self.patch_actions:
      raise UserError("action not supported.")

    model_pb = dict_to_protobuf(ModelPB, model)

    return self._grpc_request(self._grpc_stub().PatchModels,
                              PatchModelsRequest(action=action, models=[model_pb]))

  def create_model_version(self, model_id):  # type: (str) -> dict
    """ train for a model """

    return self._grpc_request(
        self._grpc_stub().PostModelVersions, PostModelVersionsRequest(model_id=_escape(model_id)))

  def predict_model(
      self,  # type: ApiClient
      model_id,  # type: str
      objs,  # type: typing.List[Input]
      version_id=None,  # type: typing.Optional[str]
      model_output_info=None  # type: typing.Optional[ModelOutputInfo]
  ):
    # type: (...) -> dict
    if not isinstance(objs, list):
      raise UserError("objs must be a list")

    for i, obj in enumerate(objs):
      if not isinstance(obj, (Image, Video)):
        raise UserError(
            "Object at position %d is not a valid type of content to add. Must be Image or Video" %
            i)

    inputs_pb = []
    for input_ in objs:
      data_ = input_.dict().get('data')
      if data_:
        inputs_pb.append(dict_to_protobuf(InputPB, input_.dict()))

    model = None
    if model_output_info:
      model_output_info_dict = model_output_info.dict().get('output_info')
      if model_output_info_dict:
        output_info = dict_to_protobuf(OutputInfoPB, model_output_info_dict)
        model = ModelPB(output_info=output_info)

    return self._grpc_request(
        self._grpc_stub().PostModelOutputs,
        PostModelOutputsRequest(
            model_id=_escape(model_id), version_id=version_id, inputs=inputs_pb, model=model))

  def get_workflows(self, public_only=False):  # type: (bool) -> dict
    """ get all workflows with pagination

    Args:
      public_only: whether to get public workflow

    Returns:
      a list of workflows in JSON format
    """

    if public_only:
      return self._grpc_request(self._grpc_stub().ListPublicWorkflows,
                                ListPublicWorkflowsRequest())
    else:
      return self._grpc_request(self._grpc_stub().ListWorkflows, ListWorkflowsRequest())

  def get_workflow(self, workflow_id):  # type: (str) -> dict
    """ get workflow basic info by workflow id

    Args:
      workflow_id: the unique identifier of the workflow

    Returns:
      the workflow info in JSON format
    """

    return self._grpc_request(
        self._grpc_stub().GetWorkflow, GetWorkflowRequest(workflow_id=workflow_id))

  def predict_workflow(self, workflow_id, objs, output_config=None):
    # type: (str, typing.List[Input], typing.Optional[ModelOutputConfig]) -> dict

    if not isinstance(objs, list):
      raise UserError("objs must be a list")

    inputs_pb = []
    for i, obj in enumerate(objs):
      if not isinstance(obj, (Image, Video)):
        raise UserError(
            "Object at position %d is not a valid type of content to add. Must be Image or Video" %
            i)

      inputs_pb.append(dict_to_protobuf(InputPB, obj.dict()))

    output_config_pb = None
    if output_config:
      output_config_ = output_config.dict().get('output_config')
      if output_config_:
        output_config_pb = dict_to_protobuf(OutputConfigPB, output_config_)

    return self._grpc_request(
        self._grpc_stub().PostWorkflowResults,
        PostWorkflowResultsRequest(
            workflow_id=_escape(workflow_id), inputs=inputs_pb, output_config=output_config_pb))

  def send_model_feedback(self, model_id, version_id, obj):
    # type: (str, typing.Optional[str], Input) -> dict

    input_pb = dict_to_protobuf(InputPB, obj.dict())

    return self._grpc_request(
        self._grpc_stub().PostModelFeedback,
        PostModelFeedbackRequest(
            model_id=_escape(model_id), version_id=version_id, input=input_pb))

  def send_search_feedback(self, obj):  # type: (Input) -> dict
    input_dict = obj.dict()

    input_pb = dict_to_protobuf(InputPB, input_dict)

    return self._grpc_request(
        self._grpc_stub().PostSearchFeedback, PostSearchFeedbackRequest(input=input_pb))

  def predict_concepts(self, objs, lang=None):
    # type: (typing.List[Input], typing.Optional[str]) -> dict

    models = self.search_models(name='general-v1.3', model_type='concept')
    model = models['models'][0]
    model_id = model['id']

    model_output_info = ModelOutputInfo(output_config=ModelOutputConfig(language=lang))
    return self.predict_model(model_id, objs, model_output_info=model_output_info)

  def predict_colors(self, objs):  # type: (typing.List[Input]) -> dict

    models = self.search_models(name='color', model_type='color')
    model = models['models'][0]
    model_id = model['id']

    return self.predict_model(model_id, objs)

  def predict_embed(self, objs, model='general*'):  # type: (typing.List[Input], str) -> dict

    found_models = self.search_models(name=model, model_type='embed')
    found_model = found_models['models'][0]
    found_model_id = found_model['id']

    return self.predict_model(found_model_id, objs)

  def run_model_evaluation(self, model_id, version_id):  # type: (str, str) -> dict
    """ run model evaluation by model id and by version id

    Args:
      model_id: the unique identifier of the model
      version_id: the model version id

    Returns:
      the model version data with evaluation metrics in JSON format
    """

    return self._grpc_request(
        self._grpc_stub().PostModelVersionMetrics,
        PostModelVersionMetricsRequest(model_id=_escape(model_id), version_id=version_id))


class pagination(object):

  def __init__(self, page=1, per_page=20):  # type: (int, int) -> None
    self.page = page
    self.per_page = per_page

  def dict(self):  # type: () -> dict
    return {'page': self.page, 'per_page': self.per_page}


class ApiStatus(object):
  """ Clarifai API Status Code """

  def __init__(self, item):  # type: (dict) -> None
    self.code = item['code']
    self.description = item['description']

  def dict(self):  # type: () -> dict
    d = {'status': {'code': self.code, 'description': self.description}}

    return d


class ApiResponse(object):
  """ Clarifai API Response """

  def __init__(self):  # type: () -> None
    self.status = None


class InputCounts(object):
  """ input counts for upload status """

  def __init__(self, item):  # type: (dict) -> None
    if not item.get('counts'):
      raise UserError('unable to initialize. need a dict with key=counts')

    counts = item['counts']

    # TODO(Rok) MEDIUM: Add the "processing" field here and in dict().
    self.processed = counts['processed']
    self.to_process = counts['to_process']
    self.errors = counts['errors']

  def dict(self):  # type: () -> dict
    d = {
        'counts': {
            'processed': self.processed,
            'to_process': self.to_process,
            'errors': self.errors
        }
    }
    return d


class ModelOutputInfo(object):

  def __init__(self, concepts=None, output_config=None):
    # type: (typing.Optional[typing.List[Concept]], typing.Optional[ModelOutputConfig]) -> None
    self.concepts = concepts  # type: typing.Optional[typing.List[Concept]]
    self.output_config = output_config  # type: typing.Optional[ModelOutputConfig]

  def dict(self):  # type: () -> dict
    output_info = {}
    if self.output_config:
      output_info.update(self.output_config.dict())
    if self.concepts:
      output_info.update({'data': {'concepts': [concept.dict() for concept in self.concepts]}})

    if output_info:
      return {'output_info': output_info}
    else:
      return {}


class ModelOutputConfig(object):

  def __init__(
      self,  # type: ModelOutputConfig
      mutually_exclusive=None,  # type: typing.Optional[bool]
      closed_environment=None,  # type: typing.Optional[bool]
      language=None,  # type: typing.Optional[str]
      min_value=None,  # type: typing.Optional[float]
      max_concepts=None,  # type: typing.Optional[int]
      select_concepts=None,  # type: typing.Optional[typing.List[Concept]]
      sample_ms=None  # type: typing.Optional[int]
  ):
    # type: (...) -> None
    self.concepts_mutually_exclusive = mutually_exclusive
    self.closed_environment = closed_environment
    self.language = language
    self.min_value = min_value
    self.max_concepts = max_concepts
    self.select_concepts = select_concepts
    self.sample_ms = sample_ms

  def dict(self):  # type: () -> dict
    output_config = {}

    if self.concepts_mutually_exclusive is not None:
      output_config['concepts_mutually_exclusive'] = self.concepts_mutually_exclusive

    if self.closed_environment is not None:
      output_config['closed_environment'] = self.closed_environment

    if self.language is not None:
      output_config['language'] = self.language

    if self.min_value is not None:
      output_config['min_value'] = self.min_value

    if self.max_concepts is not None:
      output_config['max_concepts'] = self.max_concepts

    if self.select_concepts is not None:
      output_config['select_concepts'] = [c.dict() for c in self.select_concepts]

    if self.sample_ms is not None:
      output_config['sample_ms'] = self.sample_ms

    if output_config:
      return {'output_config': output_config}
    else:
      return {}


class BoundingBox(object):

  def __init__(self, top_row, left_col, bottom_row, right_col):
    # type: (float, float, float, float) -> None
    self.top_row = top_row
    self.left_col = left_col
    self.bottom_row = bottom_row
    self.right_col = right_col

  def dict(self):  # type: () -> dict
    data = {
        'bounding_box': {
            'top_row': self.top_row,
            'left_col': self.left_col,
            'bottom_row': self.bottom_row,
            'right_col': self.right_col
        }
    }

    return data


class RegionInfo(object):

  def __init__(self, bbox=None, feedback_type=None):
    # type: (typing.Optional[BoundingBox], typing.Optional[FeedbackType]) -> None
    self.bbox = bbox
    self.feedback_type = feedback_type

  def dict(self):  # type: () -> dict

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

  def __init__(
      self,  # type: Region
      region_info=None,  # type: typing.Optional[RegionInfo]
      concepts=None,  # type: typing.Optional[typing.List[Concept]]
      face=None,  # type: typing.Optional[Face]
      region_id=None  # type: typing.Optional[str]
  ):
    # type: (...) -> None

    self.region_info = region_info
    self.concepts = concepts
    self.face = face
    self.region_id = region_id

  def dict(self):  # type: () -> dict
    data = {}
    if self.concepts:
      data['concepts'] = [c.dict() for c in self.concepts]
    if self.face:
      data.update(self.face.dict())

    region = {}
    if self.region_info:
      region.update(self.region_info.dict())
    if self.region_id:
      region['id'] = self.region_id
    if data:
      region['data'] = data
    return region


class Face(object):

  def __init__(
      self,  # type: Face
      identity=None,  # type: typing.Optional[FaceIdentity]
      age_appearance=None,  # type: typing.Optional[FaceAgeAppearance]
      gender_appearance=None,  # type: typing.Optional[FaceGenderAppearance]
      multicultural_appearance=None  # type: typing.Optional[FaceMulticulturalAppearance]
  ):
    # type: (...) -> None

    self.identity = identity
    self.age_appearance = age_appearance
    self.gender_appearance = gender_appearance
    self.multicultural_appearance = multicultural_appearance

  def dict(self):  # type: () -> dict

    data = {'face': {}}

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

  def __init__(self, concepts):  # type: (typing.List[Concept]) -> None
    self.concepts = concepts

  def dict(self):  # type: () -> dict
    data = {'identity': {'concepts': [c.dict() for c in self.concepts]}}
    return data


class FaceAgeAppearance(object):

  def __init__(self, concepts):  # type: (typing.List[Concept]) -> None
    self.concepts = concepts

  def dict(self):  # type: () -> dict
    data = {'age_appearance': {'concepts': [c.dict() for c in self.concepts]}}
    return data


class FaceGenderAppearance(object):

  def __init__(self, concepts):  # type: (typing.List[Concept]) -> None
    self.concepts = concepts

  def dict(self):  # type: () -> dict
    data = {'gender_appearance': {'concepts': [c.dict() for c in self.concepts]}}
    return data


class FaceMulticulturalAppearance(object):

  def __init__(self, concepts):  # type: (typing.List[Concept]) -> None
    self.concepts = concepts

  def dict(self):  # type: () -> dict
    data = {'multicultural_appearance': {'concepts': [c.dict() for c in self.concepts]}}
    return data
