import itertools
from abc import ABC
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2

_METHOD_INFO_ATTR = '_cf_model_method'
_METHOD_REGISTRY = {}  # class -> {method_name -> _MethodInfo}


class ModelClass(ABC):

  def load_model(self):
    raise NotImplementedError("load_model() not implemented")

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This method is used for input/output proto data conversion"""
    method_name = request.model.model_version.output_info.params.pop('_clarifai_method_name',
                                                                     'predict')
    methods = _METHOD_REGISTRY.get(self.__class__)
    method_info = methods.get(method_name)
    if method_info is None:
      raise ValueError(
          f"Method {method_name} is not a model function of {self.__class__.__name__}")
    if method_info.method_type != 'predict':
      raise ValueError(
          f"Method {method_name} was called as predict, but is a {method_info.method_type} method")
    # TODO check and convert input/output types from reqeust
    method = getattr(self, method_name)
    return method(request)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    method_name = request.model.model_version.output_info.params.pop('_clarifai_method_name',
                                                                     'generate')
    methods = _METHOD_REGISTRY.get(self.__class__)
    method_info = methods.get(method_name)
    if method_info is None:
      raise ValueError(
          f"Method {method_name} is not a model function of {self.__class__.__name__}")
    if method_info.method_type != 'generate':
      raise ValueError(
          f"Method {method_name} was called as generate, but is a {method_info.method_type} method"
      )
    # TODO check and convert input/output types from reqeust
    method = getattr(self, method_name)
    return method(request)

  def stream_wrapper(self, request_stream: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    first_request = next(request_stream)
    request_stream = itertools.chain([first_request], request_stream)

    method_name = first_request.model.model_version.output_info.params.pop(
        '_clarifai_method_name', 'stream')
    methods = _METHOD_REGISTRY.get(self.__class__)
    method_info = methods.get(method_name)
    if method_info is None:
      raise ValueError(
          f"Method {method_name} is not a model function of {self.__class__.__name__}")
    if method_info.method_type != 'stream':
      raise ValueError(
          f"Method {method_name} was called as stream, but is a {method_info.method_type} method")
    # TODO check and convert input/output types from reqeust
    method = getattr(self, method_name)
    return method(request_stream)

  def __new__(cls, *args, **kwargs):
    cls._register_cf_model_methods()
    return super().__new__(cls, *args, **kwargs)

  @classmethod
  def _register_cf_model_methods(cls):
    if cls in _METHOD_REGISTRY:
      return
    # go up the class hierarchy to find all decorated methods, and add to registry of current class
    methods = {}
    for base in reversed(cls.__mro__):
      for name, method in base.__dict__.items():
        method_info = getattr(method, _METHOD_INFO_ATTR, None)
        if not method_info:
          continue
        methods[name] = method_info
    # check for generic predict(request) -> response, etc. methods
    for name in ('predict', 'generate', 'stream'):
      try:
        method = getattr(cls, name)
      except AttributeError:
        continue
      if not hasattr(method, _METHOD_INFO_ATTR):  # not already put in registry
        methods[name] = _MethodInfo(method, method_type=name)
    # set method table for this class in the registry
    _METHOD_REGISTRY[cls] = methods


class _MethodInfo:

  def __init__(self, method, method_type):
    self.method_name = method.__name__
    self.method_type = method_type
    # TODO arg types, return type, etc.


def predict(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'predict'))
  return method


def generate(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'generate'))
  return method


def stream(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'stream'))
  return method
