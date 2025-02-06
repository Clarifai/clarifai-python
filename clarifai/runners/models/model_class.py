import itertools
from abc import ABC, abstractmethod
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2


class ModelClass(ABC):

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This method is used for input/output proto data conversion"""
    method_name = request.model.model_version.output_info.params.pop('_clarifai_method_name',
                                                                     'predict')
    if method_name not in self._clarifai_model_methods:
      raise ValueError(f"Method {method_name} a model function of {self.__class__.__name__}")
    method_info = self._clarifai_model_methods[method_name]
    if method_info.method_type != 'predict':
      raise ValueError(
          f"Method {method_name} was called as predict, but is a {method_info.method_type} method")
    # TODO check and convert input/output types from reqeust
    return self.predict(request)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    method_name = request.model.model_version.output_info.params.pop('_clarifai_method_name',
                                                                     'generate')
    if method_name not in self._clarifai_model_methods:
      raise ValueError(f"Method {method_name} a model function of {self.__class__.__name__}")
    method_info = self._clarifai_model_methods[method_name]
    if method_info.method_type != 'generate':
      raise ValueError(
          f"Method {method_name} was called as generate, but is a {method_info.method_type} method"
      )
    return self.generate(request)

  def stream_wrapper(self, request_stream: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    first_request = next(request_stream)
    request_stream = itertools.chain([first_request], request_stream)

    method_name = first_request.model.model_version.output_info.params.pop(
        '_clarifai_method_name', 'stream')
    if method_name not in self._clarifai_model_methods:
      raise ValueError(f"Method {method_name} a model function of {self.__class__.__name__}")
    method_info = self._clarifai_model_methods[method_name]
    if method_info.method_type != 'stream':
      raise ValueError(
          f"Method {method_name} was called as stream, but is a {method_info.method_type} method")
    return self.stream(request_stream)

  @abstractmethod
  def load_model(self):
    raise NotImplementedError("load_model() not implemented")

  @abstractmethod
  def predict(self,
              request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    raise NotImplementedError("run_input() not implemented")

  @abstractmethod
  def generate(self, request: service_pb2.PostModelOutputsRequest
              ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("generate() not implemented")

  @abstractmethod
  def stream(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
            ) -> Iterator[service_pb2.MultiOutputResponse]:
    raise NotImplementedError("stream() not implemented")

  @classmethod
  def _create_method_registry(cls):
    if '_clarifai_model_methods' in cls.__dict__:  # needs to be in __dict__ directly, not inherited
      return
    # go up the class hierarchy to find all decorated methods, and add to registry of current class
    cls._clarifai_model_methods = {}
    for base in reversed(cls.__mro__):
      for name, method in base.__dict__.items():
        method_info = getattr(method, '_clarfai_model_method', None)
        if not method_info:
          continue
        cls._clarifai_model_methods[name] = method_info
    # check for generic predict(request) -> response, etc. methods
    for name in ('predict', 'generate', 'stream'):
      try:
        method = getattr(cls, name)
      except AttributeError:
        continue
      if not hasattr(method, '_clarfai_model_method'):  # not already put in registry
        cls._clarifai_model_methods[name] = _MethodInfo(method, name)

  def __new__(cls, *args, **kwargs):
    cls._create_method_registry()
    return super().__new__(cls, *args, **kwargs)


class _MethodInfo:

  def __init__(self, method, method_type):
    self.method = method
    self.method_type = method_type
    self.name = method.__name__
    # TODO arg types, return type, etc.


def predict(method):
  method._clarfai_model_method = _MethodInfo(method, 'predict')
  return method


def generate(method):
  method._clarfai_model_method = _MethodInfo(method, 'generate')
  return method


def stream(method):
  method._clarfai_model_method = _MethodInfo(method, 'stream')
  return method
