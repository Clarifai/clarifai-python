import inspect
import itertools
import logging
import os
import traceback
import types
from abc import ABC
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.runners.utils import data_handler
from clarifai.runners.utils.method_signatures import (build_function_signature, deserialize,
                                                      serialize, signatures_to_json)

_METHOD_INFO_ATTR = '_cf_method_info'

_RAISE_EXCEPTIONS = os.getenv("RAISE_EXCEPTIONS", "false").lower() == "true"


class ModelClass(ABC):

  def load_model(self):
    """Load the model."""
    pass

  def predict(self, **kwargs):
    """Predict method for single or batched inputs."""
    raise NotImplementedError("predict() not implemented")

  def generate(self, **kwargs) -> Iterator:
    """Generate method for streaming outputs."""
    raise NotImplementedError("generate() not implemented")

  def stream(self, **kwargs) -> Iterator:
    """Stream method for streaming inputs and outputs."""
    raise NotImplementedError("stream() not implemented")

  def _handle_get_signatures_request(self) -> service_pb2.MultiOutputResponse:
    methods = self._get_method_info()
    signatures = {method.name: method.signature for method in methods.values()}
    resp = service_pb2.MultiOutputResponse(status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    resp.outputs.add().data.string_value = signatures_to_json(signatures)
    return resp

  def batch_predict(self, method, inputs: List[Dict[str, Any]]) -> List[Any]:
    """Batch predict method for multiple inputs."""
    outputs = []
    for input in inputs:
      output = method(**input)
      outputs.append(output)
    return outputs

  def batch_generate(self, method, inputs: List[Dict[str, Any]]) -> Iterator[List[Any]]:
    """Batch generate method for multiple inputs."""
    generators = [method(**input) for input in inputs]
    for outputs in itertools.zip_longest(*generators):
      yield outputs

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = []
    try:
      # TODO add method name field to proto
      method_name = request.model.model_version.output_info.params['_method_name']
      if method_name == '_GET_SIGNATURES':  # special case to fetch signatures, TODO add endpoint for this
        return self._handle_get_signatures_request()
      if method_name not in self._get_method_info():
        raise ValueError(f"Method {method_name} not found in model class")
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types
      inputs = self._convert_input_protos_to_python(request.inputs, signature.inputs,
                                                    python_param_types)
      if len(inputs) == 1:
        inputs = inputs[0]
        output = method(**inputs)
        outputs.append(self._convert_output_to_proto(output, signature.outputs))
      else:
        outputs = self.batch_predict(method, inputs)
        outputs = [self._convert_output_to_proto(output, signature.outputs) for output in outputs]

      return service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    except Exception as e:
      if _RAISE_EXCEPTIONS:
        raise
      logging.exception("Error in predict")
      return service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.FAILURE,
          details=str(e),
          stack_trace=traceback.format_exc().split('\n')))

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    try:
      method_name = request.model.model_version.output_info.params['_method_name']
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types

      inputs = self._convert_input_protos_to_python(request.inputs, signature.inputs,
                                                    python_param_types)
      if len(inputs) == 1:
        inputs = inputs[0]
        for output in method(**inputs):
          resp = service_pb2.MultiOutputResponse()
          self._convert_output_to_proto(output, signature.outputs, proto=resp.outputs.add())
          resp.status.code = status_code_pb2.SUCCESS
          yield resp
      else:
        for outputs in self.batch_generate(method, inputs):
          resp = service_pb2.MultiOutputResponse()
          for output in outputs:
            self._convert_output_to_proto(output, signature.outputs, proto=resp.outputs.add())
          resp.status.code = status_code_pb2.SUCCESS
          yield resp
    except Exception as e:
      if _RAISE_EXCEPTIONS:
        raise
      logging.exception("Error in generate")
      yield service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.FAILURE,
          details=str(e),
          stack_trace=traceback.format_exc().split('\n')))

  def stream_wrapper(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    try:
      request = next(request_iterator)  # get first request to determine method
      assert len(request.inputs) == 1, "Streaming requires exactly one input"

      method_name = request.model.model_version.output_info.params['_method_name']
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types

      # find the streaming vars in the signature
      streaming_var_signatures = [var for var in signature.inputs if var.streaming]
      stream_argname = set([var.name.split('.', 1)[0] for var in streaming_var_signatures])
      assert len(
          stream_argname) == 1, 'streaming methods must have exactly one streaming function arg'
      stream_argname = stream_argname.pop()

      # convert all inputs for the first request, including the first stream value
      inputs = self._convert_input_protos_to_python(request.inputs, signature.inputs,
                                                    python_param_types)
      kwargs = inputs[0]

      # first streaming item
      first_item = kwargs.pop(stream_argname)

      # streaming generator
      def InputStream():
        yield first_item
        # subsequent streaming items contain only the streaming input
        for request in request_iterator:
          item = self._convert_input_protos_to_python(request.inputs, streaming_var_signatures,
                                                      python_param_types)
          item = item[0][stream_argname]
          yield item

      # add stream generator back to the input kwargs
      kwargs[stream_argname] = InputStream()

      for output in method(**kwargs):
        resp = service_pb2.MultiOutputResponse()
        self._convert_output_to_proto(output, signature.outputs, proto=resp.outputs.add())
        resp.status.code = status_code_pb2.SUCCESS
        yield resp
    except Exception as e:
      if _RAISE_EXCEPTIONS:
        raise
      logging.exception("Error in stream")
      yield service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.FAILURE,
          details=str(e),
          stack_trace=traceback.format_exc().split('\n')))

  def _convert_input_protos_to_python(self, inputs: List[resources_pb2.Input], variables_signature,
                                      python_param_types) -> List[Dict[str, Any]]:
    result = []
    for input in inputs:
      kwargs = deserialize(input.data, variables_signature)
      # dynamic cast to annotated types
      for k, v in kwargs.items():
        if k not in python_param_types:
          continue
        kwargs[k] = data_handler.cast(v, python_param_types[k])
      result.append(kwargs)
    return result

  def _convert_output_to_proto(self, output: Any, variables_signature,
                               proto=None) -> resources_pb2.Output:
    if proto is None:
      proto = resources_pb2.Output()
    if isinstance(output, tuple):
      output = {f'return.{i}': item for i, item in enumerate(output)}
    if not isinstance(output, dict):  # TODO Output type, not just dict
      output = {'return': output}
    serialize(output, variables_signature, proto.data, is_output=True)
    return proto

  @classmethod
  def _register_model_methods(cls):
    # go up the class hierarchy to find all decorated methods, and add to registry of current class
    methods = {}
    for base in reversed(cls.__mro__):
      for name, method in base.__dict__.items():
        method_info = getattr(method, _METHOD_INFO_ATTR, None)
        if not method_info:  # regular function, not a model method
          continue
        methods[name] = method_info
    # check for generic predict(request) -> response, etc. methods
    #for name in ('predict', 'generate', 'stream'):
    #  if hasattr(cls, name):
    #    method = getattr(cls, name)
    #    if not hasattr(method, _METHOD_INFO_ATTR):  # not already put in registry
    #      methods[name] = _MethodInfo(method, method_type=name)
    # set method table for this class in the registry
    return methods

  @classmethod
  def _get_method_info(cls, func_name=None):
    if not hasattr(cls, _METHOD_INFO_ATTR):
      setattr(cls, _METHOD_INFO_ATTR, cls._register_model_methods())
    method_info = getattr(cls, _METHOD_INFO_ATTR)
    if func_name:
      return method_info[func_name]
    return method_info


class _MethodInfo:

  def __init__(self, method, method_type):
    self.name = method.__name__
    self.signature = build_function_signature(method, method_type)
    self.python_param_types = {
        p.name: p.annotation
        for p in inspect.signature(method).parameters.values()
        if p.annotation != inspect.Parameter.empty
    }
    self.python_param_types.pop('self', None)


def predict(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'predict'))
  return method


def generate(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'generate'))
  return method


def stream(method):
  setattr(method, _METHOD_INFO_ATTR, _MethodInfo(method, 'stream'))
  return method


methods = types.SimpleNamespace(predict=predict, generate=generate, stream=stream)
