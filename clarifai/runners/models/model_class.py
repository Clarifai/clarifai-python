import inspect
import itertools
import logging
import os
import traceback
from abc import ABC
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format

from clarifai.runners.utils import data_types
from clarifai.runners.utils.method_signatures import (build_function_signature, deserialize,
                                                      get_stream_from_signature, serialize,
                                                      signatures_to_json)

_METHOD_INFO_ATTR = '_cf_method_info'

_RAISE_EXCEPTIONS = os.getenv("RAISE_EXCEPTIONS", "false").lower() in ("true", "1")


class ModelClass(ABC):
  '''
  Base class for model classes that can be run as a service.

  Define predict, generate, or stream methods using the @ModelClass.method decorator.

  Example:

    from clarifai.runners.model_class import ModelClass
    from clarifai.runners.utils.data_types import NamedFields, Stream

    class MyModel(ModelClass):

      @ModelClass.method
      def predict(self, x: str, y: int) -> List[str]:
        return [x] * y

      @ModelClass.method
      def generate(self, x: str, y: int) -> Stream[str]:
        for i in range(y):
          yield x + str(i)

      @ModelClass.method
      def stream(self, input_stream: Stream[NamedFields(x=str, y=int)]) -> Stream[str]:
        for item in input_stream:
          yield item.x + ' ' + str(item.y)
  '''

  @staticmethod
  def method(func):
    setattr(func, _METHOD_INFO_ATTR, _MethodInfo(func))
    return func

  def set_output_context(self, prompt_tokens=None, completion_tokens=None):
    """This is used to set the prompt and completion tokens in the Output proto"""
    self._prompt_tokens = prompt_tokens
    self._completion_tokens = completion_tokens

  def load_model(self):
    """Load the model."""

  def _handle_get_signatures_request(self) -> service_pb2.MultiOutputResponse:
    methods = self._get_method_info()
    signatures = {method.name: method.signature for method in methods.values()}
    resp = service_pb2.MultiOutputResponse(status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    output = resp.outputs.add()
    output.status.code = status_code_pb2.SUCCESS
    output.data.text.raw = signatures_to_json(signatures)
    return resp

  def _is_data_set(self, data_msg):
    # Singular message fields
    singular_fields = ["image", "video", "metadata", "geo", "text", "audio", "ndarray"]
    for field in singular_fields:
      if data_msg.HasField(field):
        return True

    # Repeated fields
    repeated_fields = [
        "concepts", "colors", "clusters", "embeddings", "regions", "frames", "tracks",
        "time_segments", "hits", "heatmaps", "parts"
    ]
    for field in repeated_fields:
      if getattr(data_msg, field):  # checks if the list is not empty
        return True

    # Scalar fields (proto3 default: 0 for numbers, empty for strings/bytes, False for bool)
    if (data_msg.int_value != 0 or data_msg.float_value != 0.0 or data_msg.bytes_value != b"" or
        data_msg.bool_value is True or data_msg.string_value != ""):
      return True

    return False

  def _convert_input_data_to_new_format(
      self, data: resources_pb2.Data,
      input_fields: List[resources_pb2.ModelTypeField]) -> resources_pb2.Data:
    """Convert input data to new format."""
    new_data = resources_pb2.Data()
    for field in input_fields:
      part_data = self._convert_field(data, field)
      if self._is_data_set(part_data):
        # if the field is set, add it to the new data part
        part = new_data.parts.add()
        part.id = field.name
        part.data.CopyFrom(part_data)
      else:
        if field.required:
          raise ValueError(f"Field {field.name} is required but not set")
    return new_data

  def _convert_field(self, old_data: resources_pb2.Data,
                     field: resources_pb2.ModelTypeField) -> resources_pb2.Data:
    data_type = field.type
    new_data = resources_pb2.Data()
    if data_type == resources_pb2.ModelTypeField.DataType.STR:
      if old_data.HasField('text'):
        new_data.string_value = old_data.text.raw
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.IMAGE:
      if old_data.HasField('image'):
        new_data.image.CopyFrom(old_data.image)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.VIDEO:
      if old_data.HasField('video'):
        new_data.video.CopyFrom(old_data.video)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.BOOL:
      if old_data.HasField('bool_value'):
        new_data.bool_value = old_data.bool_value
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.INT:
      if old_data.HasField('int_value'):
        new_data.int_value = old_data.int_value
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.FLOAT:
      if old_data.HasField('float_value'):
        new_data.float_value = old_data.float_value
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.BYTES:
      if old_data.HasField('bytes_value'):
        new_data.bytes_value = old_data.bytes_value
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.NDARRAY:
      if old_data.HasField('ndarray'):
        new_data.ndarray.CopyFrom(old_data.ndarray)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.TEXT:
      if old_data.HasField('text'):
        new_data.text.CopyFrom(old_data.text)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.AUDIO:
      if old_data.HasField('audio'):
        new_data.audio.CopyFrom(old_data.audio)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.CONCEPT:
      if old_data.concepts:
        new_data.concepts.extend(old_data.concepts)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.REGION:
      if old_data.regions:
        new_data.regions.extend(old_data.regions)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.FRAME:
      if old_data.frames:
        new_data.frames.extend(old_data.frames)
      return new_data
    elif data_type == resources_pb2.ModelTypeField.DataType.LIST:
      new_data = resources_pb2.Data()
      if not field.type_args:
        raise ValueError("LIST type requires type_args")
      element_field = field.type_args[0]
      if element_field in (resources_pb2.ModelTypeField.DataType.CONCEPT,
                           resources_pb2.ModelTypeField.DataType.REGION,
                           resources_pb2.ModelTypeField.DataType.FRAME):
        # convert to new format
        element_data = self._convert_field(old_data, element_field)
        # part = new_data.parts.add()
        # part.data.CopyFrom(element_data)
      return element_data
    else:
      return new_data
      # raise ValueError(f"Unsupported data type: {data_type}")

  def is_old_format(self, data: resources_pb2.Data) -> bool:
    """Check if the Data proto is in the old format (without parts)."""
    if len(data.parts) > 0:
      return False  # New format uses parts

    # Check if any singular field is set
    singular_fields = [
        'image', 'video', 'metadata', 'geo', 'text', 'audio', 'ndarray', 'int_value',
        'float_value', 'bytes_value', 'bool_value', 'string_value'
    ]
    for field in singular_fields:
      if data.HasField(field):
        return True

    # Check if any repeated field has elements
    repeated_fields = [
        'concepts', 'colors', 'clusters', 'embeddings', 'regions', 'frames', 'tracks',
        'time_segments', 'hits', 'heatmaps'
    ]
    for field in repeated_fields:
      if getattr(data, field):
        return True

    return False

  def _convert_output_data_to_old_format(self, data: resources_pb2.Data) -> resources_pb2.Data:
    """Convert output data to old format."""
    old_data = resources_pb2.Data()
    part_data = data.parts[0].data
    # Handle text.raw specially (common case for text outputs)
    old_data = part_data
    if old_data.string_value:
      old_data.text.raw = old_data.string_value

    return old_data

  def _batch_predict(self, method, inputs: List[Dict[str, Any]]) -> List[Any]:
    """Batch predict method for multiple inputs."""
    outputs = []
    for input in inputs:
      output = method(**input)
      outputs.append(output)
    return outputs

  def _batch_generate(self, method, inputs: List[Dict[str, Any]]) -> Iterator[List[Any]]:
    """Batch generate method for multiple inputs."""
    generators = [method(**input) for input in inputs]
    for outputs in itertools.zip_longest(*generators):
      yield outputs

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = []
    try:
      # TODO add method name field to proto
      method_name = 'predict'
      inference_params = get_inference_params(request)
      if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
        method_name = request.inputs[0].data.metadata['_method_name']
      if method_name == '_GET_SIGNATURES':  # special case to fetch signatures, TODO add endpoint for this
        return self._handle_get_signatures_request()
      if method_name not in self._get_method_info():
        raise ValueError(f"Method {method_name} not found in model class")
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types
      for input in request.inputs:
        # check if input is in old format
        is_convert = self.is_old_format(input.data)
        if is_convert:
          # convert to new format
          new_data = self._convert_input_data_to_new_format(input.data, signature.input_fields)
          input.data.CopyFrom(new_data)
      # convert inputs to python types
      inputs = self._convert_input_protos_to_python(request.inputs, inference_params,
                                                    signature.input_fields, python_param_types)
      if len(inputs) == 1:
        inputs = inputs[0]
        output = method(**inputs)
        outputs.append(
            self._convert_output_to_proto(
                output, signature.output_fields, convert_old_format=is_convert))
      else:
        outputs = self._batch_predict(method, inputs)
        outputs = [
            self._convert_output_to_proto(
                output, signature.output_fields, convert_old_format=is_convert)
            for output in outputs
        ]

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
      method_name = 'generate'
      inference_params = get_inference_params(request)
      if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
        method_name = request.inputs[0].data.metadata['_method_name']
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types
      for input in request.inputs:
        # check if input is in old format
        is_convert = self.is_old_format(input.data)
        if is_convert:
          # convert to new format
          new_data = self._convert_input_data_to_new_format(input.data, signature.input_fields)
          input.data.CopyFrom(new_data)
      inputs = self._convert_input_protos_to_python(request.inputs, inference_params,
                                                    signature.input_fields, python_param_types)
      if len(inputs) == 1:
        inputs = inputs[0]
        for output in method(**inputs):
          resp = service_pb2.MultiOutputResponse()
          self._convert_output_to_proto(
              output,
              signature.output_fields,
              proto=resp.outputs.add(),
              convert_old_format=is_convert)
          resp.status.code = status_code_pb2.SUCCESS
          yield resp
      else:
        for outputs in self._batch_generate(method, inputs):
          resp = service_pb2.MultiOutputResponse()
          for output in outputs:
            self._convert_output_to_proto(
                output,
                signature.output_fields,
                proto=resp.outputs.add(),
                convert_old_format=is_convert)
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

      method_name = 'stream'
      inference_params = get_inference_params(request)
      if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
        method_name = request.inputs[0].data.metadata['_method_name']
      method = getattr(self, method_name)
      method_info = method._cf_method_info
      signature = method_info.signature
      python_param_types = method_info.python_param_types

      # find the streaming vars in the signature
      stream_sig = get_stream_from_signature(signature.input_fields)
      if stream_sig is None:
        raise ValueError("Streaming method must have a Stream input")
      stream_argname = stream_sig.name

      for input in request.inputs:
        # check if input is in old format
        is_convert = self.is_old_format(input.data)
        if is_convert:
          # convert to new format
          new_data = self._convert_input_data_to_new_format(input.data, signature.input_fields)
          input.data.CopyFrom(new_data)
      # convert all inputs for the first request, including the first stream value
      inputs = self._convert_input_protos_to_python(request.inputs, inference_params,
                                                    signature.input_fields, python_param_types)
      kwargs = inputs[0]

      # first streaming item
      first_item = kwargs.pop(stream_argname)

      # streaming generator
      def InputStream():
        yield first_item
        # subsequent streaming items contain only the streaming input
        for request in request_iterator:
          item = self._convert_input_protos_to_python(request.inputs, inference_params,
                                                      [stream_sig], python_param_types)
          item = item[0][stream_argname]
          yield item

      # add stream generator back to the input kwargs
      kwargs[stream_argname] = InputStream()

      for output in method(**kwargs):
        resp = service_pb2.MultiOutputResponse()
        self._convert_output_to_proto(
            output,
            signature.output_fields,
            proto=resp.outputs.add(),
            convert_old_format=is_convert)
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

  def _convert_input_protos_to_python(self, inputs: List[resources_pb2.Input],
                                      inference_params: dict,
                                      variables_signature: List[resources_pb2.ModelTypeField],
                                      python_param_types) -> List[Dict[str, Any]]:
    result = []
    for input in inputs:
      kwargs = deserialize(input.data, variables_signature, inference_params)
      # dynamic cast to annotated types
      for k, v in kwargs.items():
        if k not in python_param_types:
          continue

        if hasattr(python_param_types[k], "__args__") and getattr(
            python_param_types[k], "__origin__", None) == data_types.Stream:
          # get the type of the items in the stream
          stream_type = python_param_types[k].__args__[0]

          kwargs[k] = data_types.cast(v, stream_type)
        else:
          kwargs[k] = data_types.cast(v, python_param_types[k])
      result.append(kwargs)
    return result

  def _convert_output_to_proto(self,
                               output: Any,
                               variables_signature: List[resources_pb2.ModelTypeField],
                               proto=None,
                               convert_old_format=False) -> resources_pb2.Output:
    if proto is None:
      proto = resources_pb2.Output()
    serialize({'return': output}, variables_signature, proto.data, is_output=True)
    if convert_old_format:
      # convert to old format
      data = self._convert_output_data_to_old_format(proto.data)
      proto.data.CopyFrom(data)
    proto.status.code = status_code_pb2.SUCCESS
    if hasattr(self, "_prompt_tokens") and self._prompt_tokens is not None:
      proto.prompt_tokens = self._prompt_tokens
    if hasattr(self, "_completion_tokens") and self._completion_tokens is not None:
      proto.completion_tokens = self._completion_tokens
    self._prompt_tokens = None
    self._completion_tokens = None
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
    #      methods[name] = _MethodInfo(method)
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


# Helper function to get the inference params
def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  if request.model.model_version.id != "":
    output_info = request.model.model_version.output_info
    output_info = json_format.MessageToDict(output_info, preserving_proto_field_name=True)
    if "params" in output_info:
      inference_params = output_info["params"]
  return inference_params


class _MethodInfo:

  def __init__(self, method):
    self.name = method.__name__
    self.signature = build_function_signature(method)
    self.python_param_types = {
        p.name: p.annotation
        for p in inspect.signature(method).parameters.values()
        if p.annotation != inspect.Parameter.empty
    }
    self.python_param_types.pop('self', None)
