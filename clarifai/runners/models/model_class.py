import functools
import inspect
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.runners.utils.data_handler import MessageData, Output
from clarifai.runners.utils.method_signatures import (build_function_signature, deserialize,
                                                      serialize, signatures_to_json)


class ModelClass(ABC):

  @abstractmethod
  def load_model(self):
    """Load the model."""
    raise NotImplementedError("load_model() not implemented")

  def predict(self, **kwargs) -> Output:
    """Predict method for single or batched inputs."""
    raise NotImplementedError("predict() not implemented")

  def generate(self, **kwargs) -> Iterator[Output]:
    """Generate method for streaming outputs."""
    raise NotImplementedError("generate() not implemented")

  def stream(self, **kwargs) -> Iterator[Output]:
    """Stream method for streaming inputs and outputs."""
    raise NotImplementedError("stream() not implemented")

  @functools.lru_cache(maxsize=None)
  def _method_signature(self, func):
    signature = build_function_signature(func)
    python_param_types = {
        p.name: p.annotation
        for p in inspect.signature(func).parameters.values()
        if p.annotation != inspect.Parameter.empty
    }
    python_param_types.pop('self', None)
    return signature, python_param_types

  def _handle_get_signatures_request(self) -> service_pb2.MultiOutputResponse:
    # TODO for now just predict
    signatures = {}
    for f in [self.predict]:
      sig, _ = self._method_signature(f)
      signatures[f.__name__] = sig
    resp = service_pb2.MultiOutputResponse(status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    resp.outputs.add().data.string_value = signatures_to_json(signatures)
    return resp

  def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Output]:
    """Batch predict method for multiple inputs."""
    outputs = []
    for input in inputs:
      output = self.predict(**input)
      outputs.append(output)
    return outputs

  def batch_generate(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Output]]:
    """Batch generate method for multiple inputs."""
    generators = [self.generate(**input) for input in inputs]
    for outputs in itertools.zip_longest(*generators):
      yield [output if output is not None else Output() for output in outputs]

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = []
    try:
      # TODO add method name field to proto
      method_name = request.model.model_version.output_info.params['_method_name']
      if method_name == '_GET_SIGNATURES':
        return self._handle_get_signatures_request()
      f = getattr(self, method_name)
      signature, python_param_types = self._method_signature(f)
      inputs = self._convert_input_protos_to_python(request.inputs, signature.inputs,
                                                    python_param_types)
      if len(inputs) == 1:
        inputs = inputs[0]
        output = self.predict(**inputs)
        outputs.append(self._convert_output_to_proto(output, signature.outputs))
      else:
        outputs = self.batch_predict(inputs)
        outputs = [self._convert_output_to_proto(output, signature.outputs) for output in outputs]
      return service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    except Exception as e:
      logging.exception("Error in predict")
      return service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.FAILURE, details=str(e)),)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    try:
      inputs_signature = self._method_signature(self.generate).inputs
      outputs_signature = self._method_signature(self.generate).outputs
      # TODO get inner type of stream iterator for outputs
      inputs = self._convert_input_protos_to_python(request.inputs, inputs_signature)
      if len(inputs) == 1:
        inputs = inputs[0]
        for output in self.generate(**inputs):
          resp = service_pb2.MultiOutputResponse()
          self._convert_output_to_proto(output, outputs_signature, proto=resp.outputs.add())
          resp.status = status_pb2.Status(code=status_code_pb2.SUCCESS)
          yield resp
      else:
        for outputs in self.batch_generate(inputs):
          resp = service_pb2.MultiOutputResponse()
          for output in outputs:
            self._convert_output_to_proto(output, outputs_signature, proto=resp.outputs.add())
          resp.status = status_pb2.Status(code=status_code_pb2.SUCCESS)
          yield resp
    except Exception as e:
      yield service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.FAILURE, details=str(e)),)

  def stream_wrapper(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    # TODO: Implement this with the new method signatures
    return self.stream(request_iterator)

  def _convert_input_protos_to_python(self, inputs: List[resources_pb2.Input], variables_signature,
                                      python_param_types) -> List[Dict[str, Any]]:
    result = []
    for input in inputs:
      kwargs = deserialize(input.data, variables_signature)
      kwargs = {
          k: v.cast(python_param_types[k]) if isinstance(v, MessageData) else v
          for k, v in kwargs.items()
      }
      result.append(kwargs)
    return result

  def _convert_output_to_proto(self, output: Any, variables_signature,
                               proto=None) -> resources_pb2.Output:
    if proto is None:
      proto = resources_pb2.Output()
    if not isinstance(output, dict):  # TODO Output type, not just dict
      output = {'return': output}
    serialize(output, variables_signature, proto.data)
    return proto
