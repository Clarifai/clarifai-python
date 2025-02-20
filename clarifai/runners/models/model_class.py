import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from PIL import Image as PILImage

from clarifai.runners.utils.data_handler import (Audio, Image, Output, Text, Video,
                                                 kwargs_to_proto, metadata_to_dict)


class ModelClass(ABC):

  @abstractmethod
  def load_model(self):
    """Load the model."""
    raise NotImplementedError("load_model() not implemented")

  @abstractmethod
  def predict(self, *args, **kwargs) -> Output:
    """Predict method for single or batched inputs."""
    raise NotImplementedError("predict() not implemented")

  @abstractmethod
  def generate(self, *args, **kwargs) -> Iterator[Output]:
    """Generate method for streaming outputs."""
    raise NotImplementedError("generate() not implemented")

  @abstractmethod
  def stream(self, *args, **kwargs) -> Iterator[Output]:
    """Stream method for streaming inputs and outputs."""
    raise NotImplementedError("stream() not implemented")

  def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Output]:
    """Batch predict method for multiple inputs."""
    outputs = []
    for input in inputs:
      output = self.predict(**input)
      outputs.append(output)
    return outputs

  def batch_generate(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Output]]:
    """Batch generate method for multiple inputs."""
    NotImplementedError(
        "batch_generate() not implemented, batching with generate() is not supported.")

  def batch_stream(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Output]]:
    """Batch stream method for multiple inputs."""
    NotImplementedError("batch_stream() not implemented")

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = []
    try:
      inputs = self._convert_proto_to_python(request.inputs, self.predict)
      if len(inputs) == 1:
        inputs = inputs[0]
        output = self.predict(**inputs)
        outputs.append(self._convert_output_to_proto(output))
      else:
        outputs = self.batch_predict(inputs)
        outputs = [self._convert_output_to_proto(output) for output in outputs]
      return service_pb2.MultiOutputResponse(
          outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))
    except Exception as e:
      return service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.FAILURE, details=str(e)),)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    try:
      inputs = self._convert_proto_to_python(request.inputs, self.generate)
      if len(inputs) == 1:
        inputs = inputs[0]
        for output in self.generate(**inputs):
          output_proto = self._convert_output_to_proto(output)
          yield service_pb2.MultiOutputResponse(outputs=[output_proto])
      else:
        outputs = []
        for output in self.batch_generate(inputs):
          output_proto = self._convert_output_to_proto(output)
          outputs.append(output_proto)
        yield service_pb2.MultiOutputResponse(outputs=outputs, status=status_code_pb2.SUCCESS)
    except Exception as e:
      yield service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.FAILURE, details=str(e)),)

  def stream_wrapper(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    try:
      for request in request_iterator:
        inputs = self._convert_proto_to_python(request.inputs, self.stream)
        if len(inputs) == 1:
          inputs = inputs[0]
          for output in self.stream(**inputs):
            output_proto = self._convert_output_to_proto(output)
            yield service_pb2.MultiOutputResponse(outputs=[output_proto])
        else:
          outputs = []
          for output in self.batch_stream(inputs):
            output_proto = self._convert_output_to_proto(output)
            outputs.append(output_proto)
          yield service_pb2.MultiOutputResponse(outputs=outputs)
    except Exception as e:
      yield service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.FAILURE, details=str(e)),)

  def _convert_proto_to_python(self, inputs: List[resources_pb2.Input],
                               method) -> List[Dict[str, Any]]:

    kwargs_list = []
    predict_params = inspect.signature(method).parameters
    for input in inputs:
      kwargs = {}
      for part in input.data.parts:
        param_name = part.id
        if param_name not in predict_params:
          raise ValueError(
              f"Unknown parameter: `{param_name}` in {method.__name__} method, available parameters: {predict_params.keys()}"
          )
        param = predict_params[param_name]
        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
          raise TypeError(
              f"Missing type annotation for parameter: {param_name} in {method.__name__} method, available types: {predict_params.values()}"
          )
        value = self._convert_part_data(part.data, param_type)
        kwargs[param_name] = value

      # Check for missing required parameters
      self._validate_required_params(predict_params, kwargs)
      kwargs_list.append(kwargs)
    return kwargs_list

  def _convert_part_data(self, data: resources_pb2.Data, param_type: type) -> Any:
    if param_type == str:
      if not data.HasField("text"):
        raise ValueError('expected str datatype but the provided input is not a str')
      return data.text.raw
    elif param_type == int:
      return data.int_value
    elif param_type == float:
      return data.float_value
    elif param_type == bool:
      return data.bool_value
    elif param_type == bytes:
      return data.bytes_value
    elif param_type == np.ndarray:
      if not data.HasField("ndarray"):
        raise ValueError(
            'expected numpy.ndarray datatype but the provided input is not a numpy.ndarray')
      return np.frombuffer(
          data.ndarray.buffer, dtype=np.dtype(data.ndarray.dtype)).reshape(data.ndarray.shape)
    elif param_type == PILImage.Image:
      if not data.HasField("image"):
        raise ValueError('expected PIL.Image datatype but the provided input is not a PIL.Image')
      return Image.from_proto(data.image).to_pil()
    elif param_type == Text:
      if not data.HasField("text"):
        raise ValueError('expected Text datatype but the provided input is not a Text')
      return Text.from_proto(data.text)
    elif param_type == Image:
      if not data.HasField("image"):
        raise ValueError('expected Image datatype but the provided input is not a Image')
      return Image.from_proto(data.image)
    elif param_type == Audio:
      if not data.HasField("audio"):
        raise ValueError('expected Audio datatype but the provided input is not a Audio')
      return Audio.from_proto(data.audio)
    elif param_type == Video:
      if not data.HasField("video"):
        raise ValueError('expected Video datatype but the provided input is not a Video')
      return Video.from_proto(data.video)
    elif param_type == Any:
      raise ValueError("Any type is not supported in input parameters")
    elif param_type == List:
      list_output = []
      for part in data.parts:
        if part.data.HasField("text"):
          list_output.append(Text.from_proto(part.data.text))
        elif part.data.HasField("image"):
          list_output.append(Image(part.data.image))
        elif part.data.HasField("audio"):
          list_output.append(Audio(part.data.audio))
        elif part.data.HasField("video"):
          list_output.append(Video(part.data.video))
        elif part.data.bytes_value != b'':
          list_output.append(part.data.bytes_value)
        elif part.data.int_value != 0:
          list_output.append(part.data.int_value)
        elif part.data.float_value != 0.0:
          list_output.append(part.data.float_value)
        elif part.data.bool_value is not False:
          list_output.append(part.data.bool_value)
        elif part.data.HasField("ndarray"):
          ndarray = part.data.ndarray
          list_output.append(
              np.frombuffer(ndarray.buffer, dtype=np.dtype(ndarray.dtype)).reshape(ndarray.shape))
        elif part.data.HasField("metadata"):
          raise ValueError("Metadata in list is not supported")
      return list_output
    elif param_type == Dict:
      return metadata_to_dict(data)
    else:
      raise ValueError(f"Unsupported input type: {param_type}")

  def _validate_required_params(self, params: dict, kwargs: dict):
    for name, param in params.items():
      if param.default == inspect.Parameter.empty and name not in kwargs:
        raise ValueError(f"Missing required parameter: {name}")

  def _convert_output_to_proto(self, output: Any) -> resources_pb2.Output:
    if isinstance(output, Output):
      return output.to_proto()
    else:
      # Handle basic types (legacy support)
      data_proto = kwargs_to_proto(output)
      return resources_pb2.Output(data=data_proto)
