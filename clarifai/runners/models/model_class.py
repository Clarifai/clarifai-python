import inspect
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List
from PIL import Image as PILImage

from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.runners.utils.data_handler import Audio, Image, Output, Text, Video
 

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
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(self.predict, **input) for input in inputs]
      return [future.result() for future in futures]

  def batch_generate(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Output]]:
    """Batch generate method for multiple inputs."""
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(self.generate, **input) for input in inputs]
      return [future.result() for future in futures]

  def batch_stream(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Output]]:
    """Batch stream method for multiple inputs."""
    NotImplementedError("batch_stream() not implemented")

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = []
    inputs = self._convert_proto_to_python(request.inputs)
    if len(inputs) == 1:
      inputs = inputs[0]
      output = self.predict(**inputs)
      outputs.append(self._convert_output_to_proto(output))
    else:
      outputs = self.batch_predict(inputs)
      outputs = [self._convert_output_to_proto(output) for output in outputs]
    return service_pb2.MultiOutputResponse(outputs=outputs)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    inputs = self._convert_proto_to_python(request.inputs)
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
      yield service_pb2.MultiOutputResponse(outputs=outputs)

  def stream_wrapper(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    for request in request_iterator:
      inputs = self._convert_proto_to_python(request.inputs)
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

  def _convert_proto_to_python(self, inputs: List[resources_pb2.Input]) -> List[Dict[str, Any]]:

    kwargs_list = []
    predict_params = inspect.signature(self.predict).parameters
    for input in inputs:
      kwargs = {}
      for part in input.data.parts:
        param_name = part.id
        if param_name not in predict_params:
          raise ValueError(f"Unknown parameter: {param_name}")
        param = predict_params[param_name]
        param_type = param.annotation
        if param_type is inspect.Parameter.empty:
          raise TypeError(f"Missing type annotation for parameter: {param_name}")
        value = self._convert_part_data(part.data, param_type)
        kwargs[param_name] = value

      # Check for missing required parameters
      self._validate_required_params(predict_params, kwargs)
      kwargs_list.append(kwargs)
    return kwargs_list

  def _convert_part_data(self, data: resources_pb2.Data, param_type: type) -> Any:
    if param_type == str:
      return data.text.value
    elif param_type == int:
      return data.int_value
    elif param_type == float:
      return data.float_value
    elif param_type == bool:
      return data.boolean
    elif param_type == bytes:
      return data.base64
    elif param_type == np.ndarray:
      return np.frombuffer(data.ndarray.buffer, dtype=np.dtype(data.ndarray.dtype)).reshape(data.ndarray.shape)
    elif param_type == PILImage.Image:
      return Image.from_proto(data.image).to_pil()
    elif param_type == Text:
      return Text.from_proto(data.text)
    elif param_type == Image:
      return Image.from_proto(data.image)
    elif param_type == Audio:
      return Audio.from_proto(data.audio)
    elif param_type == Video:
      return Video.from_proto(data.video)
    elif param_type == Any:
      if data.text.value:
        return data.text.value
      elif data.image.url:
        return data.image.url
      elif data.audio.url:
        return data.audio.url
      elif data.video.url:
        return data.video.url
      elif data.int_value:
        return data.int_value
      elif data.float_value:
        return data.int_value
      elif data.boolean:
        return data.int_value
      elif data.base64:
        return data.base64
      else:
        raise ValueError(f"Unknown type: {data}")
    elif param_type == List[str]:
      return [part.data.text.value for part in data.parts]
    elif param_type == List[int]:
      return [part.data.int_value for part in data.parts]
    elif param_type == List[float]:
      return [part.data.float_value for part in data.parts]
    elif param_type == List[bool]:
      return [part.data.boolean for part in data.parts]
    elif param_type == List[PILImage.Image]:
      return [Image.from_proto(part.data.image).to_pil() for part in data.parts]
    elif param_type == List[PILImage.Image]:
      return [np.frombuffer(part.data.ndarray.buffer, dtype=np.dtype(part.data.ndarray.dtype)).reshape(part.data.ndarray.shape) for part in data.parts]
    elif param_type == List[Text]:
      return [Text.from_proto(part.data.text) for part in data.parts]
    elif param_type == List[Image]:
      return [Image.from_proto(part.data.image) for part in data.parts]
    elif param_type == List[Audio]:
      return [Audio.from_proto(part.data.audio) for part in data.parts]
    elif param_type == List[Video]:
      return [Video.from_proto(part.data.video) for part in data.parts]
    elif param_type == List:
      return [self._convert_part_data(part.data, Any) for part in data.parts]
    elif param_type == Dict[str, Text]:
      return {part.id: Text.from_proto(part.data.text) for part in data.parts}
    elif param_type == Dict[str, str]:
      return {part.id: part.data.text.value for part in data.parts}
    elif param_type == Dict[str, Image]:
      return {part.id: Image.from_proto(part.data.image) for part in data.parts}
    elif param_type == Dict[str, Audio]:
      return {part.id: Audio.from_proto(part.data.audio) for part in data.parts}
    elif param_type == Dict[str, Video]:
      return {part.id: Video.from_proto(part.data.video) for part in data.parts}
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
      data = resources_pb2.Data()
      if isinstance(output, str):
        data.text.raw = output
      elif isinstance(output, bytes):
        data.base64 = output
      elif isinstance(output, int):
        data.int_value = output
      elif isinstance(output, float):
        data.float_value = output
      elif isinstance(output, bool):
        data.boolean = output
      elif isinstance(output, Text):
        data.text.raw = output.text
      elif isinstance(output, Image):
        data.image.CopyFrom(output.to_proto())
      elif isinstance(output, Audio):
        data.audio.CopyFrom(output.to_proto())
      elif isinstance(output, Video):
        data.video.CopyFrom(output.to_proto())
      elif isinstance(output, np.ndarray):
        ndarray_proto = resources_pb2.NDArray(
            buffer=output.tobytes(), shape=output.shape, dtype=str(output.dtype))
        data.ndarray.CopyFrom(ndarray_proto)
      elif isinstance(output, PILImage.Image):
        image = Image.from_pil(output)
        data.image.CopyFrom(image.to_proto())
      else:
        raise ValueError(f"Unsupported output type: {type(output)}")
      return resources_pb2.Output(data=data)
