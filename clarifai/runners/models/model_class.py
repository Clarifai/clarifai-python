from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.runners.utils.data_handler import Audio, Image, Text, Video


class ModelClass(ABC):

  @abstractmethod
  def load_model(self):
    """Load the model."""
    raise NotImplementedError("load_model() not implemented")

  @abstractmethod
  def predict(self, *args, **kwargs) -> Union[Any, List[Any]]:
    """Predict method for single or batched inputs."""
    raise NotImplementedError("predict() not implemented")

  @abstractmethod
  def generate(self, *args, **kwargs) -> Iterator[Any]:
    """Generate method for streaming outputs."""
    raise NotImplementedError("generate() not implemented")

  @abstractmethod
  def stream(self, *args, **kwargs) -> Iterator[Any]:
    """Stream method for streaming inputs and outputs."""
    raise NotImplementedError("stream() not implemented")

  def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch predict method for multiple inputs."""
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(self.predict, **input) for input in inputs]
      return [future.result() for future in futures]

  def batch_generate(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    """Batch generate method for multiple inputs."""
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(self.generate, **input) for input in inputs]
      return [future.result() for future in futures]

  def batch_stream(self, inputs: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    """Batch stream method for multiple inputs."""
    NotImplementedError("batch_stream() not implemented")

  def _convert_to_proto(self, arg: Any) -> resources_pb2.Data:
    """Convert Python types to Clarifai protobuf Data."""
    if isinstance(arg, Text):
      return resources_pb2.Data(text=arg.to_proto())
    elif isinstance(arg, Image):
      return resources_pb2.Data(image=arg.to_proto())
    elif isinstance(arg, Audio):
      return resources_pb2.Data(audio=arg.to_proto())
    elif isinstance(arg, Video):
      return resources_pb2.Data(video=arg.to_proto())
    else:
      raise ValueError(f"Unknown type: {type(arg)}")

  def _create_input_proto(self, *args, **kwargs) -> resources_pb2.Input:
    """Create a Clarifai Input protobuf from Python args and kwargs."""
    data = resources_pb2.Data()
    for arg in args:
      part = resources_pb2.Part(data=self._convert_to_proto(arg))
      data.parts.append(part)
    for key, value in kwargs.items():
      part = resources_pb2.Part(data=self._convert_to_proto(value), id=key)
      data.parts.append(part)
    return resources_pb2.Input(data=data)

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    inputs = self._convert_proto_to_python(request.inputs[0])
    if len(inputs) == 1:
      inputs = inputs[0]
      output = self.predict(**inputs)
      output = [output]
    else:
      output = self.batch_predict(inputs)
    return self._convert_python_to_proto(output)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    inputs = self._convert_proto_to_python(request.inputs[0])
    if len(inputs) == 1:
      inputs = inputs[0]
      for output in self.generate(**inputs):
        yield self._convert_python_to_proto([output])
    else:
      for output in self.batch_generate(inputs):
        yield self._convert_python_to_proto(output)

  def stream_wrapper(self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    for request in request_iterator:
      inputs = self._convert_proto_to_python(request.inputs[0])
      if len(inputs) == 1:
        inputs = inputs[0]
        for output in self.stream(**inputs):
          yield self._convert_python_to_proto([output])
      else:
        for output in self.batch_stream(inputs):
          yield self._convert_python_to_proto(output)

  def _convert_proto_to_python(self, inputs: List[resources_pb2.Input]) -> List[Dict[str, Any]]:

    python_list = []
    for input in inputs:
      python_inputs = {}
      for part in input.data.parts:
        if part.data.HasField("text"):
          python_inputs[part.id] = Text.from_proto(part.data.text)
        elif part.data.HasField("image"):
          python_inputs[part.id] = Image(part.data.image)
        elif part.data.HasField("audio"):
          python_inputs[part.id] = Audio(part.data.audio)
        elif part.data.HasField("video"):
          python_inputs[part.id] = Video(part.data.video)
      python_list.append(python_inputs)
    return python_list

  def _convert_python_to_proto(self,
                               outputs: List[Dict[str, Any]]) -> service_pb2.MultiOutputResponse:
    response = service_pb2.MultiOutputResponse()
    for output in outputs:
      output_proto = resources_pb2.Output()
      for key, value in output.items():
        part = resources_pb2.Part(id=key)
        if isinstance(value, Text):
          part.data.text.CopyFrom(value.to_proto())
        elif isinstance(value, Image):
          part.data.image.CopyFrom(value.to_proto())
        elif isinstance(value, Audio):
          part.data.audio.CopyFrom(value.to_proto())
        elif isinstance(value, Video):
          part.data.video.CopyFrom(value.to_proto())
        output_proto.data.parts.append(part)
      response.outputs.append(output_proto)
    return response
