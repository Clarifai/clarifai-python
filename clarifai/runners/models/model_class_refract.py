import inspect
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, get_type_hints

from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.runners.utils.data_handler import Output, kwargs_to_proto, proto_to_kwargs


class ModelClass(ABC):

  @abstractmethod
  def load_model(self):
    raise NotImplementedError("load_model() not implemented")

  @abstractmethod
  def predict(self, *args, **kwargs) -> Output:
    raise NotImplementedError("predict() not implemented")

  @abstractmethod
  def generate(self, *args, **kwargs) -> Iterator[Output]:
    raise NotImplementedError("generate() not implemented")

  @abstractmethod
  def stream(self, *args, **kwargs) -> Iterator[Output]:
    raise NotImplementedError("stream() not implemented")

  def batch_predict(self, inputs: List[Dict[str, Any]]) -> List[Output]:
    with ThreadPoolExecutor() as executor:
      return list(executor.map(lambda x: self.predict(**x), inputs))

  def _process_request(self, request, process_func, is_stream=False):
    inputs = self._convert_proto_to_python(request.inputs)
    if len(inputs) == 1:
      result = process_func(**inputs[0])
      if is_stream:
        return (self._convert_output_to_proto(output) for output in result)
      else:
        return [self._convert_output_to_proto(result)]
    else:
      results = self.batch_predict(inputs) if not is_stream else []
      return [self._convert_output_to_proto(output) for output in results]

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    outputs = self._process_request(request, self.predict)
    return service_pb2.MultiOutputResponse(outputs=outputs)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    outputs = self._process_request(request, self.generate, is_stream=True)
    for output in outputs:
      yield service_pb2.MultiOutputResponse(outputs=[output])

  def stream_wrapper(self, requests: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    for request in requests:
      outputs = self._process_request(request, self.stream, is_stream=True)
      yield service_pb2.MultiOutputResponse(outputs=outputs)

  def _convert_proto_to_python(self, inputs: List[resources_pb2.Input]) -> List[Dict[str, Any]]:
    get_type_hints(self.predict)
    required_params = [
        name for name, param in inspect.signature(self.predict).parameters.items()
        if param.default == inspect.Parameter.empty
    ]
    kwargs_list = []
    for input_proto in inputs:
      kwargs = proto_to_kwargs(input_proto.data)
      missing = [name for name in required_params if name not in kwargs]
      if missing:
        raise ValueError(f"Missing required parameters: {missing}")
      kwargs_list.append(kwargs)
    return kwargs_list

  def _convert_output_to_proto(self, output: Any) -> resources_pb2.Output:
    if isinstance(output, Output):
      return output.to_proto()
    return kwargs_to_proto(**output).outputs.add()
