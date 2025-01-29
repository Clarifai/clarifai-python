from abc import ABC, abstractmethod
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2


class ModelClass(ABC):

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This method is used for input/output proto data conversion"""
    return self.predict(request)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    return self.generate(request)

  def stream_wrapper(self, request: service_pb2.PostModelOutputsRequest
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    return self.stream(request)

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
