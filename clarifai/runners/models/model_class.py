from abc import ABC, abstractmethod
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2

from clarifai.runners.utils.url_fetcher import ensure_urls_downloaded
from clarifai.utils.stream_utils import readahead


class ModelClass(ABC):

  download_request_urls = True

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    """This method is used for input/output proto data conversion"""
    # Download any urls that are not already bytes.
    if self.download_request_urls:
      ensure_urls_downloaded(request)

    return self.predict(request)

  def generate_wrapper(self, request: service_pb2.PostModelOutputsRequest
                      ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""
    # Download any urls that are not already bytes.
    if self.download_request_urls:
      ensure_urls_downloaded(request)

    return self.generate(request)

  def stream_wrapper(self, request_stream: Iterator[service_pb2.PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    """This method is used for input/output proto data conversion and yield outcome"""

    # Download any urls that are not already bytes.
    if self.download_request_urls:
      request_stream = readahead(map(ensure_urls_downloaded, request_stream))

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
