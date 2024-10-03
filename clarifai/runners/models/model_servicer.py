from itertools import tee
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from ..utils.url_fetcher import ensure_urls_downloaded


class ModelServicer(service_pb2_grpc.V2Servicer):
  """
  This is the servicer that will handle the gRPC requests from either the dev server or runner loop.
  """

  def __init__(self, model_class):
    """
    Args:
        model_class: The class that will handle the model logic. Must implement predict(),
    generate(), stream().
    """
    self.model_class = model_class

  def PostModelOutputs(self, request: service_pb2.PostModelOutputsRequest,
                       context=None) -> service_pb2.MultiOutputResponse:
    """
    This is the method that will be called when the servicer is run. It takes in an input and
    returns an output.
    """

    # Download any urls that are not already bytes.
    ensure_urls_downloaded(self.url_fetcher, request)

    try:
      return self.model_class.predict(request)
    except Exception as e:
      return service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_PREDICTION_FAILED,
          description="Failed",
          details="",
          internal_details=str(e),
      ))

  def GenerateModelOutputs(self, request: service_pb2.PostModelOutputsRequest,
                           context=None) -> Iterator[service_pb2.MultiOutputResponse]:
    """
    This is the method that will be called when the servicer is run. It takes in an input and
    returns an output.
    """
    # Download any urls that are not already bytes.
    ensure_urls_downloaded(self.url_fetcher, request)

    try:
      return self.model_class.generate(request)
    except Exception as e:
      yield service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_PREDICTION_FAILED,
          description="Failed",
          details="",
          internal_details=str(e),
      ))

  def StreamModelOutputs(self,
                         request: Iterator[service_pb2.PostModelOutputsRequest],
                         context=None) -> Iterator[service_pb2.MultiOutputResponse]:
    """
    This is the method that will be called when the servicer is run. It takes in an input and
    returns an output.
    """
    # Duplicate the iterator
    request, request_copy = tee(request)

    # Download any urls that are not already bytes.
    for req in request:
      ensure_urls_downloaded(self.url_fetcher, req)

    try:
      return self.model_class.stream(request_copy)
    except Exception as e:
      yield service_pb2.MultiOutputResponse(status=status_pb2.Status(
          code=status_code_pb2.MODEL_PREDICTION_FAILED,
          description="Failed",
          details="",
          internal_details=str(e),
      ))
