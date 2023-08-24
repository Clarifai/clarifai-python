from typing import Any, Callable, Dict, Generator

from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient


class Lister(BaseClient):
  """Lister class for obtaining paginated results from the Clarifai API."""

  def __init__(self, page_size: int = 16):
    self.default_page_size = page_size

  def list_all_pages_generator(
      self, endpoint: Callable, proto_message: Any,
      request_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Lists all pages of a resource.

    Args:
        endpoint (Callable): The endpoint to call.
        proto_message (Any): The proto message to use.
        request_data (dict): The request data to use.

    Yields:
        response_dict: The next item in the listing.
    """
    page = 1
    while True:
      request_data['page'] = page
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Listing failed with response {response!r}")
      if len(list(dict_response.keys())) == 1:
        break
      else:
        listing_resource = list(dict_response.keys())[1]
        for item in dict_response[listing_resource]:
          yield self.process_response_keys(item, listing_resource[:-1])
      page += 1
