from typing import Any, Callable, Dict, Generator

from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient


class Lister(BaseClient):
  """Lister class for obtaining paginated results from the Clarifai API."""

  def __init__(self, page_size: int = 16):
    self.default_page_size = page_size

  def list_pages_generator(self,
                           endpoint: Callable,
                           proto_message: Any,
                           request_data: Dict[str, Any],
                           page_no: int = None,
                           per_page: int = None) -> Generator[Dict[str, Any], None, None]:
    """Lists pages of a resource.

    Args:
        endpoint (Callable): The endpoint to call.
        proto_message (Any): The proto message to use.
        request_data (dict): The request data to use.
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        response_dict: The next item in the listing.
    """
    page = 1 if not page_no else page_no
    if page_no and not per_page:
      per_page = self.default_page_size
    while True:
      request_data['page'] = page
      request_data['per_page'] = per_page
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Listing failed with response {response!r}")
      if len(list(dict_response.keys())) == 1:
        break
      else:
        listing_resource = list(dict_response.keys())[1]
        for item in dict_response[listing_resource]:
          if listing_resource == "dataset_inputs":
            yield self.process_response_keys(item["input"], listing_resource[:-1])
          else:
            yield self.process_response_keys(item, listing_resource[:-1])
      if page_no is not None or per_page is not None:
        break
      page += 1
