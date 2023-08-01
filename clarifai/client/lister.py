from datetime import datetime
from typing import Any, Callable, Dict, Generator

from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

from clarifai.client.base_client import BaseClient


class Lister(BaseClient):

  def __init__(self, page_size: int = 16):
    self.default_page_size = page_size

  def convert_string_to_timestamp(self, date_str) -> Timestamp:
    """Converts a string to a Timestamp object.
    Args:
        date_str (str): The string to convert.
    Returns:
        Timestamp: The converted Timestamp object.
    """
    # Parse the string into a Python datetime object
    try:
      datetime_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
      datetime_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')

    # Convert the datetime object to a Timestamp object
    timestamp_obj = Timestamp()
    timestamp_obj.FromDatetime(datetime_obj)

    return timestamp_obj

  def convert_keys_to_snake_case(self, old_dict: Dict[str, Any],
                                 listing_resource: str) -> Dict[str, Any]:
    """Converts keys in a dictionary to snake case.
    Args:
        old_dict (dict): The dictionary to convert.
    Returns:
        new_dict (dict): The dictionary with snake case keys.
    """
    old_dict[f'{listing_resource}_id'] = old_dict['id']

    def snake_case(key):
      result = ''
      for char in key:
        if char.isupper():
          result += '_' + char.lower()
        else:
          result += char
      return result

    new_dict = {}
    for key, value in old_dict.items():
      if key in ['createdAt', 'modifiedAt']:
        value = self.convert_string_to_timestamp(value)
      if key == 'metadata':
        continue  #TODO Fix "app_duplication" error
      new_key = snake_case(key)
      new_dict[new_key] = value
    return new_dict

  def list_all_pages_generator(
      self, endpoint: Callable, proto_message: Any,
      request_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Lists all pages of a resource.
    Args:
        endpoint (Callable): The endpoint to call.
        proto_message (Any): The proto message to use.
        request_data (dict): The request data to use.
    Yields:
        dict: The next item in the listing.
    """
    page = 1
    while True:
      request_data['page'] = page
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Listing failed with response {response!r}")
      if len(list(dict_response.keys())) == 1:
        break
      else:
        listing_resource = list(dict_response.keys())[1]
        for item in dict_response[listing_resource]:
          yield self.convert_keys_to_snake_case(item, listing_resource[:-1])
      page += 1
