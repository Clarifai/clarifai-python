from math import ceil
from typing import Any, Callable, Dict, Generator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from schema import SchemaError

from clarifai.client.base import BaseClient
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.constants.search import (DEFAULT_SEARCH_ALGORITHM, DEFAULT_SEARCH_METRIC,
                                       DEFAULT_TOP_K)
from clarifai.errors import UserError
from clarifai.schema.search import get_schema


class Search(Lister, BaseClient):

  def __init__(self,
               user_id: str,
               app_id: str,
               top_k: int = None,
               metric: str = DEFAULT_SEARCH_METRIC,
               algorithm: str = DEFAULT_SEARCH_ALGORITHM,
               pagination: bool = False,
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None):
    """Initialize the Search object.

    Args:
        user_id (str): User ID.
        app_id (str): App ID.
        top_k (int, optional): Top K results to retrieve. Defaults to 10.
        metric (str, optional): Similarity metric (either 'cosine' or 'euclidean'). Defaults to 'cosine'.
        alogrithm (str, optional): Search algorithm (either 'nearest_neighbor' or 'brute_force'). Defaults to 'nearest_neighbor'.
        pagination (bool, optional): Enable pagination. Defaults to False.
        base_url (str, optional): Base API url. Defaults to "https://api.clarifai.com".
        pat (str, optional): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.

    Raises:
        UserError: If the metric is not 'cosine' or 'euclidean'.
        UserError: If the algorithm is not 'nearest_neighbor' or 'brute_force'.
    """
    if metric not in ["cosine", "euclidean"]:
      raise UserError("Metric should be either cosine or euclidean")
    if algorithm not in ["nearest_neighbor", "brute_force"]:
      raise UserError("Algorithm should be either nearest_neighbor or brute_force")
    if metric == "cosine" and algorithm == "nearest_neighbor":
      raise UserError("Cosine distance metric is not supported with nearest neighbor algorithm")
    if top_k and pagination:
      raise UserError(
          "top_k and pagination cannot be used together. Please set pagination to False.")
    if not top_k and not pagination:
      top_k = DEFAULT_TOP_K

    self.user_id = user_id
    self.app_id = app_id
    self.metric_distance = dict(cosine="COSINE_DISTANCE", euclidean="EUCLIDEAN_DISTANCE")[metric]
    self.algorithm = algorithm
    self.data_proto = resources_pb2.Data()
    self.top_k = top_k
    self.pagination = pagination
    self.inputs = Inputs(
        user_id=self.user_id, app_id=self.app_id, pat=pat, token=token, base_url=base_url)
    self.rank_filter_schema = get_schema()
    BaseClient.__init__(
        self,
        user_id=self.user_id,
        app_id=self.app_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self, page_size=1000)

  def _get_annot_proto(self, **kwargs):
    """Get an Annotation proto message based on keyword arguments.

    Args:
        **kwargs: Keyword arguments specifying the resource.

    Returns:
        resources_pb2.Annotation: An Annotation proto message.
    """
    if not kwargs:
      return resources_pb2.Annotation()

    self.data_proto = resources_pb2.Data()
    for key, value in kwargs.items():
      if key == "image_bytes":
        image_proto = self.inputs.get_input_from_bytes("", image_bytes=value).data.image
        self.data_proto.image.CopyFrom(image_proto)

      elif key == "image_url":
        image_proto = self.inputs.get_input_from_url("", image_url=value).data.image
        self.data_proto.image.CopyFrom(image_proto)

      elif key == "concepts":
        for concept in value:
          concept_proto = resources_pb2.Concept(**concept)
          self.data_proto.concepts.add().CopyFrom(concept_proto)

      elif key == "text_raw":
        text_proto = self.inputs.get_input_from_bytes(
            "", text_bytes=bytes(value, 'utf-8')).data.text
        self.data_proto.text.CopyFrom(text_proto)

      elif key == "metadata":
        metadata_struct = Struct()
        metadata_struct.update(value)
        self.data_proto.metadata.CopyFrom(metadata_struct)

      elif key == "geo_point":
        geo_point_proto = self._get_geo_point_proto(value["longitude"], value["latitude"],
                                                    value["geo_limit"])
        self.data_proto.geo.CopyFrom(geo_point_proto)

      else:
        raise UserError(f"kwargs contain key that is not supported: {key}")
    return resources_pb2.Annotation(data=self.data_proto)

  def _get_input_proto(self, **kwargs):
    """Get an Input proto message based on keyword arguments.

    Args:
        **kwargs: Keyword arguments specifying the resource.

    Returns:
        resources_pb2.Input: An Input proto message.
    """
    if not kwargs:
      return resources_pb2.Input()

    self.input_proto = resources_pb2.Input()
    self.data_proto = resources_pb2.Data()
    for key, value in kwargs.items():
      if key == "input_types":
        for input_type in value:
          if input_type == "image":
            self.data_proto.image.CopyFrom(resources_pb2.Image())
          elif input_type == "text":
            self.data_proto.text.CopyFrom(resources_pb2.Text())
          elif input_type == "audio":
            self.data_proto.audio.CopyFrom(resources_pb2.Audio())
          elif input_type == "video":
            self.data_proto.video.CopyFrom(resources_pb2.Video())
        self.input_proto.data.CopyFrom(self.data_proto)
      elif key == "input_dataset_ids":
        self.input_proto.dataset_ids.extend(value)
      elif key == "input_status_code":
        self.input_proto.status.code = value
      else:
        raise UserError(f"kwargs contain key that is not supported: {key}")
    return self.input_proto

  def _get_geo_point_proto(self, longitude: float, latitude: float,
                           geo_limit: float) -> resources_pb2.Geo:
    """Get a GeoPoint proto message based on geographical data.

    Args:
        longitude (float): Longitude coordinate.
        latitude (float): Latitude coordinate.
        geo_limit (float): Geographical limit.

    Returns:
        resources_pb2.Geo: A Geo proto message.
    """
    return resources_pb2.Geo(
        geo_point=resources_pb2.GeoPoint(longitude=longitude, latitude=latitude),
        geo_limit=resources_pb2.GeoLimit(type="withinKilometers", value=geo_limit))

  def _list_topk_generator(self, endpoint: Callable[..., Any], proto_message: Any,
                           request_data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Lists all pages of a resource.

        Args:
            endpoint (Callable): The endpoint to call.
            proto_message (Any): The proto message to use.
            request_data (dict): The request data to use.

        Yields:
            response_dict: The next item in the listing.
        """
    max_pages = ceil(self.top_k / self.default_page_size)
    total_hits = 0
    page = 1
    while (page <= max_pages):
      if (page == max_pages):
        per_page = self.top_k - total_hits
      else:
        per_page = self.default_page_size
      request_data['pagination'] = service_pb2.Pagination(page=page, per_page=per_page)
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      if response.status.code != status_code_pb2.SUCCESS:
        if "page * perPage cannot exceed" in str(response.status.details):
          msg = (f"Your top_k is set to {self.top_k}. "
                 f"The current pagination settings exceed the limit. Please reach out to "
                 f"support@clarifai.com to request an increase for your use case.\n"
                 f"req_id: {response.status.req_id}")
          raise UserError(msg)
        else:
          raise Exception(f"Listing failed with response {response!r}")

      if 'hits' not in list(dict_response.keys()):
        break
      page += 1
      total_hits += per_page
      yield response

  def _list_all_pages_generator(self,
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
      request_data['pagination'] = service_pb2.Pagination(page=page, per_page=per_page)
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      if response.status.code != status_code_pb2.SUCCESS:
        if "page * perPage cannot exceed" in str(response.status.details):
          msg = (f"Your pagination is set to {page_no*per_page}. "
                 f"The current pagination settings exceed the limit. Please reach out to "
                 f"support@clarifai.com to request an increase for your use case.\n"
                 f"req_id: {response.status.req_id}")
          raise UserError(msg)
        else:
          raise Exception(f"Listing failed with response {response!r}")
      if 'hits' not in list(dict_response.keys()):
        break
      yield response
      if page_no is not None or per_page is not None:
        break
      page += 1

  def query(
      self,
      ranks=[{}],
      filters=[{}],
      page_no: int = None,
      per_page: int = None,
  ):
    """Perform a query with rank and filters.

    Args:
        ranks (List[Dict], optional): List of rank parameters. Defaults to [{}].
        filters (List[Dict], optional): List of filter parameters. Defaults to [{}].
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Returns:
        Generator[Dict[str, Any], None, None]: A generator of query results.

    Examples:
        Get successful inputs of type image or text
        >>> from clarifai.client.search import Search
        >>> search = Search(user_id='user_id', app_id='app_id', top_k=10, metric='cosine')
        # This performs OR operation on input_types and input_status_code
        >>> res = search.query(filters=[{'input_types': ['image', 'text'], 'input_status_code': 30000}])

        Vector search over inputs
        >>> from clarifai.client.search import Search
        >>> search = Search(user_id='user_id', app_id='app_id' , metric='cosine', pagination=True)
        >>> res = search.query(ranks=[{'image_url': 'https://samples.clarifai.com/dog.tiff'}],page_no=2, per_page=5)

    Note:
        For schema of rank and filter, please refer to [schema](https://github.com/Clarifai/clarifai-python/tree/master/clarifai/schema/search.py).
        For more detailed search examples, please refer to [examples](https://github.com/Clarifai/examples/tree/main/search).
    """
    if not self.pagination and (per_page or page_no):
      raise UserError("Pagination settings are only available when pagination is enabled."
                      "Please set Search(pagination=True) while initializing Search().")
    try:
      self.rank_filter_schema.validate(ranks)
      self.rank_filter_schema.validate(filters)
    except SchemaError as err:
      raise UserError(f"Invalid rank or filter input: {err}")

    # For each rank, create a Rank proto message
    # For ranks it only allows resources_pb2.Annotation proto, so no need of splitting protos into annotation and input.
    all_ranks = []
    for rank_dict in ranks:
      rank_annot_proto = self._get_annot_proto(**rank_dict)
      all_ranks.append(resources_pb2.Rank(annotation=rank_annot_proto))

    all_filters = []
    # check for filters which is compatible with input proto
    for each_filter in filters:
      input_dict = {
          key: each_filter.pop(key)
          for key in ['input_types', 'input_dataset_ids', 'input_status_code']
          if key in each_filter
      }

      all_filters.append(
          resources_pb2.Filter(
              annotation=self._get_annot_proto(**each_filter),
              input=self._get_input_proto(**input_dict)))

    # Create a PostInputsSearchesRequest proto message
    request_data = dict(
        user_app_id=self.user_app_id,
        searches=[
            resources_pb2.Search(
                query=resources_pb2.Query(ranks=all_ranks, filters=all_filters),
                algorithm=self.algorithm,
                metric=self.metric_distance)
        ])
    # Calls PostInputsSearches for annotation ranks, input filters
    if self.pagination:
      return self._list_all_pages_generator(self.STUB.PostInputsSearches,
                                            service_pb2.PostInputsSearchesRequest, request_data,
                                            page_no, per_page)
    return self._list_topk_generator(self.STUB.PostInputsSearches,
                                     service_pb2.PostInputsSearchesRequest, request_data)
