from typing import Any, Callable, Dict, Generator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from schema import And, Optional, Regex, Schema, SchemaError

from clarifai.client.base import BaseClient
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.errors import UserError


class Search(Lister, BaseClient):

  def __init__(self, user_id, app_id, top_k: int = 10, metric: str = "cosine"):
    """Initialize the Search object.

        Args:
            user_id (str): User ID.
            app_id (str): App ID.
            top_k (int, optional): Top K results to retrieve. Defaults to 10.
            metric (str, optional): Similarity metric (either 'cosine' or 'euclidean'). Defaults to 'cosine'.

        Raises:
            UserError: If the metric is not 'cosine' or 'euclidean'.
        """
    if metric not in ["cosine", "euclidean"]:
      raise UserError("Metric should be either cosine or euclidean")

    self.user_id = user_id
    self.app_id = app_id
    self.metric_distance = dict(cosine="COSINE_DISTANCE", euclidean="EUCLIDEAN_DISTANCE")[metric]
    self.data_proto = resources_pb2.Data()

    self.inputs = Inputs(user_id=self.user_id, app_id=self.app_id)
    self.rank_filter_schema = self.init_schema()
    BaseClient.__init__(self, user_id=self.user_id, app_id=self.app_id)
    Lister.__init__(self, page_size=top_k)

  def init_schema(self) -> Schema:
    """Initialize the schema for rank and filter.

        This schema validates:

        - Rank and filter must be a list
        - Each item in the list must be a dict
        - The dict can contain these optional keys:
            - 'image_url': Valid URL string
            - 'text_raw': Non-empty string
            - 'metadata': Dict
            - 'image_bytes': Bytes
            - 'geo_point': Dict with 'longitude', 'latitude' and 'geo_limit' as float, float and int respectively
            - 'concepts': List where each item is a concept dict
        - Concept dict requires at least one of:
            - 'name': Non-empty string with dashes/underscores
            - 'id': Non-empty string
            - 'language': Non-empty string
            - 'value': 0 or 1 integer

        Returns:
            Schema: The schema for rank and filter.
        """
    # Schema for a single concept
    concept_schema = Schema({
        Optional('value'):
            And(int, lambda x: x in [0, 1]),
        Optional('id'):
            And(str, len),
        Optional('language'):
            And(str, len),
        # Non-empty strings with internal dashes and underscores.
        Optional('name'):
            And(str, len, Regex(r'^[0-9A-Za-z]+([-_][0-9A-Za-z]+)*$'))
    })

    # Schema for a rank or filter item
    rank_filter_item_schema = Schema({
        Optional('image_url'):
            And(str, Regex(r'^https?://')),
        Optional('text_raw'):
            And(str, len),
        Optional('metadata'):
            dict,
        Optional('image_bytes'):
            bytes,
        Optional('geo_point'): {
            'longitude': float,
            'latitude': float,
            'geo_limit': int
        },
        Optional("concepts"):
            And(list,
                lambda x: all(concept_schema.is_valid(item) and len(item) > 0 for item in x)),
    })

    # Schema for rank and filter args
    return Schema([rank_filter_item_schema])

  def _add_data_proto(self, resource_type, resource_proto):
    """Add data to the data_proto field.

        Args:
            resource_type (str): Indicates the type of resource.
            resource_proto (Any): The resource data to add to data_proto.
        """
    if resource_type == "image":
      self.data_proto.image.CopyFrom(resource_proto)
    elif resource_type == "concept":
      self.data_proto.concepts.add().CopyFrom(resource_proto)
    elif resource_type == "text":
      self.data_proto.text.CopyFrom(resource_proto)
    elif resource_type == "metadata":
      self.data_proto.metadata.CopyFrom(resource_proto)
    elif resource_type == "geo_point":
      self.data_proto.geo.CopyFrom(resource_proto)

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
        self._add_data_proto("image", image_proto)

      elif key == "image_url":
        image_proto = self.inputs.get_input_from_url("", image_url=value).data.image
        self._add_data_proto("image", image_proto)

      elif key == "concepts":
        for concept in value:
          concept_proto = resources_pb2.Concept(**concept)
          self._add_data_proto("concept", concept_proto)

      elif key == "text_raw":
        text_proto = self.inputs.get_input_from_bytes(
            "", text_bytes=bytes(value, 'utf-8')).data.text
        self._add_data_proto("text", text_proto)

      elif key == "metadata":
        metadata_struct = Struct()
        metadata_struct.update(value)
        self._add_data_proto("metadata", metadata_struct)

      elif key == "geo_point":
        geo_point_proto = self._get_geo_point_proto(value["longitude"], value["latitude"],
                                                    value["geo_limit"])
        self._add_data_proto("geo_point", geo_point_proto)
    return resources_pb2.Annotation(data=self.data_proto)

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

  def list_all_pages_generator(
      self, endpoint: Callable[..., Any], proto_message: Any,
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
    request_data['pagination'] = service_pb2.Pagination(page=page, per_page=self.default_page_size)
    while True:
      request_data['pagination'].page = page
      response = self._grpc_request(endpoint, proto_message(**request_data))
      dict_response = MessageToDict(response, preserving_proto_field_name=True)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Listing failed with response {response!r}")

      if 'hits' not in list(dict_response.keys()):
        break
      page += 1
      yield response

  def query(self, ranks=[{}], filters=[{}]):
    """Perform a query with rank and filters.

        Args:
            ranks (List[Dict], optional): List of rank parameters. Defaults to [{}].
            filters (List[Dict], optional): List of filter parameters. Defaults to [{}].

        Returns:
            Generator[Dict[str, Any], None, None]: A generator of query results.
        """
    try:
      self.rank_filter_schema.validate(ranks)
      self.rank_filter_schema.validate(filters)
    except SchemaError as err:
      raise UserError(f"Invalid rank or filter input: {err}")

    rank_annot_proto, filters_annot_proto = [], []
    for rank_dict in ranks:
      rank_annot_proto.append(self._get_annot_proto(**rank_dict))
    for filter_dict in filters:
      filters_annot_proto.append(self._get_annot_proto(**filter_dict))

    all_ranks = [resources_pb2.Rank(annotation=rank_annot) for rank_annot in rank_annot_proto]
    all_filters = [
        resources_pb2.Filter(annotation=filter_annot) for filter_annot in filters_annot_proto
    ]

    request_data = dict(
        user_app_id=self.user_app_id,
        searches=[
            resources_pb2.Search(
                query=resources_pb2.Query(ranks=all_ranks, filters=all_filters),
                metric=self.metric_distance)
        ])

    return self.list_all_pages_generator(self.STUB.PostAnnotationsSearches,
                                         service_pb2.PostAnnotationsSearchesRequest, request_data)
