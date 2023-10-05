from typing import Dict, Optional, Set

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict


def get_yaml_output_info_proto(yaml_model_output_info: Dict) -> Optional[resources_pb2.OutputInfo]:
  """Converts a yaml model output info to an api model output info."""
  if not yaml_model_output_info:
    return None

  return resources_pb2.OutputInfo(
      params=convert_yaml_params_to_api_params(yaml_model_output_info.get('params')))


def convert_yaml_params_to_api_params(yaml_params: Dict) -> Optional[struct_pb2.Struct]:
  """Converts a yaml model output info params to an api model output info params."""
  if not yaml_params:
    return None

  s = struct_pb2.Struct()
  s.update(yaml_params)

  return s


def is_same_yaml_model(api_model: resources_pb2.Model, yaml_model: Dict) -> bool:
  """Compares a model from the API with a model from a yaml file."""
  api_model = MessageToDict(api_model, preserving_proto_field_name=True)

  yaml_model_from_api = dict()
  for k, _ in yaml_model.items():
    if k == "output_info" and api_model["model_version"].get("output_info", "") != "":
      yaml_model_from_api[k] = dict(params=api_model["model_version"]["output_info"].get("params"))
    else:
      yaml_model_from_api[k] = api_model.get(k)
  yaml_model_from_api.update({"model_id": api_model.get("id")})

  ignore_keys = {}

  return is_dict_in_dict(yaml_model, yaml_model_from_api, ignore_keys)


def is_dict_in_dict(d1: Dict, d2: Dict, ignore_keys: Set = None) -> bool:
  """Compares two dicts recursively."""
  for k, v in d1.items():
    if ignore_keys and k in ignore_keys:
      continue
    if k not in d2:
      return False
    if isinstance(v, dict):
      if not isinstance(d2[k], dict):
        return False
      return is_dict_in_dict(d1[k], d2[k], None)
    elif v != d2[k]:
      return False

  return True
