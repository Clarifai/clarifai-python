# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interface to Clarifai Models API."""

from typing import Dict, Type

from clarifai_grpc.grpc.api import service_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub


class Models:
  """
  Interface to Clarifai models api
  """

  def __init__(self, auth: Type[ClarifaiAuthHelper]) -> None:
    self.auth = auth
    self.stub = create_stub(self.auth)

  def list_model_types(self) -> Dict:
    """
    List all API Model Types that support input and output.

    Returns:
      - A dict of;
          model_type, [{expected_input_name: value}, {expected_output_name: value}, model_descripton]
        key, value pairs respectively.i.e.
          {model_type: [{inp_field_name: value}, {output_field_name: value,...}, model_description]}

      - model_type: supported input shapes and data types dict. Structure;
          {model_type: [(supported_input_N_dims, dtypes_N),...]}

      - model_type: supported output shapes and data types dict. Structure;
          {model_type: [(supported_output_N_dims, dtypes_N),...]}
    """
    ## List model types from API
    model_types_response = self.stub.ListModelTypes(
        service_pb2.ListModelTypesRequest(),  #(user_app_id=auth.user_app_id),
        metadata=self.auth.metadata)
    # model types dict structure:
    # {model_type: [{inp_field_name: value,}, {output_field_name: value,...}, desc]}
    model_types = {}
    in_dims_dtype = {}  # {model_type: [(supported_input_N_dims, dtypes_N),...]}
    out_dims_dtype = {}  # {model_type: [(supported_output_N_dims, dtypes_N),...]}
    types_dict = MessageToDict(
        model_types_response, preserving_proto_field_name=True)['model_types']
    for i in range(len(types_dict)):
      model_id = types_dict[i]['id']
      model_desc = types_dict[i]['description']
      if 'expected_output_layers' in types_dict[i].keys():
        # expected_input_layers exist for all expected_output_layers
        # hence one conditional check
        model_types[model_id] = []
        in_dims_dtype[model_id] = []
        out_dims_dtype[model_id] = []
        expected_input = types_dict[i]['expected_input_layers']
        expected_output = types_dict[i]['expected_output_layers']
        for inp in expected_input:
          if 'data_field_name' in inp.keys():
            model_types[model_id].append({inp['data_field_name']: None})
          if 'shapes' in inp.keys():
            for dim in inp['shapes']:
              if 'dims' in dim.keys():
                if 'max_dims' in dim.keys():
                  in_dims_dtype[model_id].append((dim['dims'], dim['max_dims'], dim['data_type']))
                else:
                  in_dims_dtype[model_id].append((dim['dims'], dim['data_type']))
          continue
        temp_out = {}
        for each in expected_output:
          if 'data_field_name' in each.keys():
            temp_out[each['data_field_name']] = None
          if 'shapes' in each.keys():
            for dim in each['shapes']:
              if 'dims' in dim.keys():
                out_dims_dtype[model_id].append((dim['dims'], dim['data_type']))
          else:
            continue
        model_types[model_id].append(temp_out)
        model_types[model_id].append(model_desc)

    return {
        "Model Types": model_types,
        "Input Metadata": in_dims_dtype,
        "Output Metadata": out_dims_dtype
    }

  def post_model(self):
    """
    Post a new trained model to the Clarifai platform.
    Args:
      auth: Clarifai Auth object
    """
    raise NotImplementedError()

  def post_model_version(self):
    """
    Post a new version of an existing model in the Clarifai platform.
    Args:
      auth: Clarifai Auth object
    """
    raise NotImplementedError()
