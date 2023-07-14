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

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

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

  def init_model(
      self,
      model_id: str,
      model_type: str,
      description: str = "",
  ):
    """Init a new model on Clarifai platform.

    Args:
        model_id (str): Clarifai model id
        model_type (str): Clarifai model type
        description (str, optional): a description of the model. Defaults to "".

    Returns:
        dict: Clarifai api response
    """
    user_data_object = self.auth.get_user_app_id_proto()
    post_models_response = self.stub.PostModels(
        service_pb2.PostModelsRequest(
            user_app_id=user_data_object,
            models=[resources_pb2.Model(id=model_id, notes=description,
                                        model_type_id=model_type)]),
        metadata=self.auth.metadata)

    return MessageToDict(post_models_response, preserving_proto_field_name=True)

  def post_model_version(
      self,
      model_id: str,
      model_zip_url: str,
      input: dict,
      outputs: dict,
  ):
    """Post a new version of an existing model in the Clarifai platform.

    Args:
        model_id (str): Clarifai model id
        model_zip_url (str]): url of zip of model
        model_zip_url (str): url of zip of model
        input (dict): a dict where the key is clarifai input field and the value is triton model input,
            {clarifai_input_field: triton_input_filed}.
        outputs (dict): a dict where the keys are clarifai output fields and the values are triton model outputs,
            {clarifai_output_field1: triton_output_filed1, clarifai_output_field2: triton_output_filed2,...}.

    Returns:
        dict: clarifai api response
    """
    user_data_object = self.auth.get_user_app_id_proto()

    def _parse_fields_map(x):
      """parse input, outputs to Struct"""
      _fields_map = Struct()
      _fields_map.update(x)
      return _fields_map

    input_fields_map = _parse_fields_map(input)
    output_fields_map = _parse_fields_map(outputs)
    post_model_versions = self.stub.PostModelVersions(
        service_pb2.PostModelVersionsRequest(
            user_app_id=user_data_object,
            model_id=model_id,
            model_versions=[
                resources_pb2.ModelVersion(
                    pretrained_model_config=resources_pb2.PretrainedModelConfig(
                        model_zip_url=model_zip_url,
                        input_fields_map=input_fields_map,
                        output_fields_map=output_fields_map))
            ]),
        metadata=self.auth.metadata)

    return MessageToDict(post_model_versions, preserving_proto_field_name=True)

  def upload_model(
      self,
      model_id: str,
      model_zip_url: str,
      input: dict,
      outputs: dict,
      model_type: str,
      description: str = "",
  ):
    """Doing 2 requests for initializing and creating version for a new trained model to the Clarifai platform.

    Args:
        model_id (str): Clarifai model id
        model_zip_url (str): url of zip of model
        input (dict): a dict where the key is clarifai input field and the value is triton model input,
            {clarifai_input_field: triton_input_filed}
        outputs (dict): a dict where the keys are clarifai output fields and the values are triton model outputs,
            {clarifai_output_field1: triton_output_filed1, clarifai_output_field2: triton_output_filed2,...}
        model_type (str): Clarifai model type.
        description (str, optional): a description of the model. Defaults to "".

    Returns:
        dict: Clarifai api response
    """
    init_resp = self.init_model(model_id, model_type, description)
    if init_resp["status"]["code"] != "SUCCESS":
      return init_resp
    version_resp = self.post_model_version(model_id, model_zip_url, input, outputs)

    return version_resp

  def delete_model(self, model_id: str):
    """Delete model api by model id

    Args:
        model_id (str): Clarifai model id

    Returns:
        dict: clarifai api response
    """
    user_data_object = self.auth.get_user_app_id_proto()
    delete_model_response = self.stub.DeleteModel(
        service_pb2.DeleteModelRequest(
            user_app_id=user_data_object,
            model_id=model_id,
        ),
        metadata=self.auth.metadata)

    return MessageToDict(delete_model_response, preserving_proto_field_name=True)

  def delete_model_version(self, model_id: str, version_id: str):
    """Delete specific version of model

    Args:
        model_id (str): Clarifai model id
        version_id (str): version id of model that will be removed

    Returns:
        dict: Clarifai API response
    """
    user_data_object = self.auth.get_user_app_id_proto()
    delete_model_response = self.stub.DeleteModelVersion(
        service_pb2.DeleteModelVersionRequest(
            user_app_id=user_data_object, model_id=model_id, version_id=version_id),
        metadata=self.auth.metadata)

    return MessageToDict(delete_model_response, preserving_proto_field_name=True)

  def get_model(self, model_id: str):
    """Get model by id

    Args:
        model_id (str): Clarifai model id

    Returns:
        dict: Clarifai API response
    """
    user_data_object = self.auth.get_user_app_id_proto()
    response = self.stub.GetModel(
        service_pb2.GetModelRequest(
            user_app_id=user_data_object,
            model_id=model_id,
        ),
        metadata=self.auth.metadata)

    return MessageToDict(response, preserving_proto_field_name=True)
