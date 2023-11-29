from typing import Any, Dict, List

from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.service_pb2 import MultiModelTypeResponse
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct


def response_to_templates(response: MultiModelTypeResponse, model_type_id: str) -> List[str]:
  """Converts the response from the API to a list of templates for the given model type id."""
  dict_response = MessageToDict(response)
  templates = []
  for model_type in dict_response['modelTypes']:
    if model_type['id'] == model_type_id:
      for modeltypefield in model_type['modelTypeFields']:
        if modeltypefield['path'].split('.')[-1] == "template":
          templates = [template['id'] for template in modeltypefield['modelTypeEnumOptions']]
  return templates


def response_to_model_params(response: MultiModelTypeResponse,
                             model_type_id: str,
                             template: str = None) -> Dict[str, Any]:
  """Converts the response from the API to a dictionary of model params for the given model type id."""
  dict_response = MessageToDict(response)
  params = {}
  if model_type_id != "clusterer":
    params["dataset_id"] = ""
    params["dataset_version_id"] = ""
  if model_type_id not in ["clusterer", "text-to-text"]:
    params["concepts"] = []
  params["train_params"] = dict()

  for model_type in dict_response['modelTypes']:
    if model_type['id'] == model_type_id:
      #iterate through the model type fields
      for modeltypefield in model_type['modelTypeFields']:
        _path = modeltypefield['path'].split('.')
        #removing the fields which are not required
        if (_path[0] in ["'eval_info'"]) or (_path[1] in ["dataset", "data"]) or (_path[-1] in [
            "dataset_id", "dataset_version_id"
        ]) or ("internalOnly" in modeltypefield.keys()):
          continue
        #checking the template model type fields
        if _path[-1] != "template":
          if _path[0] == 'train_info' or _path[0] == 'input_info':
            try:
              params["train_params"][_path[-1]] = modeltypefield['defaultValue']
            except Exception:
              params["train_params"][_path[-1]] = None
          if _path[0] == 'output_info':
            params["inference_params"] = dict()
            try:
              params["inference_params"][_path[-1]] = modeltypefield['defaultValue']
            except Exception:
              params["inference_params"][_path[-1]] = None
        else:
          if 'modelTypeEnumOptions' in modeltypefield.keys():
            #check given template is valid
            all_templates = [template['id'] for template in modeltypefield['modelTypeEnumOptions']]
            if template not in all_templates:
              raise ValueError(f"Invalid template {template} for model type {model_type_id}. "
                               f"Valid templates are {all_templates}")
            for modeltypeenum in modeltypefield['modelTypeEnumOptions']:
              #finding the given template
              if modeltypeenum['id'] == template:
                params['train_params']["template"] = modeltypeenum['id']
                #iterate through the template fields
                for modeltypeenumfield in modeltypeenum['modelTypeFields']:
                  if "internalOnly" in modeltypeenumfield.keys():
                    continue
                  try:
                    params["train_params"][modeltypeenumfield['path'].split('.')[
                        -1]] = modeltypeenumfield['defaultValue']
                  except Exception:
                    params["train_params"][modeltypeenumfield['path'].split('.')[-1]] = None
  #custom config
  if "custom_config" in params['train_params'].keys():
    # Write the content to the file
    file_path = params['train_params']['template'] + ".py"
    with open(file_path, "w") as script_file:
      script_file.write(params['train_params']['custom_config'])
    params['train_params']['custom_config'] = file_path

  return params


def params_parser(params_dict: dict) -> Dict[str, Any]:
  """Converts the params dictionary to a dictionary of model specific params for the given model"""
  #dict parser
  train_dict = {}
  train_dict["train_info"] = dict()
  train_dict["output_info"] = dict()

  train_dict["train_info"]['params'] = Struct()
  if 'custom_config' in params_dict['train_params'].keys():
    # Open and read the Python file
    with open(params_dict['train_params']['custom_config'], 'r') as python_file:
      custom_config = python_file.read()
    params_dict['train_params']['custom_config'] = custom_config

  if 'base_embed_model' in params_dict['train_params'].keys():
    train_dict["input_info"] = dict()
    train_dict["input_info"]['base_embed_model'] = params_dict['train_params']['base_embed_model']
    train_dict['input_info'] = resources_pb2.InputInfo(**train_dict['input_info'])
    del params_dict['train_params']['base_embed_model']

  train_dict["train_info"]['params'].update(params_dict["train_params"])
  if 'dataset_id' in params_dict.keys():
    train_dict["train_info"]['params']['dataset_id'] = params_dict['dataset_id']
    train_dict["train_info"]['params']['dataset_version_id'] = params_dict['dataset_version_id']
  train_dict['train_info'] = resources_pb2.TrainInfo(**train_dict['train_info'])

  if 'concepts' in params_dict.keys():
    train_dict["output_info"]['data'] = resources_pb2.Data(
        concepts=[resources_pb2.Concept(id=concept_id) for concept_id in params_dict["concepts"]])
  if 'inference_params' in params_dict.keys():
    train_dict["output_info"]['params'] = Struct()
    train_dict['output_info']['params'].update(params_dict["inference_params"])
  train_dict['output_info'] = resources_pb2.OutputInfo(**train_dict['output_info'])

  return train_dict


def response_to_param_info(response: MultiModelTypeResponse,
                           model_type_id: str,
                           param: str,
                           template: str = None) -> Dict[str, Any]:
  """Converts the response from the API to a dictionary of model param info for the given model type id."""
  dict_response = MessageToDict(response)
  for model_type in dict_response['modelTypes']:
    if model_type['id'] == model_type_id:
      #iterate through the model type fields
      for modeltypefield in model_type['modelTypeFields']:
        if modeltypefield['path'].split('.')[-1] == param:
          if param == 'template':
            del modeltypefield['placeholder']
            del modeltypefield['modelTypeEnumOptions']
            return modeltypefield
          modeltypefield['param'] = modeltypefield.pop('path').split('.')[-1]
          del modeltypefield['placeholder']
          return modeltypefield
        #checking the template model type fields
        if modeltypefield['path'].split('.')[-1] == "template":
          for modeltypeenum in modeltypefield['modelTypeEnumOptions']:
            if modeltypeenum['id'] == template:
              #iterate through the template fields
              for modeltypeenumfield in modeltypeenum['modelTypeFields']:
                if modeltypeenumfield['path'].split('.')[-1] == param:
                  modeltypeenumfield['param'] = modeltypeenumfield.pop('path').split('.')[-1]
                  del modeltypeenumfield['placeholder']
                  return modeltypeenumfield


def find_and_replace_key(nested_dict: Dict, target_key: str, replacement_value: Any) -> None:
  """Finds and replaces the target key with the replacement value in the nested dictionary."""
  for key, value in nested_dict.items():
    if key == target_key:
      nested_dict[key] = replacement_value
    elif isinstance(value, dict):
      find_and_replace_key(value, target_key, replacement_value)
