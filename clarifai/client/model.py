import os
import time
from typing import Any, Dict, Generator, List, Tuple, Union

import numpy as np
import requests
import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Input
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.client.dataset import Dataset
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.constants.model import MAX_MODEL_PREDICT_INPUTS, TRAINABLE_MODEL_TYPES
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.logging import get_logger
from clarifai.utils.misc import BackoffIterator
from clarifai.utils.model_train import (find_and_replace_key, params_parser,
                                        response_to_model_params, response_to_param_info,
                                        response_to_templates)


class Model(Lister, BaseClient):
  """Model is a class that provides access to Clarifai API endpoints related to Model information."""

  def __init__(self,
               url: str = None,
               model_id: str = None,
               model_version: Dict = {'id': ""},
               base_url: str = "https://api.clarifai.com",
               pat: str = None,
               token: str = None,
               root_certificates_path: str = None,
               **kwargs):
    """Initializes a Model object.

    Args:
        url (str): The URL to initialize the model object.
        model_id (str): The Model ID to interact with.
        model_version (dict): The Model Version to interact with.
        base_url (str): Base API url. Default "https://api.clarifai.com"
        pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
        token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
        root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
        **kwargs: Additional keyword arguments to be passed to the Model.
    """
    if url and model_id:
      raise UserError("You can only specify one of url or model_id.")
    if not url and not model_id:
      raise UserError("You must specify one of url or model_id.")
    if url:
      user_id, app_id, _, model_id, model_version_id = ClarifaiUrlHelper.split_clarifai_url(url)
      model_version = {'id': model_version_id}
      kwargs = {'user_id': user_id, 'app_id': app_id}
    self.kwargs = {**kwargs, 'id': model_id, 'model_version': model_version,}
    self.model_info = resources_pb2.Model(**self.kwargs)
    self.logger = get_logger(logger_level="INFO", name=__name__)
    self.training_params = {}
    BaseClient.__init__(
        self,
        user_id=self.user_id,
        app_id=self.app_id,
        base=base_url,
        pat=pat,
        token=token,
        root_certificates_path=root_certificates_path)
    Lister.__init__(self)

  def list_training_templates(self) -> List[str]:
    """Lists all the training templates for the model type.

    Returns:
        templates (List): List of training templates for the model type.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> print(model.list_training_templates())
    """
    if not self.model_info.model_type_id:
      self.load_info()
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")
    request = service_pb2.ListModelTypesRequest(user_app_id=self.user_app_id,)
    response = self._grpc_request(self.STUB.ListModelTypes, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    templates = response_to_templates(
        response=response, model_type_id=self.model_info.model_type_id)

    return templates

  def get_params(self, template: str = None, save_to: str = 'params.yaml') -> Dict[str, Any]:
    """Returns the model params for the model type and yaml file.

    Args:
        template (str): The template to use for the model type.
        yaml_file (str): The yaml file to save the model params.

    Returns:
        params (Dict): Dictionary of model params for the model type.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_params = model.get_params(template='template', yaml_file='model_params.yaml')
    """
    if not self.model_info.model_type_id:
      self.load_info()
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")
    if template is None and self.model_info.model_type_id not in [
        "clusterer", "embedding-classifier"
    ]:
      raise UserError(
          f"Template should be provided for {self.model_info.model_type_id} model type")
    if template is not None and self.model_info.model_type_id in [
        "clusterer", "embedding-classifier"
    ]:
      raise UserError(
          f"Template should not be provided for {self.model_info.model_type_id} model type")

    request = service_pb2.ListModelTypesRequest(user_app_id=self.user_app_id,)
    response = self._grpc_request(self.STUB.ListModelTypes, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    params = response_to_model_params(
        response=response, model_type_id=self.model_info.model_type_id, template=template)
    #yaml file
    assert save_to.endswith('.yaml'), "File extension should be .yaml"
    with open(save_to, 'w') as f:
      yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    #updating the global model params
    self.training_params.update(params)

    return params

  def update_params(self, **kwargs) -> None:
    """Updates the model params for the model.

    Args:
        **kwargs: model params to update.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_params = model.get_params(template='template', yaml_file='model_params.yaml')
        >>> model.update_params(batch_size = 8, dataset_version = 'dataset_version_id')
    """
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")
    if len(self.training_params) == 0:
      raise UserError(
          f"Run 'model.get_params' to get the params for the {self.model_info.model_type_id} model type"
      )
    #getting all the keys in nested dictionary
    all_keys = [key for key in self.training_params.keys()] + [
        key for key in self.training_params.values() if isinstance(key, dict) for key in key
    ]
    #checking if the given params are valid
    if not set(kwargs.keys()).issubset(all_keys):
      raise UserError("Invalid params")
    #updating the global model params
    for key, value in kwargs.items():
      find_and_replace_key(self.training_params, key, value)

  def get_param_info(self, param: str) -> Dict[str, Any]:
    """Returns the param info for the param.

    Args:
        param (str): The param to get the info for.

    Returns:
        param_info (Dict): Dictionary of model param info for the param.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_params = model.get_params(template='template', yaml_file='model_params.yaml')
        >>> model.get_param_info('param')
    """
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")
    if len(self.training_params) == 0:
      raise UserError(
          f"Run 'model.get_params' to get the params for the {self.model_info.model_type_id} model type"
      )

    all_keys = [key for key in self.training_params.keys()] + [
        key for key in self.training_params.values() if isinstance(key, dict) for key in key
    ]
    if param not in all_keys:
      raise UserError(f"Invalid param: '{param}' for model type '{self.model_info.model_type_id}'")
    template = self.training_params['train_params']['template'] if 'template' in all_keys else None

    request = service_pb2.ListModelTypesRequest(user_app_id=self.user_app_id,)
    response = self._grpc_request(self.STUB.ListModelTypes, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    param_info = response_to_param_info(
        response=response,
        model_type_id=self.model_info.model_type_id,
        param=param,
        template=template)

    return param_info

  def train(self, yaml_file: str = None) -> str:
    """Trains the model based on the given yaml file or model params.

    Args:
        yaml_file (str): The yaml file for the model params.

    Returns:
        model_version_id (str): The model version ID for the model.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_params = model.get_params(template='template', yaml_file='model_params.yaml')
        >>> model.train('model_params.yaml')
    """
    if not self.model_info.model_type_id:
      self.load_info()
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")
    if not yaml_file and len(self.training_params) == 0:
      raise UserError("Provide yaml file or run 'model.get_params()'")

    if yaml_file:
      with open(yaml_file, 'r') as file:
        params_dict = yaml.safe_load(file)
    else:
      params_dict = self.training_params
    #getting all the concepts for the model type
    if self.model_info.model_type_id not in ["clusterer", "text-to-text"]:
      concepts = self._list_concepts()
    train_dict = params_parser(params_dict, concepts)
    request = service_pb2.PostModelVersionsRequest(
        user_app_id=self.user_app_id,
        model_id=self.id,
        model_versions=[resources_pb2.ModelVersion(**train_dict)])
    response = self._grpc_request(self.STUB.PostModelVersions, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel Training Started\n%s", response.status)

    return response.model.model_version.id

  def training_status(self, version_id: str = None, training_logs: bool = False) -> Dict[str, str]:
    """Get the training status for the model version. Also stores training logs

    Args:
        version_id (str): The version ID to get the training status for.
        training_logs (bool): Whether to save the training logs in a file.

    Returns:
        training_status (Dict): Dictionary of training status for the model version.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model.training_status(version_id='version_id',training_logs=True)
    """
    if not version_id and not self.model_info.model_version.id:
      raise UserError(
          "Model version ID is missing. Please provide a `model_version` with a valid `id` as an argument or as a URL in the following format: '{user_id}/{app_id}/models/{your_model_id}/model_version_id/{your_version_model_id}' when initializing."
      )

    if not self.model_info.model_type_id or not self.model_info.model_version.train_log:
      self.load_info()
    if self.model_info.model_type_id not in TRAINABLE_MODEL_TYPES:
      raise UserError(f"Model type {self.model_info.model_type_id} is not trainable")

    if training_logs:
      try:
        if self.model_info.model_version.train_log:
          log_response = requests.get(self.model_info.model_version.train_log)
          log_response.raise_for_status()  # Check for any HTTP errors
          with open(version_id + '.log', 'wb') as file:
            for chunk in log_response.iter_content(chunk_size=4096):  # 4KB
              file.write(chunk)
          self.logger.info(f"\nTraining logs are saving in '{version_id+'.log'}' file")

      except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while getting training logs: {e}")

    return self.model_info.model_version.status

  def delete_version(self, version_id: str) -> None:
    """Deletes a model version for the Model.

    Args:
        version_id (str): The version ID to delete.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model.delete_version(version_id='version_id')
    """
    request = service_pb2.DeleteModelVersionRequest(
        user_app_id=self.user_app_id, model_id=self.id, version_id=version_id)

    response = self._grpc_request(self.STUB.DeleteModelVersion, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel Version Deleted\n%s", response.status)

  def create_version(self, **kwargs) -> 'Model':
    """Creates a model version for the Model.

    Args:
        **kwargs: Additional keyword arguments to be passed to Model Version.
          - description (str): The description of the model version.
          - concepts (list[Concept]): The concepts to associate with the model version.
          - output_info (resources_pb2.OutputInfo(): The output info to associate with the model version.

    Returns:
        Model: A Model object for the specified model ID.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("url")
                    or
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_version = model.create_version(description='model_version_description')
    """
    if self.model_info.model_type_id in TRAINABLE_MODEL_TYPES:
      raise UserError(
          f"{self.model_info.model_type_id} is a trainable model type. Use 'model.train()' to train the model"
      )

    request = service_pb2.PostModelVersionsRequest(
        user_app_id=self.user_app_id,
        model_id=self.id,
        model_versions=[resources_pb2.ModelVersion(**kwargs)])

    response = self._grpc_request(self.STUB.PostModelVersions, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info("\nModel Version created\n%s", response.status)

    kwargs.update({'app_id': self.app_id, 'user_id': self.user_id})
    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    kwargs = self.process_response_keys(dict_response['model'], 'model')

    return Model(base_url=self.base, pat=self.pat, token=self.token, **kwargs)

  def list_versions(self, page_no: int = None,
                    per_page: int = None) -> Generator['Model', None, None]:
    """Lists all the versions for the model.

    Args:
        page_no (int): The page number to list.
        per_page (int): The number of items per page.

    Yields:
        Model: Model objects for the versions of the model.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("url") # Example URL: https://clarifai.com/clarifai/main/models/general-image-recognition
                    or
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> all_model_versions = list(model.list_versions())

    Note:
        Defaults to 16 per page if page_no is specified and per_page is not specified.
        If both page_no and per_page are None, then lists all the resources.
    """
    request_data = dict(
        user_app_id=self.user_app_id,
        model_id=self.id,
    )
    all_model_versions_info = self.list_pages_generator(
        self.STUB.ListModelVersions,
        service_pb2.ListModelVersionsRequest,
        request_data,
        per_page=per_page,
        page_no=page_no)

    for model_version_info in all_model_versions_info:
      model_version_info['id'] = model_version_info['model_version_id']
      del model_version_info['model_version_id']
      try:
        del model_version_info['train_info']['dataset']['version']['metrics']
      except KeyError:
        pass
      yield Model.from_auth_helper(
          auth=self.auth_helper,
          model_id=self.id,
          **dict(self.kwargs, model_version=model_version_info))

  def predict(self, inputs: List[Input], inference_params: Dict = {}, output_config: Dict = {}):
    """Predicts the model based on the given inputs.

    Args:
        inputs (list[Input]): The inputs to predict, must be less than 128.
    """
    if not isinstance(inputs, list):
      raise UserError('Invalid inputs, inputs must be a list of Input objects.')
    if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
      raise UserError(f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}."
                     )  # TODO Use Chunker for inputs len > 128

    self._override_model_version(inference_params, output_config)
    request = service_pb2.PostModelOutputsRequest(
        user_app_id=self.user_app_id,
        model_id=self.id,
        version_id=self.model_version.id,
        inputs=inputs,
        model=self.model_info)

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    while True:
      response = self._grpc_request(self.STUB.PostModelOutputs, request)

      if response.status.code == status_code_pb2.MODEL_DEPLOYING and \
        time.time() - start_time < 60 * 10: # 10 minutes
        self.logger.info(f"{self.id} model is still deploying, please wait...")
        time.sleep(next(backoff_iterator))
        continue

      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model Predict failed with response {response.status!r}")
      else:
        break

    return response

  def predict_by_filepath(self,
                          filepath: str,
                          input_type: str,
                          inference_params: Dict = {},
                          output_config: Dict = {}):
    """Predicts the model based on the given filepath.

    Args:
        filepath (str): The filepath to predict.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.
        inference_params (dict): The inference params to override.
        output_config (dict): The output config to override.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("url") # Example URL: https://clarifai.com/clarifai/main/models/general-image-recognition
                    or
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_prediction = model.predict_by_filepath('/path/to/image.jpg', 'image')
        >>> model_prediction = model.predict_by_filepath('/path/to/text.txt', 'text')
    """
    if not os.path.isfile(filepath):
      raise UserError('Invalid filepath.')

    with open(filepath, "rb") as f:
      file_bytes = f.read()

    return self.predict_by_bytes(file_bytes, input_type, inference_params, output_config)

  def predict_by_bytes(self,
                       input_bytes: bytes,
                       input_type: str,
                       inference_params: Dict = {},
                       output_config: Dict = {}):
    """Predicts the model based on the given bytes.

    Args:
        input_bytes (bytes): File Bytes to predict on.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.
        inference_params (dict): The inference params to override.
        output_config (dict): The output config to override.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("https://clarifai.com/openai/chat-completion/models/GPT-4")
        >>> model_prediction = model.predict_by_bytes(b'Write a tweet on future of AI',
                                                      input_type='text',
                                                      inference_params=dict(temperature=str(0.7), max_tokens=30)))
    """
    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError(
          f"Got input type {input_type} but expected one of image, text, video, audio.")
    if not isinstance(input_bytes, bytes):
      raise UserError('Invalid bytes.')

    if input_type == "image":
      input_proto = Inputs.get_input_from_bytes("", image_bytes=input_bytes)
    elif input_type == "text":
      input_proto = Inputs.get_input_from_bytes("", text_bytes=input_bytes)
    elif input_type == "video":
      input_proto = Inputs.get_input_from_bytes("", video_bytes=input_bytes)
    elif input_type == "audio":
      input_proto = Inputs.get_input_from_bytes("", audio_bytes=input_bytes)

    return self.predict(
        inputs=[input_proto], inference_params=inference_params, output_config=output_config)

  def predict_by_url(self,
                     url: str,
                     input_type: str,
                     inference_params: Dict = {},
                     output_config: Dict = {}):
    """Predicts the model based on the given URL.

    Args:
        url (str): The URL to predict.
        input_type (str): The type of input. Can be 'image', 'text', 'video' or 'audio.
        inference_params (dict): The inference params to override.
        output_config (dict): The output config to override.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("url") # Example URL: https://clarifai.com/clarifai/main/models/general-image-recognition
                    or
        >>> model = Model(model_id='model_id', user_id='user_id', app_id='app_id')
        >>> model_prediction = model.predict_by_url('url', 'image')
    """
    if input_type not in {'image', 'text', 'video', 'audio'}:
      raise UserError(
          f"Got input type {input_type} but expected one of image, text, video, audio.")

    if input_type == "image":
      input_proto = Inputs.get_input_from_url("", image_url=url)
    elif input_type == "text":
      input_proto = Inputs.get_input_from_url("", text_url=url)
    elif input_type == "video":
      input_proto = Inputs.get_input_from_url("", video_url=url)
    elif input_type == "audio":
      input_proto = Inputs.get_input_from_url("", audio_url=url)

    return self.predict(
        inputs=[input_proto], inference_params=inference_params, output_config=output_config)

  def _override_model_version(self, inference_params: Dict = {}, output_config: Dict = {}) -> None:
    """Overrides the model version.

    Args:
        inference_params (dict): The inference params to override.
        output_config (dict): The output config to override.
          min_value (float): The minimum value of the prediction confidence to filter.
          max_concepts (int): The maximum number of concepts to return.
          select_concepts (list[Concept]): The concepts to select.
          sample_ms (int): The number of milliseconds to sample.
    """
    params = Struct()
    if inference_params is not None:
      params.update(inference_params)

    self.model_info.model_version.output_info.CopyFrom(
        resources_pb2.OutputInfo(
            output_config=resources_pb2.OutputConfig(**output_config), params=params))

  def _list_concepts(self) -> List[str]:
    """Lists all the concepts for the model type.

    Returns:
        concepts (List): List of concepts for the model type.
    """
    request_data = dict(user_app_id=self.user_app_id)
    all_concepts_infos = self.list_pages_generator(self.STUB.ListConcepts,
                                                   service_pb2.ListConceptsRequest, request_data)
    return [concept_info['concept_id'] for concept_info in all_concepts_infos]

  def load_info(self) -> None:
    """Loads the model info."""
    request = service_pb2.GetModelRequest(
        user_app_id=self.user_app_id,
        model_id=self.id,
        version_id=self.model_info.model_version.id)
    response = self._grpc_request(self.STUB.GetModel, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)

    dict_response = MessageToDict(response, preserving_proto_field_name=True)
    self.kwargs = self.process_response_keys(dict_response['model'])
    self.model_info = resources_pb2.Model(**self.kwargs)

  def __getattr__(self, name):
    return getattr(self.model_info, name)

  def __str__(self):
    if len(self.kwargs) < 10:
      self.load_info()

    init_params = [param for param in self.kwargs.keys()]
    attribute_strings = [
        f"{param}={getattr(self.model_info, param)}" for param in init_params
        if hasattr(self.model_info, param)
    ]
    return f"Model Details: \n{', '.join(attribute_strings)}\n"

  def list_evaluations(self) -> resources_pb2.EvalMetrics:
    """List all eval_metrics of current model version

    Raises:
        Exception: Failed to call API

    Returns:
        resources_pb2.EvalMetrics
    """
    assert self.model_info.model_version.id, "Model version is empty. Please provide `model_version` as arguments or with a URL as the format '{user_id}/{app_id}/models/{your_model_id}/model_version_id/{your_version_model_id}' when initializing."
    request = service_pb2.ListModelVersionEvaluationsRequest(
        user_app_id=self.user_app_id,
        model_id=self.id,
        model_version_id=self.model_info.model_version.id)
    response = self._grpc_request(self.STUB.ListModelVersionEvaluations, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)

    return response.eval_metrics

  def evaluate(self,
               dataset: Dataset = None,
               dataset_id: str = None,
               dataset_app_id: str = None,
               dataset_user_id: str = None,
               dataset_version_id: str = None,
               eval_id: str = None,
               extended_metrics: dict = None,
               eval_info: dict = None) -> resources_pb2.EvalMetrics:
    """ Run evaluation

    Args:
      dataset (Dataset): If Clarifai Dataset is set, it will ignore other arguments prefixed with 'dataset_'.
      dataset_id (str): Dataset Id. Default is None.
      dataset_app_id (str): App ID for cross app evaluation, leave it as None to use Model App ID. Default is None.
      dataset_user_id (str): User ID for cross app evaluation, leave it as None to use Model User ID. Default is None.
      dataset_version_id (str): Dataset version Id. Default is None.
      eval_id (str): Specific ID for the evaluation. You must specify this parameter to either overwrite the result with the dataset ID or format your evaluation in an informative manner. If you don't, it will use random ID from system. Default is None.
      extended_metrics (dict): user custom metrics result. Default is None.
      eval_info (dict): custom eval info. Default is empty dict.

    Return
      eval_metrics

    """
    assert self.model_info.model_version.id, "Model version is empty. Please provide `model_version` as arguments or with a URL as the format '{user_id}/{app_id}/models/{your_model_id}/model_version_id/{your_version_model_id}' when initializing."

    if dataset:
      self.logger.info("Using dataset, ignore other arguments prefixed with 'dataset_'")
      dataset_id = dataset.id
      dataset_app_id = dataset.app_id
      dataset_user_id = dataset.user_id
      dataset_version_id = dataset.version.id
    else:
      self.logger.warning(
          "Arguments prefixed with `dataset_` will be removed soon, please use dataset")

    gt_dataset = resources_pb2.Dataset(
        id=dataset_id,
        app_id=dataset_app_id or self.auth_helper.app_id,
        user_id=dataset_user_id or self.auth_helper.user_id,
        version=resources_pb2.DatasetVersion(id=dataset_version_id))

    metrics = None
    if isinstance(extended_metrics, dict):
      metrics = Struct()
      metrics.update(extended_metrics)
      metrics = resources_pb2.ExtendedMetrics(user_metrics=metrics)

    eval_info_params = None
    if isinstance(eval_info, dict):
      eval_info_params = Struct()
      eval_info_params.update(eval_info)
      eval_info_params = resources_pb2.EvalInfo(params=eval_info_params)

    eval_metric = resources_pb2.EvalMetrics(
        id=eval_id,
        model=resources_pb2.Model(
            id=self.id,
            app_id=self.auth_helper.app_id,
            user_id=self.auth_helper.user_id,
            model_version=resources_pb2.ModelVersion(id=self.model_info.model_version.id),
        ),
        extended_metrics=metrics,
        ground_truth_dataset=gt_dataset,
        eval_info=eval_info_params,
    )
    request = service_pb2.PostEvaluationsRequest(
        user_app_id=self.user_app_id,
        eval_metrics=[eval_metric],
    )
    response = self._grpc_request(self.STUB.PostEvaluations, request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self.logger.info(
        "\nModel evaluation in progress. Kindly allow a few minutes for completion. Processing time may vary based on the model and dataset sizes."
    )

    return response.eval_metrics

  def get_eval_by_id(
      self,
      eval_id: str,
      label_counts=False,
      test_set=False,
      binary_metrics=False,
      confusion_matrix=False,
      metrics_by_class=False,
      metrics_by_area=False,
  ) -> resources_pb2.EvalMetrics:
    """Get detail eval_metrics by eval_id with extra metric fields

    Args:
        eval_id (str): eval id
        label_counts (bool, optional): Set True to get label counts. Defaults to False.
        test_set (bool, optional): Set True to get test set. Defaults to False.
        binary_metrics (bool, optional): Set True to get binary metric. Defaults to False.
        confusion_matrix (bool, optional): Set True to get confusion matrix. Defaults to False.
        metrics_by_class (bool, optional): Set True to get metrics by class. Defaults to False.
        metrics_by_area (bool, optional): Set True to get metrics by area. Defaults to False.

    Raises:
        Exception: Failed to call API

    Returns:
        resources_pb2.EvalMetrics: eval_metrics
    """
    request = service_pb2.GetEvaluationRequest(
        user_app_id=self.user_app_id,
        evaluation_id=eval_id,
        fields=resources_pb2.FieldsValue(
            label_counts=label_counts,
            test_set=test_set,
            binary_metrics=binary_metrics,
            confusion_matrix=confusion_matrix,
            metrics_by_class=metrics_by_class,
            metrics_by_area=metrics_by_area,
        ))
    response = self._grpc_request(self.STUB.GetEvaluation, request)

    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)

    return response.eval_metrics

  def get_latest_eval(self,
                      label_counts=False,
                      test_set=False,
                      binary_metrics=False,
                      confusion_matrix=False,
                      metrics_by_class=False,
                      metrics_by_area=False) -> Union[resources_pb2.EvalMetrics, None]:
    """
    Run `get_eval_by_id` method with latest `eval_id`

    Args:
      label_counts (bool, optional): Set True to get label counts. Defaults to False.
      test_set (bool, optional): Set True to get test set. Defaults to False.
      binary_metrics (bool, optional): Set True to get binary metric. Defaults to False.
      confusion_matrix (bool, optional): Set True to get confusion matrix. Defaults to False.
      metrics_by_class (bool, optional): Set True to get metrics by class. Defaults to False.
      metrics_by_area (bool, optional): Set True to get metrics by area. Defaults to False.

    Returns:
      eval_metric if model is evaluated otherwise None.

    """

    _latest = self.list_evaluations()[0]
    result = None
    if _latest.status.code == status_code_pb2.MODEL_EVALUATED:
      result = self.get_eval_by_id(
          eval_id=_latest.id,
          label_counts=label_counts,
          test_set=test_set,
          binary_metrics=binary_metrics,
          confusion_matrix=confusion_matrix,
          metrics_by_class=metrics_by_class,
          metrics_by_area=metrics_by_area)

    return result

  def get_eval_by_dataset(self, dataset: Dataset) -> List[resources_pb2.EvalMetrics]:
    """Get all eval data of dataset

    Args:
        dataset (Dataset): Clarifai dataset

    Returns:
        List[resources_pb2.EvalMetrics]
    """
    _id = dataset.id
    app = dataset.app_id or self.app_id
    user_id = dataset.user_id or self.user_id
    version = dataset.version.id

    list_eval: resources_pb2.EvalMetrics = self.list_evaluations()
    outputs = []
    for _eval in list_eval:
      if _eval.status.code == status_code_pb2.MODEL_EVALUATED:
        gt_ds = _eval.ground_truth_dataset
        if (_id == gt_ds.id and user_id == gt_ds.user_id and app == gt_ds.app_id):
          if not version or version == gt_ds.version.id:
            outputs.append(_eval)

    return outputs

  def get_raw_eval(self,
                   dataset: Dataset = None,
                   eval_id: str = None,
                   return_format: str = 'array') -> Union[resources_pb2.EvalTestSetEntry, Tuple[
                       np.array, np.array, list, List[Input]], Tuple[List[dict], List[dict]]]:
    """Get ground truths, predictions and input information. Do not pass dataset and eval_id at same time

    Args:
        dataset (Dataset): Clarifai dataset, get eval data of latest eval result of dataset.
        eval_id (str): Evaluation ID, get eval data of specific eval id.
        return_format (str, optional): Choice {proto, array, coco}. !Note that `coco` is only applicable for 'visual-detector'. Defaults to 'array'.

    Returns:

        Depends on `return_format`.

        * if return_format == proto
          `resources_pb2.EvalTestSetEntry`

        * if return_format == array
          `Tuple(np.array, np.array, List[str], List[Input])`: Tuple has 4 elements (y, y_pred, concept_ids, inputs).
            y, y_pred, concept_ids can be used to compute metrics. 'inputs' can be use to download
            - if model is 'classifier': 'y' and 'y_pred' are both arrays with a shape of (num_inputs,)
            - if model is 'visual-detector': 'y' and 'y_pred' are arrays with a shape of (num_inputs,), where each element is array has shape (num_annotation, 6) consists of [x_min, y_min, x_max, y_max, concept_index, score]. The score is always 1 for 'y'

        * if return_format == coco: Applicable only for 'visual-detector'
          `Tuple[List[Dict], List[Dict]]`: Tuple has 2 elemnts where first element is COCO Ground Truth and last one is COCO Prediction Annotation

    Example Usages:
    ------
    * Evaluate `visual-classifier` using sklearn

    ```python
    import os
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    import numpy as np
    from clarifai.client.model import Model
    from clarifai.client.dataset import Dataset
    os.environ["CLARIFAI_PAT"] = "???"
    model = Model(url="url/of/model/includes/version-id")
    dataset = Dataset(dataset_id="dataset-id")
    y, y_pred, clss, input_protos = model.get_raw_eval(dataset, return_format="array")
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    report = classification_report(y, y_pred, target_names=clss)
    print(report)
    acc = accuracy_score(y, y_pred)
    print("acc ", acc)
    ```

    * Evaluate `visual-detector` using COCOeval

    ```python
    import os
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from clarifai.client.model import Model
    from clarifai.client.dataset import Dataset
    os.environ["CLARIFAI_PAT"] = "???" # Insert your PAT
    model = Model(url=model_url)
    dataset = Dataset(url=dataset_url)
    y, y_pred = model.get_raw_eval(dataset, return_format="coco")
    # save as files to load in COCO API
    def save_annot(d, path):
      with open(path, "w") as fp:
        json.dump(d, fp, indent=2)
    gt_path = os.path.join("gt.json")
    pred_path = os.path.join("pred.json")
    save_annot(y, gt_path)
    save_annot(y_pred, pred_path)

    cocoGt = COCO(gt_path)
    cocoPred = COCO(pred_path)
    cocoEval = COCOeval(cocoGt, cocoPred, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize() # Print out result of all classes with all area type
    # Example:
    # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.863
    # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.973
    # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.939
    # ...
    ```

    """
    from clarifai.utils.evaluation.testset_annotation_parser import (
        parse_eval_annotation_classifier, parse_eval_annotation_detector,
        parse_eval_annotation_detector_coco)

    valid_model_types = ["visual-classifier", "text-classifier", "visual-detector"]
    supported_format = ['proto', 'array', 'coco']
    assert return_format in supported_format, ValueError(
        f"Expected return_format in {supported_format}, got {return_format}")
    self.load_info()
    model_type_id = self.model_info.model_type_id
    assert model_type_id in valid_model_types, \
      f"This method only supports model types {valid_model_types}, but your model type is {self.model_info.model_type_id}."
    assert not (dataset and
                eval_id), "Using both `dataset` and `eval_id`, but only one should be passed."
    assert not dataset or not eval_id, "Please provide either `dataset` or `eval_id`, but nothing was passed."
    if model_type_id.endswith("-classifier") and return_format == "coco":
      raise ValueError(
          f"return_format coco only applies for `visual-detector`, however your model is `{model_type_id}`"
      )

    if dataset:
      eval_by_ds = self.get_eval_by_dataset(dataset)
      if len(eval_by_ds) == 0:
        raise Exception(f"Model is not valuated with dataset: {dataset}")
      eval_id = eval_by_ds[0].id

    detail_eval_data = self.get_eval_by_id(eval_id=eval_id, test_set=True, metrics_by_class=True)

    if return_format == "proto":
      return detail_eval_data.test_set
    else:
      if model_type_id.endswith("-classifier"):
        return parse_eval_annotation_classifier(detail_eval_data)
      elif model_type_id == "visual-detector":
        if return_format == "array":
          return parse_eval_annotation_detector(detail_eval_data)
        elif return_format == "coco":
          return parse_eval_annotation_detector_coco(detail_eval_data)

  def export(self, export_dir: str = None) -> None:
    """Export the model, stores the exported model as model.tar file

    Args:
        export_dir (str): The directory to save the exported model.

    Example:
        >>> from clarifai.client.model import Model
        >>> model = Model("url")
        >>> model.export('/path/to/export_model_dir')
    """
    assert self.model_info.model_version.id, "Model version ID is missing. Please provide a `model_version` with a valid `id` as an argument or as a URL in the following format: '{user_id}/{app_id}/models/{your_model_id}/model_version_id/{your_version_model_id}' when initializing."
    try:
      if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    except OSError as e:
      raise Exception(f"An error occurred while creating the directory: {e}")

    def _get_export_response():
      get_export_request = service_pb2.GetModelVersionExportRequest(
          user_app_id=self.user_app_id,
          model_id=self.id,
          version_id=self.model_info.model_version.id,
      )
      response = self._grpc_request(self.STUB.GetModelVersionExport, get_export_request)

      if response.status.code != status_code_pb2.SUCCESS and response.status.code != status_code_pb2.CONN_DOES_NOT_EXIST:
        raise Exception(response.status)

      return response

    def _download_exported_model(
        get_model_export_response: service_pb2.SingleModelVersionExportResponse,
        local_filepath: str):
      model_export_url = get_model_export_response.export.url
      model_export_file_size = get_model_export_response.export.size

      response = requests.get(model_export_url, stream=True)
      response.raise_for_status()

      with open(local_filepath, 'wb') as f:
        progress = tqdm(
            total=model_export_file_size, unit='B', unit_scale=True, desc="Exporting model")
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
          progress.update(len(chunk))
        progress.close()

      self.logger.info(
          f"Model ID {self.id} with version {self.model_info.model_version.id} exported successfully to {export_dir}/model.tar"
      )

    get_export_response = _get_export_response()
    if get_export_response.status.code == status_code_pb2.CONN_DOES_NOT_EXIST:
      put_export_request = service_pb2.PutModelVersionExportsRequest(
          user_app_id=self.user_app_id,
          model_id=self.id,
          version_id=self.model_info.model_version.id,
      )

      response = self._grpc_request(self.STUB.PutModelVersionExports, put_export_request)
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(response.status)

      self.logger.info(
          f"Model ID {self.id} with version {self.model_info.model_version.id} export started, please wait..."
      )
      time.sleep(5)
      start_time = time.time()
      backoff_iterator = BackoffIterator(10)
      while True:
        get_export_response = _get_export_response()
        if get_export_response.export.status.code == status_code_pb2.MODEL_EXPORTING and \
          time.time() - start_time < 60 * 30: # 30 minutes
          self.logger.info(
              f"Model ID {self.id} with version {self.model_info.model_version.id} is still exporting, please wait..."
          )
          time.sleep(next(backoff_iterator))
        elif get_export_response.export.status.code == status_code_pb2.MODEL_EXPORTED:
          _download_exported_model(get_export_response, os.path.join(export_dir, "model.tar"))
          break
        elif time.time() - start_time > 60 * 30:
          raise Exception(
              f"""Model Export took too long. Please try again or contact support@clarifai.com
              Req ID: {get_export_response.status.req_id}""")
    elif get_export_response.export.status.code == status_code_pb2.MODEL_EXPORTED:
      _download_exported_model(get_export_response, os.path.join(export_dir, "model.tar"))
