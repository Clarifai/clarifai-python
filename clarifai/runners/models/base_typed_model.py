import itertools
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.service_pb2 import PostModelOutputsRequest
from google.protobuf import json_format

from ..utils.data_handler import InputDataHandler, OutputDataHandler
from .model_runner import ModelRunner


class AnyAnyModel(ModelRunner):

  def load_model(self):
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    raise NotImplementedError

  def parse_input_request(
      self, input_request: service_pb2.PostModelOutputsRequest) -> Tuple[List[Dict], Dict]:
    list_input_dict = [
        InputDataHandler.from_proto(input).to_python() for input in input_request.inputs
    ]
    inference_params = json_format.MessageToDict(
        input_request.model.model_version.output_info.params)

    return list_input_dict, inference_params

  def convert_output_to_proto(self, outputs: list):
    assert (isinstance(outputs, Iterator) or isinstance(outputs, list) or
            isinstance(outputs, tuple)), "outputs must be an Iterator"
    output_protos = []
    for output in outputs:
      if isinstance(output, OutputDataHandler):
        output = output.proto
      elif isinstance(output, resources_pb2.Output):
        pass
      else:
        raise NotImplementedError
      output_protos.append(output)

    return service_pb2.MultiOutputResponse(outputs=output_protos)

  def predict_wrapper(
      self, request: service_pb2.PostModelOutputsRequest) -> service_pb2.MultiOutputResponse:
    list_dict_input, inference_params = self.parse_input_request(request)
    outputs = self.predict(list_dict_input, inference_parameters=inference_params)
    return self.convert_output_to_proto(outputs)

  def generate_wrapper(
      self, request: PostModelOutputsRequest) -> Iterator[service_pb2.MultiOutputResponse]:
    list_dict_input, inference_params = self.parse_input_request(request)
    outputs = self.generate(list_dict_input, inference_parameters=inference_params)
    for output in outputs:
      yield self.convert_output_to_proto(output)

  def _preprocess_stream(
      self, request: Iterator[PostModelOutputsRequest]) -> Iterator[Tuple[List[Dict], List[Dict]]]:
    """Return generator of processed data (from proto to python) and inference parameters like predict and generate"""
    for i, req in enumerate(request):
      input_data, _ = self.parse_input_request(req)
      yield input_data

  def stream_wrapper(self, request: Iterator[PostModelOutputsRequest]
                    ) -> Iterator[service_pb2.MultiOutputResponse]:
    first_request = next(request)
    _, inference_params = self.parse_input_request(first_request)
    request_iterator = itertools.chain([first_request], request)
    outputs = self.stream(self._preprocess_stream(request_iterator), inference_params)
    for output in outputs:
      yield self.convert_output_to_proto(output)

  def predict(self, input_data: List[Dict],
              inference_parameters: Dict[str, Any] = {}) -> List[OutputDataHandler]:
    """
    Prediction method.

    Args:
    -----
    - input_data: is list of dict where key is input type name.
      * image: np.ndarray
      * text: str
      * audio: bytes

    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Returns:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError

  def generate(self, input_data: List[Dict],
               inference_parameters: Dict[str, Any] = {}) -> Iterator[List[OutputDataHandler]]:
    """
    Generate method.

    Args:
    -----
    - input_data: is list of dict where key is input type name.
      * image: np.ndarray
      * text: str
      * audio: bytes

    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Yield:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError

  def stream(self, inputs: Iterator[List[Dict[str, Any]]],
             inference_params: Dict[str, Any]) -> Iterator[List[OutputDataHandler]]:
    """
    Stream method.

    Args:
    -----
    input_request: is an Iterator of Tuple which
    - First element (List[Dict[str, Union[np.ndarray, str, bytes]]]) is list of dict input data type which keys and values are:
        * image: np.ndarray
        * text: str
        * audio: bytes

    - Second element (Dict[str, Union[bool, str, float, int]]): is a dict of inference_parameters

    Yield:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError


class VisualInputModel(AnyAnyModel):

  def parse_input_request(
      self, input_request: service_pb2.PostModelOutputsRequest) -> Tuple[List[Dict], Dict]:
    list_input_dict = [
        InputDataHandler.from_proto(input).image(format="np") for input in input_request.inputs
    ]
    inference_params = json_format.MessageToDict(
        input_request.model.model_version.output_info.params)

    return list_input_dict, inference_params

  def load_model(self):
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    raise NotImplementedError

  def predict(self, input_data: List[np.ndarray],
              inference_parameters: Dict[str, Any] = {}) -> List[OutputDataHandler]:
    """
    Prediction method.

    Args:
    -----
    - input_data(List[np.ndarray]): is list of image as np.ndarray type
    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Returns:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError


class TextInputModel(AnyAnyModel):

  def load_model(self):
    """
    Load inference time artifacts that are called frequently .e.g. models, tokenizers, etc.
    in this method so they are loaded only once for faster inference.
    """
    raise NotImplementedError

  def parse_input_request(
      self, input_request: service_pb2.PostModelOutputsRequest) -> Tuple[List[Dict], Dict]:
    list_input_text = [InputDataHandler.from_proto(input).text for input in input_request.inputs]
    inference_params = json_format.MessageToDict(
        input_request.model.model_version.output_info.params)

    return list_input_text, inference_params

  def predict(self, input_data: List[str],
              inference_parameters: Dict[str, Any] = {}) -> List[OutputDataHandler]:
    """
    Prediction method.

    Args:
    -----
    - input_data(List[str]): is list of text as str type
    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Returns:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError

  def generate(self, input_data: List[str],
               inference_parameters: Dict[str, Any] = {}) -> Iterator[List[OutputDataHandler]]:
    """
    Prediction method.

    Args:
    -----
    - input_data(List[str]): is list of text as str type
    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Yield:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError

  def stream(self, inputs: Iterator[List[str]],
             inference_params: Dict[str, Any]) -> Iterator[List[OutputDataHandler]]:
    """
    Stream method.

    Args:
    -----
    input_request: is an Iterator of Tuple which
    - First element (List[str]) is list of input text:
    - Second element (Dict[str, Union[bool, str, float, int]]): is a dict of inference_parameters

    Yield:
    --------
      List of OutputDataHandler
    """
    raise NotImplementedError
