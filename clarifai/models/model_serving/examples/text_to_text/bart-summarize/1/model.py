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
"""Triton inference server Python Backend Model."""

import os
import sys

try:
  import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:
  pass
from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig


class TritonPythonModel:
  """
  Triton Python BE Model.
  """

  def initialize(self, args):
    """
    Triton server init.
    """
    args["model_repository"] = args["model_repository"].replace("/1/model.py", "")
    sys.path.append(os.path.dirname(__file__))
    from inference import InferenceModel

    self.inference_obj = InferenceModel()

    # Read input_name from config file
    self.config_msg = ModelConfig()
    with open(os.path.join(args["model_repository"], "config.pbtxt"), "r") as f:
      cfg = f.read()
    text_format.Merge(cfg, self.config_msg)
    self.input_name = [inp.name for inp in self.config_msg.input][0]

  def execute(self, requests):
    """
    Serve model inference requests.
    """
    responses = []

    for request in requests:
      in_batch = pb_utils.get_input_tensor_by_name(request, self.input_name)
      in_batch = in_batch.as_numpy()
      inference_response = self.inference_obj.get_predictions(in_batch)
      responses.append(inference_response)

    return responses
