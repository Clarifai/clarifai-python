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
from clarifai.models.model_serving.model_config.inference_parameter import parse_req_parameters


class TritonPythonModel:
  """
  Triton Python BE Model.
  """

  def initialize(self, args):
    """
    Triton server init.
    """
    sys.path.append(os.path.dirname(__file__))
    from inference import InferenceModel

    self.inference_obj = InferenceModel()

    # Read input_name from config file
    self.input_names = [inp.name for inp in self.inference_obj.config.serving_backend.triton.input]

  def execute(self, requests):
    """
    Serve model inference requests.
    """
    responses = []

    for request in requests:
      try:
        parameters = request.parameters()
      except Exception:
        print(
            "It seems this triton version does not support `parameters()` in request. "
            "Please upgrade tritonserver version otherwise can not use `inference_parameters`. Error message: {e}"
        )
        parameters = None

      parameters = parse_req_parameters(parameters) if parameters else {}

      if len(self.input_names) == 1:
        in_batch = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
        in_batch = in_batch.as_numpy()
        data = in_batch
      else:
        data = {}
        for input_name in self.input_names:
          in_batch = pb_utils.get_input_tensor_by_name(request, input_name)
          in_batch = in_batch.as_numpy() if in_batch is not None else []
          data.update({input_name: in_batch})

      inference_response = self.inference_obj._tritonserver_predict(
          input_data=data, inference_parameters=parameters)
      responses.append(inference_response)

    return responses
