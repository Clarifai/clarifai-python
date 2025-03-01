import time
from typing import Any, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.constants.model import MAX_MODEL_PREDICT_INPUTS
from clarifai.errors import UserError
from clarifai.utils.misc import BackoffIterator, status_is_retryable


class ModelClient:
  '''
  Client for calling model predict, generate, and stream methods.
  '''

  def __init__(self, stub, request_template: service_pb2.PostModelOutputsRequest = None):
    '''
        Initialize the model client.

        Args:
            stub: The gRPC stub for the model.
            request_template: The template for the request to send to the model, including
            common fields like model_id, model_version, cluster, etc.
        '''
    self.STUB = stub
    self.request_template = request_template or service_pb2.PostModelOutputsRequest()
    self._method_signatures = None

  def _fetch_signatures(self):
    '''
      Fetch the method signatures from the model.

      Returns:
          Dict: The method signatures.
      '''
    if self._method_signatures is not None:
      return
    #request = resources_pb2.GetModelSignaturesRequest()
    #response = self.stub.GetModelSignatures(request)
    #self._method_signatures = json.loads(response.signatures)  # or define protos
    # TODO this could use a new endpoint to get the signatures
    # for local grpc models, we'll also have to add the endpoint to the model servicer
    # for now we'll just use the predict endpoint with a special method name

    # TODO need to move location of this to avoid circular import
    from clarifai.runners.utils.method_signatures import signatures_from_json
    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)
    request.model.model_version.output_info.params['_method_name'] = '_GET_SIGNATURES'
    request.inputs.add()  # empty input for this method
    response = self.STUB.PostModelOutputs(request)
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    method_signatures = signatures_from_json(response.outputs[0].data.string_value)
    self._method_signatures = {method.name: method for method in method_signatures}

  def predict(
      self,
      inputs,  # TODO set up functions according to fetched signatures?
      method_name: str = 'predict',
  ) -> Any:
    # TODO need to move location of this to avoid circular import
    from clarifai.runners.utils.method_signatures import deserialize, serialize
    self._fetch_signatures()
    input_signature = self._method_signatures[method_name].input_variables
    output_signature = self._method_signatures[method_name].output_variables

    batch_input = True
    if isinstance(inputs, dict):
      inputs = [inputs]
      batch_input = False
    if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
      raise UserError(f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}.")

    proto_inputs = []
    for input in inputs:
      proto = resources_pb2.Input()
      serialize(input, input_signature, proto.data)
      proto_inputs.append(proto)

    response = self._predict_by_proto(proto_inputs, method_name)

    outputs = []
    for output in response.outputs:
      outputs.append(deserialize(output.data, output_signature))
    if batch_input:
      return outputs
    return outputs[0]

  def _predict_by_proto(
      self,
      inputs: List[resources_pb2.Input],
      method_name: str,
  ) -> service_pb2.MultiOutputResponse:
    """Predicts the model based on the given inputs.

      Args:
          inputs (List[resources_pb2.Input]): The inputs to predict.
          compute_cluster_id (str): The compute cluster ID to use for the model.
          nodepool_id (str): The nodepool ID to use for the model.
          deployment_id (str): The deployment ID to use for the model.
          user_id (str): The user ID to use for nodepool or deployment.
          inference_params (Dict): Inference parameters to override.
          output_config (Dict): Output configuration to override.

      Returns:
          service_pb2.MultiOutputResponse: The prediction response(s).
      """
    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)
    request.model.model_version.output_info.params['_method_name'] = method_name
    request.inputs.extend(inputs)

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    while True:
      response = self.STUB.PostModelOutputs(request)
      if status_is_retryable(
          response.status.code) and time.time() - start_time < 60 * 10:  # 10 minutes
        self.logger.info(f"{self.id} model predict failed with response {response.status!r}")
        time.sleep(next(backoff_iterator))
        continue

      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model Predict failed with response {response.status!r}")
      break

    return response
