import time
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.constants.model import MAX_MODEL_PREDICT_INPUTS
from clarifai.errors import UserError
from clarifai.runners.utils.method_signatures import deserialize, serialize, signatures_from_json
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
    self._fetch_signatures()
    self._define_functions()

  def _fetch_signatures(self):
    '''
      Fetch the method signatures from the model.

      Returns:
          Dict: The method signatures.
      '''
    #request = resources_pb2.GetModelSignaturesRequest()
    #response = self.stub.GetModelSignatures(request)
    #self._method_signatures = json.loads(response.signatures)  # or define protos
    # TODO this could use a new endpoint to get the signatures
    # for local grpc models, we'll also have to add the endpoint to the model servicer
    # for now we'll just use the predict endpoint with a special method name

    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)
    request.model.model_version.output_info.params['_method_name'] = '_GET_SIGNATURES'
    request.inputs.add()  # empty input for this method
    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    while True:
      response = self.STUB.PostModelOutputs(request)
      if status_is_retryable(
          response.status.code) and time.time() - start_time < 60 * 10:  # 10 minutes
        self.logger.info(f"Retrying model info fetch with response {response.status!r}")
        time.sleep(next(backoff_iterator))
        continue

      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model failed with response {response.status!r}")
      break
    if response.status.code != status_code_pb2.SUCCESS:
      raise Exception(response.status)
    self._method_signatures = signatures_from_json(response.outputs[0].data.string_value)

  def _define_functions(self):
    '''
    Define the functions based on the method signatures.
    '''
    for method_name, method_signature in self._method_signatures.items():
      # define the function in this client instance
      input_vars = method_signature.inputs
      output_vars = method_signature.outputs
      if method_signature.method_type == 'predict':
        call_func = self._predict
      elif method_signature.method_type == 'generate':
        call_func = self._generate
      elif method_signature.method_type == 'stream':
        call_func = self._stream
      else:
        raise ValueError(f"Unknown method type {method_signature.method_type}")

      def bind_f(method_name, call_func, input_vars, output_vars):

        def f(*args, **kwargs):
          for var, arg in zip(input_vars, args):  # handle positional with zip shortest
            kwargs[var.name] = arg
          return call_func(kwargs, method_name)

        return f

      # need to bind method_name to the value, not the mutating loop variable
      f = bind_f(method_name, call_func, input_vars, output_vars)

      # set names and docstrings
      # note we could also have used exec with strings from the signature to define the
      # function, but this is safer (no xss), and docstrings with the signature is ok enough
      f.__name__ = method_name
      f.__qualname__ = f'{self.__class__.__name__}.{method_name}'
      input_spec = ', '.join(
          f'{var.name}: {var.data_type}{" = " + var.default if not var.required else ""}'
          for var in input_vars)
      if len(output_vars) == 1 and output_vars[0].name == 'return':
        output_spec = output_vars[0].data_type
      else:
        output_spec = f'Output({", ".join(f"{var.name}={var.data_type}" for var in output_vars)})'
      f.__doc__ = f'''{method_name}(self, {input_spec}) -> {output_spec}\n'''
      #f.__doc__ += method_signature.description  # TODO
      setattr(self, method_name, f)

  def _predict(
      self,
      inputs,  # TODO set up functions according to fetched signatures?
      method_name: str = 'predict',
  ) -> Any:
    input_signature = self._method_signatures[method_name].inputs
    output_signature = self._method_signatures[method_name].outputs

    batch_input = True
    if isinstance(inputs, dict):
      inputs = [inputs]
      batch_input = False

    proto_inputs = []
    for input in inputs:
      proto = resources_pb2.Input()
      serialize(input, input_signature, proto.data)
      proto_inputs.append(proto)

    response = self._predict_by_proto(proto_inputs, method_name)
    #print(response)

    outputs = []
    for output in response.outputs:
      outputs.append(deserialize(output.data, output_signature))
    if batch_input:
      return outputs
    return outputs[0]

  def _predict_by_proto(
      self,
      inputs: List[resources_pb2.Input],
      method_name: str = None,
      inference_params: Dict = None,
      output_config: Dict = None,
  ) -> service_pb2.MultiOutputResponse:
    """Predicts the model based on the given inputs.

      Args:
          inputs (List[resources_pb2.Input]): The inputs to predict.
          method_name (str): The remote method name to call.
          inference_params (Dict): Inference parameters to override.
          output_config (Dict): Output configuration to override.

      Returns:
          service_pb2.MultiOutputResponse: The prediction response(s).
      """
    if not isinstance(inputs, list):
      raise UserError('Invalid inputs, inputs must be a list of Input objects.')
    if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
      raise UserError(f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}.")

    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)

    request.inputs.extend(inputs)

    if method_name:
      # TODO put in new proto field?
      request.model.model_version.output_info.params['_method_name'] = method_name
    if inference_params:
      request.model.model_version.output_info.params.update(inference_params)
    if output_config:
      request.model.model_version.output_info.output_config.MergeFromDict(output_config)

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    while True:
      response = self.STUB.PostModelOutputs(request)
      if status_is_retryable(
          response.status.code) and time.time() - start_time < 60 * 10:  # 10 minutes
        self.logger.info(f"Model predict failed with response {response.status!r}")
        time.sleep(next(backoff_iterator))
        continue

      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model Predict failed with response {response.status!r}")
      break

    return response

  def _generate(
      self,
      inputs,  # TODO set up functions according to fetched signatures?
      method_name: str = 'generate',
  ) -> Any:
    input_signature = self._method_signatures[method_name].inputs
    output_signature = self._method_signatures[method_name].outputs

    batch_input = True
    if isinstance(inputs, dict):
      inputs = [inputs]
      batch_input = False

    proto_inputs = []
    for input in inputs:
      proto = resources_pb2.Input()
      serialize(input, input_signature, proto.data)
      proto_inputs.append(proto)

    response_stream = self._generate_by_proto(proto_inputs, method_name)
    #print(response)

    for response in response_stream:
      outputs = []
      for output in response.outputs:
        outputs.append(deserialize(output.data, output_signature))
      if batch_input:
        yield outputs
      yield outputs[0]

  def _generate_by_proto(
      self,
      inputs: List[resources_pb2.Input],
      method_name: str = None,
      inference_params: Dict = {},
      output_config: Dict = {},
  ):
    """Generate the stream output on model based on the given inputs.

    Args:
        inputs (list[Input]): The inputs to generate, must be less than 128.
        method_name (str): The remote method name to call.
        inference_params (dict): The inference params to override.
        output_config (dict): The output config to override.
    """
    if not isinstance(inputs, list):
      raise UserError('Invalid inputs, inputs must be a list of Input objects.')
    if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
      raise UserError(f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}."
                     )  # TODO Use Chunker for inputs len > 128

    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)

    request.inputs.extend(inputs)

    if method_name:
      # TODO put in new proto field?
      request.model.model_version.output_info.params['_method_name'] = method_name
    if inference_params:
      request.model.model_version.output_info.params.update(inference_params)
    if output_config:
      request.model.model_version.output_info.output_config.MergeFromDict(output_config)

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    started = False
    while not started:
      stream_response = self.STUB.GenerateModelOutputs(request)
      try:
        response = next(stream_response)  # get the first response
      except StopIteration:
        raise Exception("Model Generate failed with no response")
      if status_is_retryable(response.status.code) and \
              time.time() - start_time < 60 * 10:
        self.logger.info("Model is still deploying, please wait...")
        time.sleep(next(backoff_iterator))
        continue
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model Generate failed with response {response.status!r}")
      started = True

    yield response  # yield the first response

    for response in stream_response:
      if response.status.code != status_code_pb2.SUCCESS:
        raise Exception(f"Model Generate failed with response {response.status!r}")
      yield response

  def _req_iterator(self,
                    input_iterator: Iterator[List[resources_pb2.Input]],
                    method_name: str = None,
                    inference_params: Dict = {},
                    output_config: Dict = {}):
    request = service_pb2.PostModelOutputsRequest()
    request.CopyFrom(self.request_template)
    request.model.model_version.output_info.params['_method_name'] = method_name
    if inference_params:
      request.model.model_version.output_info.params.update(inference_params)
    if output_config:
      request.model.model_version.output_info.output_config.MergeFromDict(output_config)
    for inputs in input_iterator:
      req = service_pb2.PostModelOutputsRequest()
      req.CopyFrom(request)
      req.inputs.extend(inputs)
      yield req

  def _stream_by_proto(self,
                       inputs: Iterator[List[resources_pb2.Input]],
                       method_name: str = None,
                       inference_params: Dict = {},
                       output_config: Dict = {}):
    """Generate the stream output on model based on the given stream of inputs.
    """
    # if not isinstance(inputs, Iterator[List[Input]]):
    #   raise UserError('Invalid inputs, inputs must be a iterator of list of Input objects.')

    request = self._req_iterator(inputs, method_name, inference_params, output_config)

    start_time = time.time()
    backoff_iterator = BackoffIterator(10)
    generation_started = False
    while True:
      if generation_started:
        break
      stream_response = self.STUB.StreamModelOutputs(request)
      for response in stream_response:
        if status_is_retryable(response.status.code) and \
                time.time() - start_time < 60 * 10:
          self.logger.info("Model is still deploying, please wait...")
          time.sleep(next(backoff_iterator))
          break
        if response.status.code != status_code_pb2.SUCCESS:
          raise Exception(f"Model Predict failed with response {response.status!r}")
        else:
          if not generation_started:
            generation_started = True
          yield response
