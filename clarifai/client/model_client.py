import asyncio
import inspect
import time
from typing import Any, Dict, Iterator, List

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.auth.register import V2Stub
from clarifai.constants.model import MAX_MODEL_PREDICT_INPUTS
from clarifai.errors import UserError
from clarifai.runners.utils import code_script, method_signatures
from clarifai.runners.utils.method_signatures import (
    CompatibilitySerializer,
    deserialize,
    get_stream_from_signature,
    serialize,
    signatures_from_json,
)
from clarifai.runners.utils.openai_convertor import is_openai_chat_format
from clarifai.utils.logging import logger
from clarifai.utils.misc import BackoffIterator, status_is_retryable


def is_async_context():
    """Check if code is running in an async context."""
    try:
        asyncio.get_running_loop()
        import sys

        # In Jupyter, to check if we're actually in an async cell. Becaue by default jupyter considers it as async.
        if 'ipykernel' in sys.modules:
            return False
        return True
    except RuntimeError:
        return False


class ModelClient:
    '''
    Client for calling model predict, generate, and stream methods.
    '''

    def __init__(
        self,
        stub,
        async_stub: V2Stub = None,
        request_template: service_pb2.PostModelOutputsRequest = None,
    ):
        '''
        Initialize the model client.

        Args:
            stub: The gRPC stub for the model.
            request_template: The template for the request to send to the model, including
            common fields like model_id, model_version, cluster, etc.
        '''
        self.STUB = stub
        self.async_stub = async_stub
        self.request_template = request_template or service_pb2.PostModelOutputsRequest()
        self._method_signatures = None
        self._defined = False

    def fetch(self):
        '''
        Fetch function signature definitions from the model and define the functions in the client
        '''
        if self._defined:
            return
        try:
            self._fetch_signatures()
            self._define_functions()
        finally:
            self._defined = True

    def __getattr__(self, name):
        if not self._defined:
            self.fetch()
        try:
            return self.__getattribute__(name)
        except AttributeError as e:
            # Provide helpful error message with available methods
            available_methods = []
            if self._method_signatures:
                available_methods = list(self._method_signatures.keys())

            error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"

            if available_methods:
                error_msg += f". Available methods: {available_methods}"
                raise AttributeError(error_msg) from e
            else:
                error_msg += ". This model is a non-pythonic model. Please use the old inference methods i.e. predict_by_url, predict_by_bytes, etc."
                raise Exception(error_msg) from e

    def _fetch_signatures(self):
        '''
        Fetch the method signatures from the model.

        Returns:
            Dict: The method signatures.
        '''
        try:
            response = self.STUB.GetModelVersion(
                service_pb2.GetModelVersionRequest(
                    user_app_id=self.request_template.user_app_id,
                    model_id=self.request_template.model_id,
                    version_id=self.request_template.version_id,
                )
            )

            method_signatures = None
            if response.status.code == status_code_pb2.SUCCESS:
                method_signatures = response.model_version.method_signatures
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Model failed with response {response!r}")
            self._method_signatures = {}
            for method_signature in method_signatures:
                method_name = method_signature.name
                # check for duplicate method names
                if method_name in self._method_signatures:
                    raise ValueError(f"Duplicate method name {method_name}")
                self._method_signatures[method_name] = method_signature
            if not self._method_signatures:  # if no method signatures, try to fetch from the model
                self._fetch_signatures_backup()
        except Exception:
            # try to fetch from the model
            self._fetch_signatures_backup()
            if not self._method_signatures:
                raise ValueError("Failed to fetch method signatures from model and backup method")

    def _fetch_signatures_backup(self):
        '''
        This is a temporary method of fetching the method signatures from the model.

        Returns:
            Dict: The method signatures.
        '''

        request = service_pb2.PostModelOutputsRequest()
        request.CopyFrom(self.request_template)
        # request.model.model_version.output_info.params['_method_name'] = '_GET_SIGNATURES'
        inp = request.inputs.add()  # empty input for this method
        inp.data.parts.add()  # empty part for this input
        inp.data.metadata['_method_name'] = '_GET_SIGNATURES'
        start_time = time.time()
        backoff_iterator = BackoffIterator(10)
        while True:
            response = self.STUB.PostModelOutputs(request)
            if (
                status_is_retryable(response.status.code) and time.time() - start_time < 60 * 10
            ):  # 10 minutes
                logger.info(f"Retrying model info fetch with response {response.status!r}")
                time.sleep(next(backoff_iterator))
                continue
            break
        if response.status.code == status_code_pb2.INPUT_UNSUPPORTED_FORMAT or (
            response.status.code == status_code_pb2.SUCCESS
            and response.outputs[0].data.text.raw == ''
        ):
            # return codes/values from older models that don't support _GET_SIGNATURES
            self._method_signatures = {}
            self._define_compatability_functions()
            return
        if response.status.code != status_code_pb2.SUCCESS:
            if response.outputs[0].status.description.startswith("cannot identify image file"):
                raise Exception(
                    "Failed to fetch method signatures from model and backup method. This model is a non-pythonic model. Please use the old inference methods i.e. predict_by_url, predict_by_bytes, etc."
                )
            raise Exception(f"Model failed with response {response!r}")
        self._method_signatures = signatures_from_json(response.outputs[0].data.text.raw)

    def _define_functions(self):
        '''
        Define the functions based on the method signatures.
        '''
        for method_name, method_signature in self._method_signatures.items():
            # define the function in this client instance
            if resources_pb2.RunnerMethodType.Name(method_signature.method_type) == 'UNARY_UNARY':
                call_func = self._predict
                async_call_func = self._async_predict
            elif (
                resources_pb2.RunnerMethodType.Name(method_signature.method_type)
                == 'UNARY_STREAMING'
            ):
                call_func = self._generate
                async_call_func = self._async_generate
            elif (
                resources_pb2.RunnerMethodType.Name(method_signature.method_type)
                == 'STREAMING_STREAMING'
            ):
                call_func = self._stream
                async_call_func = self._async_stream
            else:
                raise ValueError(f"Unknown method type {method_signature.method_type}")

            # method argnames, in order, collapsing nested keys to corresponding user function args
            method_argnames = []
            for var in method_signature.input_fields:
                outer = var.name.split('.', 1)[0]
                if outer in method_argnames:
                    continue
                method_argnames.append(outer)

            def bind_f(method_name, method_argnames, call_func, async_call_func):
                def sync_f(*args, **kwargs):
                    if len(args) > len(method_argnames):
                        raise TypeError(
                            f"{method_name}() takes {len(method_argnames)} positional arguments but {len(args)} were given"
                        )

                    if len(args) + len(kwargs) > len(method_argnames):
                        raise TypeError(
                            f"{method_name}() got an unexpected keyword argument {next(iter(kwargs))}"
                        )
                    if len(args) == 1 and (not kwargs) and isinstance(args[0], list):
                        batch_inputs = args[0]
                        # Validate each input is a dictionary
                        is_batch_input_valid = all(
                            isinstance(input, dict) for input in batch_inputs
                        )
                        if is_batch_input_valid and (not is_openai_chat_format(batch_inputs)):
                            # If the batch input is valid, call the function with the batch inputs and the method name
                            return call_func(batch_inputs, method_name)

                    for name, arg in zip(
                        method_argnames, args
                    ):  # handle positional with zip shortest
                        if name in kwargs:
                            raise TypeError(f"Multiple values for argument {name}")
                        kwargs[name] = arg
                    return call_func(kwargs, method_name)

                async def async_f(*args, **kwargs):
                    # Async version to call the async function
                    if len(args) > len(method_argnames):
                        raise TypeError(
                            f"{method_name}() takes {len(method_argnames)} positional arguments but {len(args)} were given"
                        )
                    if len(args) + len(kwargs) > len(method_argnames):
                        raise TypeError(
                            f"{method_name}() got an unexpected keyword argument {next(iter(kwargs))}"
                        )
                    if len(args) == 1 and (not kwargs) and isinstance(args[0], list):
                        batch_inputs = args[0]
                        # Validate each input is a dictionary
                        is_batch_input_valid = all(
                            isinstance(input, dict) for input in batch_inputs
                        )
                        if is_batch_input_valid and (not is_openai_chat_format(batch_inputs)):
                            # If the batch input is valid, call the function with the batch inputs and the method name
                            return async_call_func(batch_inputs, method_name)

                    for name, arg in zip(
                        method_argnames, args
                    ):  # handle positional with zip shortest
                        if name in kwargs:
                            raise TypeError(f"Multiple values for argument {name}")
                        kwargs[name] = arg

                    return async_call_func(kwargs, method_name)

                class MethodWrapper:
                    def __call__(self, *args, **kwargs):
                        if is_async_context():
                            return async_f(*args, **kwargs)
                        return sync_f(*args, **kwargs)

                return MethodWrapper()

            # need to bind method_name to the value, not the mutating loop variable
            f = bind_f(method_name, method_argnames, call_func, async_call_func)

            # set names, annotations and docstrings
            f.__name__ = method_name
            f.__qualname__ = f'{self.__class__.__name__}.{method_name}'
            input_annotations = code_script._get_annotations_source(method_signature)
            return_annotation = input_annotations.pop('return', (None, None, None))[0]
            sig = inspect.signature(f).replace(
                parameters=[
                    inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v[0])
                    for k, v in input_annotations.items()
                ],
                return_annotation=return_annotation,
            )
            f.__signature__ = sig
            f.__doc__ = method_signature.description
            setattr(self, method_name, f)

    def available_methods(self) -> List[str]:
        """Get the available methods for this model.

        Returns:
            List[str]: The available methods.
        """
        if not self._defined:
            self.fetch()
        return self._method_signatures.keys()

    def method_signature(self, method_name: str) -> str:
        """Get the method signature for a method.

        Args:
            method_name (str): The name of the method.

        Returns:
            str: The method signature.
        """
        if not self._defined:
            self.fetch()
        return method_signatures.get_method_signature(self._method_signatures[method_name])

    def generate_client_script(
        self,
        base_url: str = None,
        use_ctx: bool = False,
    ) -> str:
        """Generate a client script for this model.

        Returns:
            str: The client script.
        """
        if not self._defined:
            self.fetch()
        method_signatures = []
        for _, method_signature in self._method_signatures.items():
            method_signatures.append(method_signature)
        return code_script.generate_client_script(
            method_signatures,
            user_id=self.request_template.user_app_id.user_id,
            app_id=self.request_template.user_app_id.app_id,
            model_id=self.request_template.model_id,
            base_url=base_url,
            deployment_id=self.request_template.runner_selector.deployment.id,
            compute_cluster_id=self.request_template.runner_selector.nodepool.compute_cluster.id,
            nodepool_id=self.request_template.runner_selector.nodepool.id,
            use_ctx=use_ctx,
        )

    def _define_compatability_functions(self):
        serializer = CompatibilitySerializer()

        def predict(input: Any) -> Any:
            proto = resources_pb2.Input()
            serializer.serialize(proto.data, input)
            # always use text.raw for compat
            if proto.data.string_value:
                proto.data.text.raw = proto.data.string_value
                proto.data.string_value = ''
            response = self._predict_by_proto([proto])
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Model predict failed with response {response!r}")
            response_data = response.outputs[0].data
            if response_data.text.raw:
                response_data.string_value = response_data.text.raw
                response_data.text.raw = ''
            return serializer.deserialize(response_data)

        self.predict = predict

    def _predict(
        self,
        inputs,  # TODO set up functions according to fetched signatures?
        method_name: str = 'predict',
    ) -> Any:
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

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

        outputs = []
        for output in response.outputs:
            outputs.append(deserialize(output.data, output_signature, is_output=True))
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
            for inp in request.inputs:
                inp.data.metadata['_method_name'] = method_name
        if inference_params:
            request.model.model_version.output_info.params.update(inference_params)
        if output_config:
            request.model.model_version.output_info.output_config.MergeFrom(
                resources_pb2.OutputConfig(**output_config)
            )

        start_time = time.time()
        backoff_iterator = BackoffIterator(10)
        while True:
            response = self.STUB.PostModelOutputs(request)
            if (
                status_is_retryable(response.status.code) and time.time() - start_time < 60 * 10
            ):  # 10 minutes
                logger.info("Model is still deploying, please wait...")
                time.sleep(next(backoff_iterator))
                continue

            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Model predict failed with response {response!r}")
            break
        return response

    async def _async_predict(
        self,
        inputs,
        method_name: str = 'predict',
    ) -> Any:
        """Asynchronously process inputs and make predictions.

        Args:
            inputs: Input data to process
            method_name (str): Name of the method to call

        Returns:
            Processed prediction results
        """
        # method_name is set to 'predict' by default, this is because to replicate the input and output signature behaviour of sync to async predict.
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

        batch_input = True
        if isinstance(inputs, dict):
            inputs = [inputs]
            batch_input = False

        proto_inputs = []
        for input in inputs:
            proto = resources_pb2.Input()
            serialize(input, input_signature, proto.data)
            proto_inputs.append(proto)
        response = await self._async_predict_by_proto(proto_inputs, method_name)
        outputs = []
        for output in response.outputs:
            outputs.append(deserialize(output.data, output_signature, is_output=True))

        return outputs if batch_input else outputs[0]

    async def _async_predict_by_proto(
        self,
        inputs: List[resources_pb2.Input],
        method_name: str = None,
        inference_params: Dict = None,
        output_config: Dict = None,
    ) -> service_pb2.MultiOutputResponse:
        """Asynchronously predicts the model based on the given inputs.

        Args:
            inputs (List[resources_pb2.Input]): The inputs to predict.
            method_name (str): The remote method name to call.
            inference_params (Dict): Inference parameters to override.
            output_config (Dict): Output configuration to override.

        Returns:
            service_pb2.MultiOutputResponse: The prediction response(s).

        Raises:
            UserError: If inputs are invalid or exceed maximum limit.
            Exception: If the model prediction fails.
        """
        if not isinstance(inputs, list):
            raise UserError('Invalid inputs, inputs must be a list of Input objects.')
        if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
            raise UserError(f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}.")

        request = service_pb2.PostModelOutputsRequest()
        request.CopyFrom(self.request_template)
        request.inputs.extend(inputs)

        if method_name:
            for inp in request.inputs:
                inp.data.metadata['_method_name'] = method_name
        if inference_params:
            request.model.model_version.output_info.params.update(inference_params)
        if output_config:
            request.model.model_version.output_info.output_config.MergeFrom(
                resources_pb2.OutputConfig(**output_config)
            )

        start_time = time.time()
        backoff_iterator = BackoffIterator(10)

        while True:
            try:
                response = await self.async_stub.PostModelOutputs(request)

                if (
                    status_is_retryable(response.status.code)
                    and time.time() - start_time < 60 * 10
                ):  # 10 minutes
                    logger.info("Model is still deploying, please wait...")
                    await asyncio.sleep(next(backoff_iterator))
                    continue

                if response.status.code != status_code_pb2.SUCCESS:
                    raise Exception(f"Model predict failed with response {response!r}")

                return response

            except Exception as e:
                if time.time() - start_time >= 10 * 1:  # 10 minutes timeout
                    raise Exception("Model prediction timed out after 10 minutes") from e
                logger.error(f"Error during prediction: {e}")
                await asyncio.sleep(next(backoff_iterator))
                continue

    def _generate(
        self,
        inputs,  # TODO set up functions according to fetched signatures?
        method_name: str = 'generate',
    ) -> Any:
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

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

        for response in response_stream:
            outputs = []
            for output in response.outputs:
                outputs.append(deserialize(output.data, output_signature, is_output=True))
            if batch_input:
                yield outputs
            else:
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
            raise UserError(
                f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}."
            )  # TODO Use Chunker for inputs len > 128

        request = service_pb2.PostModelOutputsRequest()
        request.CopyFrom(self.request_template)

        request.inputs.extend(inputs)

        if method_name:
            # TODO put in new proto field?
            for inp in request.inputs:
                inp.data.metadata['_method_name'] = method_name
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
            if status_is_retryable(response.status.code) and time.time() - start_time < 60 * 10:
                logger.info("Model is still deploying, please wait...")
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

    async def _async_generate(
        self,
        inputs,
        method_name: str = 'generate',
    ) -> Any:
        # method_name is set to 'generate' by default, this is because to replicate the input and output signature behaviour of sync to async generate.
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

        batch_input = True
        if isinstance(inputs, dict):
            inputs = [inputs]
            batch_input = False

        proto_inputs = []
        for input in inputs:
            proto = resources_pb2.Input()
        serialize(input, input_signature, proto.data)
        proto_inputs.append(proto)
        response_stream = self._async_generate_by_proto(proto_inputs, method_name)

        async for response in response_stream:
            outputs = []
            for output in response.outputs:
                outputs.append(deserialize(output.data, output_signature, is_output=True))
            if batch_input:
                yield outputs
            else:
                yield outputs[0]

    async def _async_generate_by_proto(
        self,
        inputs: List[resources_pb2.Input],
        method_name: str = None,
        inference_params: Dict = {},
        output_config: Dict = {},
    ):
        """Generate the async stream output on model based on the given inputs.

        Args:
            inputs (list[Input]): The inputs to generate, must be less than 128.
            method_name (str): The remote method name to call.
            inference_params (dict): The inference params to override.
            output_config (dict): The output config to override.
        """
        if not isinstance(inputs, list):
            raise UserError('Invalid inputs, inputs must be a list of Input objects.')
        if len(inputs) > MAX_MODEL_PREDICT_INPUTS:
            raise UserError(
                f"Too many inputs. Max is {MAX_MODEL_PREDICT_INPUTS}."
            )  # TODO Use Chunker for inputs len > 128

        request = service_pb2.PostModelOutputsRequest()
        request.CopyFrom(self.request_template)

        request.inputs.extend(inputs)

        if method_name:
            # TODO put in new proto field?
            for inp in request.inputs:
                inp.data.metadata['_method_name'] = method_name
        if inference_params:
            request.model.model_version.output_info.params.update(inference_params)
        if output_config:
            request.model.model_version.output_info.output_config.MergeFromDict(output_config)

        start_time = time.time()
        backoff_iterator = BackoffIterator(10)
        started = False
        while not started:
            # stream response returns gRPC async iterable - UnaryStreamCall
            stream_response = self.async_stub.GenerateModelOutputs(request)
            stream_resp = await stream_response  # get the async iterable
            iterator = stream_resp.__aiter__()  # get the async iterator for the response
            try:
                response = await iterator.__anext__()  # getting the first response
            except StopAsyncIteration:
                raise Exception("Model Generate failed with no response")
            if status_is_retryable(response.status.code) and time.time() - start_time < 60 * 10:
                logger.info("Model is still deploying, please wait...")
                await asyncio.sleep(next(backoff_iterator))
                continue
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Model Generate failed with response {response.status!r}")
            started = True

        yield response  # yield the first response

        async for response in iterator:
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Model Generate failed with response {response.status!r}")
            yield response

    def _stream(
        self,
        inputs,
        method_name: str = 'stream',
    ) -> Any:
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

        if isinstance(inputs, list):
            assert len(inputs) == 1, 'streaming methods do not support batched calls'
            inputs = inputs[0]
        assert isinstance(inputs, dict)
        kwargs = inputs

        # find the streaming vars in the input signature, and the streaming input python param
        stream_sig = get_stream_from_signature(input_signature)
        if stream_sig is None:
            raise ValueError("Streaming method must have a Stream input")
        stream_argname = stream_sig.name

        # get the streaming input generator from the user-provided function arg values
        user_inputs_generator = kwargs.pop(stream_argname)

        def _input_proto_stream():
            # first item contains all the inputs and the first stream item
            proto = resources_pb2.Input()
            try:
                item = next(user_inputs_generator)
            except StopIteration:
                return  # no items to stream
            kwargs[stream_argname] = item
            serialize(kwargs, input_signature, proto.data)

            yield proto

            # subsequent items are just the stream items
            for item in user_inputs_generator:
                proto = resources_pb2.Input()
                serialize({stream_argname: item}, [stream_sig], proto.data)
                yield proto

        response_stream = self._stream_by_proto(_input_proto_stream(), method_name)

        for response in response_stream:
            assert len(response.outputs) == 1, 'streaming methods must have exactly one output'
            yield deserialize(response.outputs[0].data, output_signature, is_output=True)

    def _req_iterator(
        self,
        input_iterator: Iterator[List[resources_pb2.Input]],
        method_name: str = None,
        inference_params: Dict = {},
        output_config: Dict = {},
    ):
        request = service_pb2.PostModelOutputsRequest()
        request.CopyFrom(self.request_template)
        if inference_params:
            request.model.model_version.output_info.params.update(inference_params)
        if output_config:
            request.model.model_version.output_info.output_config.MergeFromDict(output_config)
        for inputs in input_iterator:
            req = service_pb2.PostModelOutputsRequest()
            req.CopyFrom(request)
            if isinstance(inputs, list):
                req.inputs.extend(inputs)
            else:
                req.inputs.append(inputs)
            # TODO: put into new proto field?
            if method_name:
                for inp in req.inputs:
                    inp.data.metadata['_method_name'] = method_name
            yield req

    def _stream_by_proto(
        self,
        inputs: Iterator[List[resources_pb2.Input]],
        method_name: str = None,
        inference_params: Dict = {},
        output_config: Dict = {},
    ):
        """Generate the stream output on model based on the given stream of inputs."""
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
                if (
                    status_is_retryable(response.status.code)
                    and time.time() - start_time < 60 * 10
                ):
                    logger.info("Model is still deploying, please wait...")
                    time.sleep(next(backoff_iterator))
                    break
                if response.status.code != status_code_pb2.SUCCESS:
                    raise Exception(f"Model Predict failed with response {response.status!r}")
                else:
                    if not generation_started:
                        generation_started = True
                    yield response

    # TODO: Test async streaming.
    async def _async_stream(
        self,
        inputs,
        method_name: str = 'stream',
    ) -> Any:
        # method_name is set to 'stream' by default, this is because to replicate the input and output signature behaviour of sync to async stream.
        input_signature = self._method_signatures[method_name].input_fields
        output_signature = self._method_signatures[method_name].output_fields

        if isinstance(inputs, list):
            assert len(inputs) == 1, 'streaming methods do not support batched calls'
            inputs = inputs[0]
        assert isinstance(inputs, dict)
        kwargs = inputs

        # find the streaming vars in the input signature, and the streaming input python param
        stream_sig = get_stream_from_signature(input_signature)
        if stream_sig is None:
            raise ValueError("Streaming method must have a Stream input")
        stream_argname = stream_sig.name

        # get the streaming input generator from the user-provided function arg values
        user_inputs_generator = kwargs.pop(stream_argname)

        async def _input_proto_stream():
            # first item contains all the inputs and the first stream item
            proto = resources_pb2.Input()
            try:
                item = await user_inputs_generator.__anext__()
            except StopAsyncIteration:
                return  # no items to stream
            kwargs[stream_argname] = item
            serialize(kwargs, input_signature, proto.data)

            yield proto

        # subsequent items are just the stream items
        async for item in user_inputs_generator:
            proto = resources_pb2.Input()
            serialize({stream_argname: item}, [stream_sig], proto.data)
            yield proto

        response_stream = await self._async_stream_by_proto(_input_proto_stream(), method_name)

        async for response in response_stream:
            assert len(response.outputs) == 1, 'streaming methods must have exactly one output'
            yield deserialize(response.outputs[0].data, output_signature, is_output=True)

    async def _async_stream_by_proto(
        self,
        inputs: Iterator[List[resources_pb2.Input]],
        method_name: str = None,
        inference_params: Dict = {},
        output_config: Dict = {},
    ):
        """Generate the async stream output on model based on the given stream of inputs."""
        # if not isinstance(inputs, Iterator[List[Input]]):
        #   raise UserError('Invalid inputs, inputs must be a iterator of list of Input objects.')

        request = self._req_iterator(inputs, method_name, inference_params, output_config)

        start_time = time.time()
        backoff_iterator = BackoffIterator(10)
        generation_started = False
        while True:
            if generation_started:
                break
            stream_response = await self.async_stub.StreamModelOutputs(request)
            async for response in stream_response:
                if (
                    status_is_retryable(response.status.code)
                    and time.time() - start_time < 60 * 10
                ):
                    logger.info("Model is still deploying, please wait...")
                    await asyncio.sleep(next(backoff_iterator))
                    break
                if response.status.code != status_code_pb2.SUCCESS:
                    raise Exception(f"Model Predict failed with response {response.status!r}")
                else:
                    if not generation_started:
                        generation_started = True
                    yield response
