import inspect
import itertools
import os
import threading
import traceback
from abc import ABC
from collections import abc
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.runners.utils import data_types
from clarifai.runners.utils.data_utils import DataConverter
from clarifai.runners.utils.method_signatures import (
    build_function_signature,
    deserialize,
    get_stream_from_signature,
    serialize,
    signatures_to_json,
)
from clarifai.runners.utils.model_utils import is_proto_style_method
from clarifai.utils.logging import logger

_METHOD_INFO_ATTR = '_cf_method_info'

_RAISE_EXCEPTIONS = os.getenv("RAISE_EXCEPTIONS", "false").lower() in ("true", "1")

FALLBACK_METHOD_PROTO = 'PostModelOutputs'
FALLBACK_METHOD_PYTHON = 'predict'


class ModelClass(ABC):
    '''
    Base class for model classes that can be run as a service.

    Define predict, generate, or stream methods using the @ModelClass.method decorator.

    Example:

      from clarifai.runners.model_class import ModelClass
      from clarifai.runners.utils.data_types import NamedFields
      from typing import List, Iterator

      class MyModel(ModelClass):

        @ModelClass.method
        def predict(self, x: str, y: int) -> List[str]:
          return [x] * y

        @ModelClass.method
        def generate(self, x: str, y: int) -> Iterator[str]:
          for i in range(y):
            yield x + str(i)

        @ModelClass.method
        def stream(self, input_stream: Iterator[NamedFields(x=str, y=int)]) -> Iterator[str]:
          for item in input_stream:
            yield item.x + ' ' + str(item.y)
    '''

    def __init__(self):
        super().__init__()
        self._thread_local = threading.local()

    @staticmethod
    def method(func):
        setattr(func, _METHOD_INFO_ATTR, _MethodInfo(func))
        return func

    def set_output_context(self, prompt_tokens=None, completion_tokens=None):
        """Set the prompt and completion tokens for the Output proto.
        In batch mode, call this once per output, in order, before returning each output.
        """
        if not hasattr(self._thread_local, 'token_contexts'):
            self._thread_local.token_contexts = []
        self._thread_local.token_contexts.append((prompt_tokens, completion_tokens))

    def load_model(self):
        """Load the model."""

    def _handle_get_signatures_request(self) -> service_pb2.MultiOutputResponse:
        methods = self._get_method_infos()
        signatures = {
            method.name: method.signature
            for method in methods.values()
            if method.signature is not None
        }
        resp = service_pb2.MultiOutputResponse(
            status=status_pb2.Status(code=status_code_pb2.SUCCESS)
        )
        output = resp.outputs.add()
        output.status.code = status_code_pb2.SUCCESS
        output.data.text.raw = signatures_to_json(signatures)
        return resp

    def _batch_predict(self, method, inputs: List[Dict[str, Any]]) -> List[Any]:
        """Batch predict method for multiple inputs."""
        outputs = []
        for input in inputs:
            output = method(**input)
            outputs.append(output)
        return outputs

    def _batch_generate(self, method, inputs: List[Dict[str, Any]]) -> Iterator[List[Any]]:
        """Batch generate method for multiple inputs."""
        generators = [method(**input) for input in inputs]
        for outputs in itertools.zip_longest(*generators):
            yield outputs

    def predict_wrapper(
        self, request: service_pb2.PostModelOutputsRequest
    ) -> service_pb2.MultiOutputResponse:
        outputs = []
        try:
            # TODO add method name field to proto
            # to support old callers who might not pass in the method name we have a few defaults.
            # first we look for a PostModelOutputs method that is implemented as protos and use that
            # if it exists.
            # if not we default to 'predict'.
            method_name = None
            if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
                method_name = request.inputs[0].data.metadata['_method_name']
            if method_name is None and FALLBACK_METHOD_PROTO in self._get_method_infos():
                _info = self._get_method_infos(FALLBACK_METHOD_PROTO)
                if _info.proto_method:
                    method_name = FALLBACK_METHOD_PROTO
            if method_name is None:
                method_name = FALLBACK_METHOD_PYTHON
            if (
                method_name == '_GET_SIGNATURES'
            ):  # special case to fetch signatures, TODO add endpoint for this
                return self._handle_get_signatures_request()
            if method_name not in self._get_method_infos():
                raise ValueError(f"Method {method_name} not found in model class")
            method = getattr(self, method_name)
            method_info = self._get_method_infos(method_name)
            signature = method_info.signature
            proto_method = method_info.proto_method

            # If this is an old predict(proto) -> proto method, just call it and return
            # the response.
            if proto_method:
                out_proto = method(request)
                # if we already have out_proto.status.code set then return
                if out_proto.status.code != status_code_pb2.ZERO:
                    return out_proto

                successes = [
                    out.status.code == status_code_pb2.SUCCESS for out in out_proto.outputs
                ]
                if all(successes):
                    # If all outputs are successful, we can return the response.
                    out_proto.status.CopyFrom(
                        status_pb2.Status(code=status_code_pb2.SUCCESS, description='Success')
                    )
                    return out_proto
                if any(successes):
                    # If some outputs are successful and some are not, we return a mixed status.
                    out_proto.status.CopyFrom(
                        status_pb2.Status(
                            code=status_code_pb2.MIXED_STATUS, description='Mixed Status'
                        )
                    )
                    return out_proto
                # If all outputs are failures, we return a failure status.
                out_proto.status.CopyFrom(
                    status_pb2.Status(code=status_code_pb2.FAILURE, description='Failed')
                )
                return out_proto

            python_param_types = method_info.python_param_types
            for input in request.inputs:
                # check if input is in old format
                is_convert = DataConverter.is_old_format(input.data)
                if is_convert:
                    # convert to new format
                    new_data = DataConverter.convert_input_data_to_new_format(
                        input.data, signature.input_fields
                    )
                    input.data.CopyFrom(new_data)

            # convert inputs to python types
            inputs = self._convert_input_protos_to_python(
                request.inputs, signature.input_fields, python_param_types
            )
            if len(inputs) == 1:
                inputs = inputs[0]
                output = method(**inputs)
                outputs.append(
                    self._convert_output_to_proto(
                        output, signature.output_fields, convert_old_format=is_convert
                    )
                )
            else:
                outputs = self._batch_predict(method, inputs)
                outputs = [
                    self._convert_output_to_proto(
                        output, signature.output_fields, convert_old_format=is_convert
                    )
                    for output in outputs
                ]

            return service_pb2.MultiOutputResponse(
                outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS)
            )
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            logger.exception("Error in predict")
            return service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.FAILURE,
                    details=str(e),
                    stack_trace=traceback.format_exc().split('\n'),
                )
            )

    def generate_wrapper(
        self, request: service_pb2.PostModelOutputsRequest
    ) -> Iterator[service_pb2.MultiOutputResponse]:
        try:
            assert len(request.inputs) == 1, "Generate requires exactly one input"
            method_name = 'generate'
            if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
                method_name = request.inputs[0].data.metadata['_method_name']
            method = getattr(self, method_name)
            method_info = self._get_method_infos(method_name)
            signature = method_info.signature
            python_param_types = method_info.python_param_types
            for input in request.inputs:
                # check if input is in old format
                is_convert = DataConverter.is_old_format(input.data)
                if is_convert:
                    # convert to new format
                    new_data = DataConverter.convert_input_data_to_new_format(
                        input.data, signature.input_fields
                    )
                    input.data.CopyFrom(new_data)
            inputs = self._convert_input_protos_to_python(
                request.inputs, signature.input_fields, python_param_types
            )
            if len(inputs) == 1:
                inputs = inputs[0]
                for output in method(**inputs):
                    resp = service_pb2.MultiOutputResponse()
                    self._convert_output_to_proto(
                        output,
                        signature.output_fields,
                        proto=resp.outputs.add(),
                        convert_old_format=is_convert,
                    )
                    resp.status.code = status_code_pb2.SUCCESS
                    yield resp
            else:
                for outputs in self._batch_generate(method, inputs):
                    resp = service_pb2.MultiOutputResponse()
                    for output in outputs:
                        self._convert_output_to_proto(
                            output,
                            signature.output_fields,
                            proto=resp.outputs.add(),
                            convert_old_format=is_convert,
                        )
                    resp.status.code = status_code_pb2.SUCCESS
                    yield resp
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            logger.exception("Error in generate")
            yield service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.FAILURE,
                    details=str(e),
                    stack_trace=traceback.format_exc().split('\n'),
                )
            )

    def stream_wrapper(
        self, request_iterator: Iterator[service_pb2.PostModelOutputsRequest]
    ) -> Iterator[service_pb2.MultiOutputResponse]:
        try:
            request = next(request_iterator)  # get first request to determine method
            assert len(request.inputs) == 1, "Streaming requires exactly one input"

            method_name = 'stream'
            if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
                method_name = request.inputs[0].data.metadata['_method_name']
            method = getattr(self, method_name)
            method_info = self._get_method_infos(method_name)
            signature = method_info.signature
            python_param_types = method_info.python_param_types

            # find the streaming vars in the signature
            stream_sig = get_stream_from_signature(signature.input_fields)
            if stream_sig is None:
                raise ValueError("Streaming method must have a Stream input")
            stream_argname = stream_sig.name

            for input in request.inputs:
                # check if input is in old format
                is_convert = DataConverter.is_old_format(input.data)
                if is_convert:
                    # convert to new format
                    new_data = DataConverter.convert_input_data_to_new_format(
                        input.data, signature.input_fields
                    )
                    input.data.CopyFrom(new_data)
            # convert all inputs for the first request, including the first stream value
            inputs = self._convert_input_protos_to_python(
                request.inputs, signature.input_fields, python_param_types
            )
            kwargs = inputs[0]

            # first streaming item
            first_item = kwargs.pop(stream_argname)

            # streaming generator
            def InputStream():
                yield first_item
                # subsequent streaming items contain only the streaming input
                for request in request_iterator:
                    item = self._convert_input_protos_to_python(
                        request.inputs, [stream_sig], python_param_types
                    )
                    item = item[0][stream_argname]
                    yield item

            # add stream generator back to the input kwargs
            kwargs[stream_argname] = InputStream()

            for output in method(**kwargs):
                resp = service_pb2.MultiOutputResponse()
                self._convert_output_to_proto(
                    output,
                    signature.output_fields,
                    proto=resp.outputs.add(),
                    convert_old_format=is_convert,
                )
                resp.status.code = status_code_pb2.SUCCESS
                yield resp
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            logger.exception("Error in stream")
            yield service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.FAILURE,
                    details=str(e),
                    stack_trace=traceback.format_exc().split('\n'),
                )
            )

    def _convert_input_protos_to_python(
        self,
        inputs: List[resources_pb2.Input],
        variables_signature: List[resources_pb2.ModelTypeField],
        python_param_types,
    ) -> List[Dict[str, Any]]:
        result = []
        for input in inputs:
            kwargs = deserialize(input.data, variables_signature)
            # dynamic cast to annotated types
            for k, v in kwargs.items():
                if k not in python_param_types:
                    continue

                if hasattr(python_param_types[k], "__args__") and (
                    getattr(python_param_types[k], "__origin__", None)
                    in [abc.Iterator, abc.Generator, abc.Iterable]
                ):
                    # get the type of the items in the stream
                    stream_type = python_param_types[k].__args__[0]

                    kwargs[k] = data_types.cast(v, stream_type)
                else:
                    kwargs[k] = data_types.cast(v, python_param_types[k])
            result.append(kwargs)
        return result

    def _convert_output_to_proto(
        self,
        output: Any,
        variables_signature: List[resources_pb2.ModelTypeField],
        proto=None,
        convert_old_format=False,
    ) -> resources_pb2.Output:
        if proto is None:
            proto = resources_pb2.Output()
        serialize({'return': output}, variables_signature, proto.data, is_output=True)
        if convert_old_format:
            # convert to old format
            data = DataConverter.convert_output_data_to_old_format(proto.data)
            proto.data.CopyFrom(data)
        proto.status.code = status_code_pb2.SUCCESS
        # Per-output token context support
        token_contexts = getattr(self._thread_local, 'token_contexts', None)
        prompt_tokens = completion_tokens = None
        if token_contexts and len(token_contexts) > 0:
            prompt_tokens, completion_tokens = token_contexts.pop(0)
            # If this was the last, clean up
            if len(token_contexts) == 0:
                del self._thread_local.token_contexts
        if prompt_tokens is not None:
            proto.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            proto.completion_tokens = completion_tokens
        return proto

    @classmethod
    def _register_model_methods(cls):
        # go up the class hierarchy to find all decorated methods, and add to registry of current class
        methods = {}
        for base in reversed(cls.__mro__):
            for name, method in base.__dict__.items():
                # Skip class attributes, mocked objects, and non-methods
                if not callable(method) or isinstance(method, (classmethod, staticmethod)):
                    continue
                # Skip any mocked objects or attributes
                if isinstance(method, MagicMock) or hasattr(method, '_mock_return_value'):
                    continue
                # Only include methods that have been decorated with @ModelClass.method
                method_info = getattr(method, _METHOD_INFO_ATTR, None)
                if not method_info:  # regular function, not a model method
                    continue
                methods[name] = method_info
        # check for generic predict(request) -> response, etc. methods
        # older models never had generate or stream so don't bother with them.
        for name in [FALLBACK_METHOD_PROTO]:  # , 'GenerateModelOutputs', 'StreamModelOutputs'):
            if hasattr(cls, name) and name not in methods:
                method = getattr(cls, name)
                if not callable(method):
                    continue
                if is_proto_style_method(method):
                    # If this is a proto-style method, we can add it to the registry as a special case.
                    methods[name] = _MethodInfo(method, proto_method=True)
        # set method table for this class in the registry
        return methods

    @classmethod
    def _get_method_infos(cls, func_name=None):
        # FIXME: this is a re-use of the _METHOD_INFO_ATTR attribute to store the method info
        # for all methods on the class. Should use a different attribute name to avoid confusion.
        if not hasattr(cls, _METHOD_INFO_ATTR):
            setattr(cls, _METHOD_INFO_ATTR, cls._register_model_methods())
        method_infos = getattr(cls, _METHOD_INFO_ATTR)
        if func_name:
            return method_infos[func_name]
        return method_infos


class _MethodInfo:
    def __init__(self, method, proto_method=False):
        """Initialize a MethodInfo instance.

        Args:
            method: The method to wrap.
            old_method: If True, this is an old proto-style method that returns a proto directly.
        """
        self.name = method.__name__
        self.proto_method = proto_method
        if not proto_method:
            self.signature = build_function_signature(method)
        else:
            self.signature = None
        self.python_param_types = {
            p.name: p.annotation
            for p in inspect.signature(method).parameters.values()
            if p.annotation != inspect.Parameter.empty
        }
        self.python_param_types.pop('self', None)
