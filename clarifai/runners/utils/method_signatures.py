import inspect
import json
from collections import abc, namedtuple
from typing import Dict, List, Tuple, get_args, get_origin

import numpy as np
import PIL.Image
import yaml
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message as MessageProto

from clarifai.runners.utils import data_types, data_utils
from clarifai.runners.utils.code_script import _get_base_type, _parse_default_value
from clarifai.runners.utils.serializers import (
    AtomicFieldSerializer,
    JSONSerializer,
    ListSerializer,
    MessageSerializer,
    NamedFieldsSerializer,
    NDArraySerializer,
    Serializer,
    TupleSerializer,
)


def build_function_signature(func):
    '''
    Build a signature for the given function.
    '''
    sig = inspect.signature(func)

    # check if func is bound, and if not, remove self/cls
    if (
        getattr(func, '__self__', None) is None
        and sig.parameters
        and list(sig.parameters.values())[0].name in ('self', 'cls')
    ):
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])

    return_annotation = sig.return_annotation
    if return_annotation == inspect.Parameter.empty:
        raise TypeError('Function must have a return annotation')

    input_sigs = []
    input_streaming = []
    for p in sig.parameters.values():
        model_type_field, _, streaming = build_variable_signature(p.name, p.annotation, p.default)
        input_sigs.append(model_type_field)
        input_streaming.append(streaming)

    output_sig, output_type, output_streaming = build_variable_signature(
        'return', return_annotation, is_output=True
    )
    # TODO: flatten out "return" layer if not needed

    # check for streams and determine method type
    if sum(input_streaming) > 1:
        raise TypeError('streaming methods must have at most one streaming input')
    input_streaming = any(input_streaming)
    if not (input_streaming or output_streaming):
        method_type = 'UNARY_UNARY'
    elif not input_streaming and output_streaming:
        method_type = 'UNARY_STREAMING'
    elif input_streaming and output_streaming:
        method_type = 'STREAMING_STREAMING'
    else:
        raise TypeError('stream methods with streaming inputs must have streaming outputs')

    method_signature = resources_pb2.MethodSignature()

    method_signature.name = func.__name__
    method_signature.method_type = getattr(resources_pb2.RunnerMethodType, method_type)
    assert method_type in ('UNARY_UNARY', 'UNARY_STREAMING', 'STREAMING_STREAMING')
    # method_signature.method_type = method_type
    method_signature.description = inspect.cleandoc(func.__doc__ or '')

    method_signature.input_fields.extend(input_sigs)
    method_signature.output_fields.append(output_sig)
    return method_signature


def _process_input_field(field: resources_pb2.ModelTypeField) -> str:
    base_type = _get_base_type(field)
    if field.iterator:
        type_str = f"Iterator[{base_type}]"
    else:
        type_str = base_type
    default = _parse_default_value(field)
    param = f"{field.name}: {type_str}"
    if default is not None:
        param += f" = {default}"
    return param


def _process_output_field(field: resources_pb2.ModelTypeField) -> str:
    base_type = _get_base_type(field)
    if field.iterator:
        return f"Iterator[{base_type}]"
    else:
        return base_type


def get_method_signature(method_signature: resources_pb2.MethodSignature) -> str:
    """
    Get the method signature of a method in a model.
    """
    # Process input fields
    input_params = []
    for input_field in method_signature.input_fields:
        param_str = _process_input_field(input_field)
        input_params.append(param_str)

    # Process output field
    if not method_signature.output_fields:
        raise ValueError("MethodSignature must have at least one output field")
    output_field = method_signature.output_fields[0]
    return_type = _process_output_field(output_field)

    # Generate function signature
    function_def = f"def {method_signature.name}({', '.join(input_params)}) -> {return_type}:"
    return function_def


def build_variable_signature(name, annotation, default=inspect.Parameter.empty, is_output=False):
    '''
    Build a data proto signature and get the normalized python type for the given annotation.
    '''

    # check valid names (should already be constrained by python naming, but check anyway)
    if not name.isidentifier():
        raise ValueError(f'Invalid variable name: {name}')

    # get fields for each variable based on type
    tp, streaming = _normalize_type(annotation)

    sig = resources_pb2.ModelTypeField()
    sig.name = name
    sig.iterator = streaming

    if not is_output:
        sig.required = default is inspect.Parameter.empty
        if not sig.required:
            if isinstance(default, data_utils.Param):
                sig = default.to_proto(sig)
            else:
                sig = data_utils.Param.set_default(sig, default)

    _fill_signature_type(sig, tp)

    return sig, type, streaming


def _fill_signature_type(sig, tp):
    try:
        if tp in _DATA_TYPES:
            sig.type = _DATA_TYPES[tp].type
            return
    except TypeError:
        pass  # not hashable type

    # Handle NamedFields with annotations
    # Check for dynamically generated NamedFields subclasses (from type annotations)
    if (
        inspect.isclass(tp)
        and issubclass(tp, data_types.NamedFields)
        and hasattr(tp, '__annotations__')
    ):
        sig.type = resources_pb2.ModelTypeField.DataType.NAMED_FIELDS
        for name, inner_type in tp.__annotations__.items():
            inner_sig = sig.type_args.add()
            inner_sig.name = name
            _fill_signature_type(inner_sig, inner_type)
        return

    # Handle NamedFields instances (dict-like)
    if isinstance(tp, data_types.NamedFields):
        sig.type = resources_pb2.ModelTypeField.DataType.NAMED_FIELDS
        for name, inner_type in tp.items():
            inner_sig = sig.type_args.add()
            inner_sig.name = name
            _fill_signature_type(inner_sig, inner_type)
        return

    origin = get_origin(tp)
    args = get_args(tp)

    # Handle Tuple type
    if origin is tuple:
        sig.type = resources_pb2.ModelTypeField.DataType.TUPLE
        for inner_type in args:
            inner_sig = sig.type_args.add()
            inner_sig.name = sig.name + '_item'
            _fill_signature_type(inner_sig, inner_type)
        return

    # Handle List type
    if origin is list:
        sig.type = resources_pb2.ModelTypeField.DataType.LIST
        inner_sig = sig.type_args.add()
        inner_sig.name = sig.name + '_item'
        _fill_signature_type(inner_sig, args[0])
        return

    raise TypeError(f'Unsupported type: {tp}')


def serializer_from_signature(signature):
    '''
    Get the serializer for the given signature.
    '''
    if signature.type in _SERIALIZERS_BY_TYPE_ENUM:
        return _SERIALIZERS_BY_TYPE_ENUM[signature.type]
    if signature.type == resources_pb2.ModelTypeField.DataType.LIST:
        return ListSerializer(serializer_from_signature(signature.type_args[0]))
    if signature.type == resources_pb2.ModelTypeField.DataType.TUPLE:
        return TupleSerializer([serializer_from_signature(sig) for sig in signature.type_args])
    if signature.type == resources_pb2.ModelTypeField.DataType.NAMED_FIELDS:
        return NamedFieldsSerializer(
            {sig.name: serializer_from_signature(sig) for sig in signature.type_args}
        )
    raise ValueError(f'Unsupported type: {signature.type}')


def signatures_to_json(signatures):
    assert isinstance(signatures, dict), (
        'Expected dict of signatures {name: signature}, got %s' % type(signatures)
    )
    # TODO change to proto when ready
    signatures = {name: MessageToDict(sig) for name, sig in signatures.items()}
    return json.dumps(signatures)


def signatures_from_json(json_str):
    signatures_dict = json.loads(json_str)
    assert isinstance(signatures_dict, dict), "Expected JSON to decode into a dictionary"

    return {
        name: ParseDict(sig_dict, resources_pb2.MethodSignature())
        for name, sig_dict in signatures_dict.items()
    }
    # d = json.loads(json_str, object_pairs_hook=_SignatureDict)
    # return d


def signatures_to_yaml(signatures):
    # XXX go in/out of json to get the correct format and python dict types
    d = json.loads(signatures_to_json(signatures))

    def _filter_empty(d):
        if isinstance(d, (list, tuple)):
            return [_filter_empty(v) for v in d if v]
        if isinstance(d, dict):
            return {k: _filter_empty(v) for k, v in d.items() if v}
        return d

    return yaml.dump(_filter_empty(d), default_flow_style=False)


def signatures_from_yaml(yaml_str):
    d = yaml.safe_load(yaml_str)
    return signatures_from_json(json.dumps(d))


def serialize(kwargs, signatures, proto=None, is_output=False):
    '''
    Serialize the given kwargs into the proto using the given signatures.
    '''
    if proto is None:
        proto = resources_pb2.Data()
    unknown = set(kwargs.keys()) - set(sig.name for sig in signatures)
    if unknown:
        if unknown == {'return'} and len(signatures) > 1:
            raise TypeError(
                'Got a single return value, but expected multiple outputs {%s}'
                % ', '.join(sig.name for sig in signatures)
            )
        raise TypeError('Got unexpected key: %s' % ', '.join(unknown))
    for sig_i, sig in enumerate(signatures):
        if sig.name not in kwargs:
            if sig.required:
                raise TypeError(f'Missing required argument: {sig.name}')
            continue  # skip missing fields, they can be set to default on the server
        data = kwargs[sig.name]
        default = data_utils.Param.get_default(sig)
        if data is None and default is None:
            continue
        serializer = serializer_from_signature(sig)
        # TODO determine if any (esp the first) var can go in the proto without parts
        # and whether to put this in the signature or dynamically determine it
        # add the part to the proto
        part = proto.parts.add()
        part.id = sig.name
        serializer.serialize(part.data, data)
    return proto


def deserialize(proto, signatures, is_output=False):
    '''
    Deserialize the given proto into kwargs using the given signatures.
    '''
    if isinstance(signatures, dict):
        signatures = [signatures]  # TODO update return key level and make consistnet
    kwargs = {}
    parts_by_name = {part.id: part for part in proto.parts}
    for sig_i, sig in enumerate(signatures):
        serializer = serializer_from_signature(sig)
        part = parts_by_name.get(sig.name)
        if part is not None:
            kwargs[sig.name] = serializer.deserialize(part.data)
        else:
            if sig_i == 0:
                # possible inlined first value
                value = serializer.deserialize(proto)
                if id(value) not in _ZERO_VALUE_IDS:
                    # note missing values are not set to defaults, since they are not in parts
                    # an actual zero value passed in must be set in an explicit part
                    kwargs[sig.name] = value
                continue

            if sig.required or is_output:  # TODO allow optional outputs?
                raise ValueError(f'Missing required field: {sig.name}')
            continue
    if len(kwargs) == 1 and 'return' in kwargs:
        return kwargs['return']
    return kwargs


def get_stream_from_signature(signatures):
    '''
    Get the stream signature from the given signatures.
    '''
    for sig in signatures:
        if sig.iterator:
            return sig
    return None


def _is_empty_proto_data(data):
    if isinstance(data, np.ndarray):
        return False
    if isinstance(data, MessageProto):
        return not data.ByteSize()
    return not data


def _normalize_type(tp):
    '''
    Normalize the types for the given parameter.
    Returns the normalized type and whether the parameter is streaming.
    '''
    # stream type indicates streaming, not part of the data itself
    # it can only be used at the top-level of the var type
    streaming = get_origin(tp) in [abc.Iterator, abc.Generator, abc.Iterable]
    if streaming:
        tp = get_args(tp)[0]

    return _normalize_data_type(tp), streaming


def _normalize_data_type(tp):
    # container types that need to be serialized as parts
    if get_origin(tp) is list and get_args(tp):
        return List[_normalize_data_type(get_args(tp)[0])]

    if get_origin(tp) is tuple:
        if not get_args(tp):
            raise TypeError('Tuple must have types specified')
        return Tuple[tuple(_normalize_data_type(val) for val in get_args(tp))]

    if isinstance(tp, (tuple, list)):
        return Tuple[tuple(_normalize_data_type(val) for val in tp)]

    if tp is data_types.NamedFields:
        raise TypeError('NamedFields must have types specified')

    # Handle dynamically generated NamedFields subclasses with annotations
    if (
        isinstance(tp, type)
        and issubclass(tp, data_types.NamedFields)
        and hasattr(tp, '__annotations__')
    ):
        return data_types.NamedFields(
            **{k: _normalize_data_type(v) for k, v in tp.__annotations__.items()}
        )

    if isinstance(tp, (dict, data_types.NamedFields)):
        return data_types.NamedFields(
            **{name: _normalize_data_type(val) for name, val in tp.items()}
        )

    # check if numpy array type, and if so, use ndarray
    if get_origin(tp) is np.ndarray:
        return np.ndarray

    # check for PIL images (sometimes types use the module, sometimes the class)
    # set these to use the Image data handler
    if tp in (data_types.Image, PIL.Image.Image):
        return data_types.Image

    if tp is PIL.Image:
        raise TypeError('Use PIL.Image.Image instead of PIL.Image module')

    # jsonable list and dict, these can be serialized as json
    # (tuple we want to keep as a tuple for args and returns, so don't include here)
    if tp in (list, dict, Dict) or (get_origin(tp) in (list, dict, Dict) and _is_jsonable(tp)):
        return data_types.JSON

    # check for known data types
    try:
        if tp in _DATA_TYPES:
            return tp
    except TypeError:
        pass  # not hashable type

    raise TypeError(f'Unsupported type: {tp}')


def _is_jsonable(tp):
    if tp in (dict, list, tuple, str, int, float, bool, type(None)):
        return True
    if get_origin(tp) in (tuple, list, dict):
        return all(_is_jsonable(val) for val in get_args(tp))
    return False


# type: name of the data type
# data_field: name of the field in the data proto
# serializer: serializer for the data type
_DataType = namedtuple('_DataType', ('type', 'serializer'))

_ZERO_VALUE_IDS = {id(None), id(''), id(b''), id(0), id(0.0), id(False)}

# simple, non-container types that correspond directly to a data field
_DATA_TYPES = {
    str: _DataType(
        resources_pb2.ModelTypeField.DataType.STR, AtomicFieldSerializer('string_value')
    ),
    bytes: _DataType(
        resources_pb2.ModelTypeField.DataType.BYTES, AtomicFieldSerializer('bytes_value')
    ),
    int: _DataType(resources_pb2.ModelTypeField.DataType.INT, AtomicFieldSerializer('int_value')),
    float: _DataType(
        resources_pb2.ModelTypeField.DataType.FLOAT, AtomicFieldSerializer('float_value')
    ),
    bool: _DataType(
        resources_pb2.ModelTypeField.DataType.BOOL, AtomicFieldSerializer('bool_value')
    ),
    np.ndarray: _DataType(
        resources_pb2.ModelTypeField.DataType.NDARRAY, NDArraySerializer('ndarray')
    ),
    data_types.JSON: _DataType(
        resources_pb2.ModelTypeField.DataType.JSON_DATA, JSONSerializer('string_value')
    ),  # TODO change to json_value when new proto is ready
    data_types.Text: _DataType(
        resources_pb2.ModelTypeField.DataType.TEXT, MessageSerializer('text', data_types.Text)
    ),
    data_types.Image: _DataType(
        resources_pb2.ModelTypeField.DataType.IMAGE, MessageSerializer('image', data_types.Image)
    ),
    data_types.Concept: _DataType(
        resources_pb2.ModelTypeField.DataType.CONCEPT,
        MessageSerializer('concepts', data_types.Concept),
    ),
    data_types.Region: _DataType(
        resources_pb2.ModelTypeField.DataType.REGION,
        MessageSerializer('regions', data_types.Region),
    ),
    data_types.Frame: _DataType(
        resources_pb2.ModelTypeField.DataType.FRAME, MessageSerializer('frames', data_types.Frame)
    ),
    data_types.Audio: _DataType(
        resources_pb2.ModelTypeField.DataType.AUDIO, MessageSerializer('audio', data_types.Audio)
    ),
    data_types.Video: _DataType(
        resources_pb2.ModelTypeField.DataType.VIDEO, MessageSerializer('video', data_types.Video)
    ),
}

_SERIALIZERS_BY_TYPE_ENUM = {dt.type: dt.serializer for dt in _DATA_TYPES.values()}


class CompatibilitySerializer(Serializer):
    '''
    Serialization of basic value types, used for backwards compatibility
    with older models that don't have type signatures.
    '''

    def serialize(self, data_proto, value):
        tp = _normalize_data_type(type(value))

        try:
            serializer = _DATA_TYPES[tp].serializer
        except (KeyError, TypeError):
            raise TypeError(f'serializer currently only supports basic types, got {tp}')

        serializer.serialize(data_proto, value)

    def deserialize(self, data_proto):
        fields = [k.name for k, _ in data_proto.ListFields()]
        if 'parts' in fields:
            raise ValueError('serializer does not support parts')
        serializers = [
            serializer
            for serializer in _SERIALIZERS_BY_TYPE_ENUM.values()
            if serializer.field_name in fields
        ]
        if not serializers:
            raise ValueError('Returned data not recognized')
        if len(serializers) != 1:
            raise ValueError('Only single output supported for serializer')
        serializer = serializers[0]
        return serializer.deserialize(data_proto)
