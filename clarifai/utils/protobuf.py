from google.protobuf import struct_pb2, symbol_database, wrappers_pb2
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from clarifai.utils.logging import logger

# Initialize the Protobuf symbol database for dynamic message type resolution
db = symbol_database.Default()

# Define standard wrapper types for Protobuf well-known types
WRAPPER_TYPES = {
    wrappers_pb2.Int32Value,
    wrappers_pb2.Int64Value,
    wrappers_pb2.StringValue,
    wrappers_pb2.UInt32Value,
    wrappers_pb2.UInt64Value,
}


def dict_to_protobuf(pb_obj: Message, data: dict) -> None:
    """Recursively convert a nested dictionary to a Protobuf message object.

    Args:
        pb_obj: The target Protobuf message instance to populate.
        data: Source dictionary containing the data to convert.
    """

    for field, value in data.items():
        if field not in pb_obj.DESCRIPTOR.fields_by_name:
            logger.warning(
                f"Ignoring unknown field '{field}' in message '{pb_obj.DESCRIPTOR.name}'"
            )
            continue

        field_descriptor = pb_obj.DESCRIPTOR.fields_by_name[field]

        try:
            # Handle repeated fields (lists)
            if field_descriptor.label == FieldDescriptor.LABEL_REPEATED:
                _handle_repeated_field(pb_obj, field_descriptor, field, value)

            # Handle message fields (nested messages)
            elif field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
                _handle_message_field(pb_obj, field_descriptor, field, value)

            # Handle enums as string or int.
            # Alternative is to use MessageToDict with use_integers_for_enums=True everywhere
            # we end up calling dict_to_protobuf afterwards.
            elif field_descriptor.type == FieldDescriptor.TYPE_ENUM:
                if isinstance(value, str):
                    enum_value = field_descriptor.enum_type.values_by_name.get(value)
                    if enum_value is not None:
                        setattr(pb_obj, field, enum_value.number)
                elif isinstance(value, int):
                    enum_value = field_descriptor.enum_type.values_by_number.get(value)
                    if enum_value is not None:
                        setattr(pb_obj, field, enum_value.number)

            # Handle scalar fields
            elif value:
                setattr(pb_obj, field, value)

        except Exception as e:
            logger.error(f"Error processing field '{field}': {str(e)}")
            raise


def _handle_repeated_field(
    pb_obj: Message, field_descriptor: FieldDescriptor, field: str, value: list
) -> None:
    """Process repeated fields (both scalar and message types)."""
    if not isinstance(value, list):
        logger.warning(f"Expected list for repeated field '{field}', got {type(value).__name__}")
        return

    repeated_field = getattr(pb_obj, field)

    # Handle repeated message fields
    if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
        for item in value:
            if isinstance(item, dict):
                item_msg = repeated_field.add()
                dict_to_protobuf(item_msg, item)
            elif isinstance(item, Message):
                repeated_field.add().CopyFrom(item)
            else:
                logger.warning(
                    f"Invalid type {type(item).__name__} in repeated message field '{field}'"
                )
    else:
        # Handle repeated scalar fields
        try:
            repeated_field.extend(value)
        except TypeError as e:
            logger.error(f"Type mismatch in repeated scalar field '{field}': {str(e)}")
            raise


def _handle_message_field(
    pb_obj: Message, field_descriptor: FieldDescriptor, field: str, value: object
) -> None:
    """Process message-type fields including special types."""
    msg_class = db.GetSymbol(field_descriptor.message_type.full_name)
    target_field = getattr(pb_obj, field)

    # Handle special message types
    if msg_class is Timestamp:
        _set_timestamp_value(target_field, value)
    elif msg_class is struct_pb2.Struct:
        _set_struct_value(target_field, value)
    elif msg_class in WRAPPER_TYPES:
        _set_wrapper_value(target_field, msg_class, value)
    # Handle nested messages
    elif isinstance(value, dict):
        nested_pb = msg_class()
        dict_to_protobuf(nested_pb, value)
        target_field.CopyFrom(nested_pb)
    elif isinstance(value, Message):
        target_field.CopyFrom(value)
    else:
        logger.warning(f"Invalid type {type(value).__name__} for message field '{field}'")


def _set_timestamp_value(target_field: Message, value: object) -> None:
    """Set timestamp value from dict or numeric timestamp."""
    ts = Timestamp()
    if isinstance(value, dict):
        ts.seconds = value.get('seconds', 0)
        ts.nanos = value.get('nanos', 0)
    elif isinstance(value, (int, float)):
        ts.seconds = int(value)
        ts.nanos = int((value - ts.seconds) * 1e9)
    elif isinstance(value, Timestamp):
        ts = value
    else:
        logger.warning(f"Unsupported timestamp format: {type(value).__name__}")
    target_field.CopyFrom(ts)


def _set_struct_value(target_field: Message, value: object) -> None:
    """Convert dictionary to Protobuf Struct."""
    if isinstance(value, struct_pb2.Struct):
        struct = value
    else:
        if not isinstance(value, dict):
            logger.warning(f"Expected dict for Struct field, got {type(value).__name__}")
            return

        struct = struct_pb2.Struct()
        try:
            struct.update(value)
        except ValueError as e:
            logger.error(f"Invalid value in Struct: {str(e)}")
            raise
    target_field.CopyFrom(struct)


def _set_wrapper_value(target_field: Message, wrapper_type: type, value: object) -> None:
    """Set value for wrapper types (e.g., Int32Value)."""
    try:
        wrapper = wrapper_type(value=value)
    except TypeError as e:
        logger.error(f"Invalid value for {wrapper_type.__name__}: {str(e)}")
        raise
    target_field.CopyFrom(wrapper)
