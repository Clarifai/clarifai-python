# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/api/feedback.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/clarifai/api/feedback.proto',
  package='clarifai.api',
  syntax='proto3',
  serialized_pb=_b('\n!proto/clarifai/api/feedback.proto\x12\x0c\x63larifai.api\"\x8a\x01\n\x0c\x46\x65\x65\x64\x62\x61\x63kInfo\x12\x13\n\x0b\x65nd_user_id\x18\x01 \x01(\t\x12\x12\n\nsession_id\x18\x02 \x01(\t\x12+\n\nevent_type\x18\x03 \x01(\x0e\x32\x17.clarifai.api.EventType\x12\x11\n\toutput_id\x18\x04 \x01(\t\x12\x11\n\tsearch_id\x18\x05 \x01(\t*<\n\tEventType\x12\r\n\tundefined\x10\x00\x12\x10\n\x0csearch_click\x10\x01\x12\x0e\n\nannotation\x10\x02\x42Z\n\x1b\x63larifai2.internal.grpc.apiZ\x03\x61pi\xa2\x02\x04\x43\x41IP\xaa\x02\x16\x43larifai.Internal.GRPC\xc2\x02\x01_\xca\x02\x11\x43larifai\\Internalb\x06proto3')
)

_EVENTTYPE = _descriptor.EnumDescriptor(
  name='EventType',
  full_name='clarifai.api.EventType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='undefined', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='search_click', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='annotation', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=192,
  serialized_end=252,
)
_sym_db.RegisterEnumDescriptor(_EVENTTYPE)

EventType = enum_type_wrapper.EnumTypeWrapper(_EVENTTYPE)
undefined = 0
search_click = 1
annotation = 2



_FEEDBACKINFO = _descriptor.Descriptor(
  name='FeedbackInfo',
  full_name='clarifai.api.FeedbackInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='end_user_id', full_name='clarifai.api.FeedbackInfo.end_user_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='session_id', full_name='clarifai.api.FeedbackInfo.session_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='event_type', full_name='clarifai.api.FeedbackInfo.event_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_id', full_name='clarifai.api.FeedbackInfo.output_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='search_id', full_name='clarifai.api.FeedbackInfo.search_id', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=52,
  serialized_end=190,
)

_FEEDBACKINFO.fields_by_name['event_type'].enum_type = _EVENTTYPE
DESCRIPTOR.message_types_by_name['FeedbackInfo'] = _FEEDBACKINFO
DESCRIPTOR.enum_types_by_name['EventType'] = _EVENTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FeedbackInfo = _reflection.GeneratedProtocolMessageType('FeedbackInfo', (_message.Message,), dict(
  DESCRIPTOR = _FEEDBACKINFO,
  __module__ = 'proto.clarifai.api.feedback_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.FeedbackInfo)
  ))
_sym_db.RegisterMessage(FeedbackInfo)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\033clarifai2.internal.grpc.apiZ\003api\242\002\004CAIP\252\002\026Clarifai.Internal.GRPC\302\002\001_\312\002\021Clarifai\\Internal'))
# @@protoc_insertion_point(module_scope)