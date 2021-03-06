# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/api/face.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from clarifai.rest.grpc.proto.clarifai.api import concept_pb2 as proto_dot_clarifai_dot_api_dot_concept__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/clarifai/api/face.proto',
  package='clarifai.api',
  syntax='proto3',
  serialized_pb=_b('\n\x1dproto/clarifai/api/face.proto\x12\x0c\x63larifai.api\x1a proto/clarifai/api/concept.proto\"7\n\x0c\x46\x61\x63\x65Identity\x12\'\n\x08\x63oncepts\x18\x01 \x03(\x0b\x32\x15.clarifai.api.Concept\"2\n\x07\x46\x61\x63\x65\x41ge\x12\'\n\x08\x63oncepts\x18\x01 \x03(\x0b\x32\x15.clarifai.api.Concept\"=\n\x12\x46\x61\x63\x65GenderIdentity\x12\'\n\x08\x63oncepts\x18\x01 \x03(\x0b\x32\x15.clarifai.api.Concept\"9\n\x0e\x46\x61\x63\x65MCAffinity\x12\'\n\x08\x63oncepts\x18\x01 \x03(\x0b\x32\x15.clarifai.api.Concept\"\xe0\x01\n\x04\x46\x61\x63\x65\x12,\n\x08identity\x18\x01 \x01(\x0b\x32\x1a.clarifai.api.FaceIdentity\x12-\n\x0e\x61ge_appearance\x18\x02 \x01(\x0b\x32\x15.clarifai.api.FaceAge\x12;\n\x11gender_appearance\x18\x03 \x01(\x0b\x32 .clarifai.api.FaceGenderIdentity\x12>\n\x18multicultural_appearance\x18\x04 \x01(\x0b\x32\x1c.clarifai.api.FaceMCAffinityBZ\n\x1b\x63larifai2.internal.grpc.apiZ\x03\x61pi\xa2\x02\x04\x43\x41IP\xaa\x02\x16\x43larifai.Internal.GRPC\xc2\x02\x01_\xca\x02\x11\x43larifai\\Internalb\x06proto3')
  ,
  dependencies=[proto_dot_clarifai_dot_api_dot_concept__pb2.DESCRIPTOR,])




_FACEIDENTITY = _descriptor.Descriptor(
  name='FaceIdentity',
  full_name='clarifai.api.FaceIdentity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='concepts', full_name='clarifai.api.FaceIdentity.concepts', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=81,
  serialized_end=136,
)


_FACEAGE = _descriptor.Descriptor(
  name='FaceAge',
  full_name='clarifai.api.FaceAge',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='concepts', full_name='clarifai.api.FaceAge.concepts', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=138,
  serialized_end=188,
)


_FACEGENDERIDENTITY = _descriptor.Descriptor(
  name='FaceGenderIdentity',
  full_name='clarifai.api.FaceGenderIdentity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='concepts', full_name='clarifai.api.FaceGenderIdentity.concepts', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=190,
  serialized_end=251,
)


_FACEMCAFFINITY = _descriptor.Descriptor(
  name='FaceMCAffinity',
  full_name='clarifai.api.FaceMCAffinity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='concepts', full_name='clarifai.api.FaceMCAffinity.concepts', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=253,
  serialized_end=310,
)


_FACE = _descriptor.Descriptor(
  name='Face',
  full_name='clarifai.api.Face',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity', full_name='clarifai.api.Face.identity', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='age_appearance', full_name='clarifai.api.Face.age_appearance', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gender_appearance', full_name='clarifai.api.Face.gender_appearance', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='multicultural_appearance', full_name='clarifai.api.Face.multicultural_appearance', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=313,
  serialized_end=537,
)

_FACEIDENTITY.fields_by_name['concepts'].message_type = proto_dot_clarifai_dot_api_dot_concept__pb2._CONCEPT
_FACEAGE.fields_by_name['concepts'].message_type = proto_dot_clarifai_dot_api_dot_concept__pb2._CONCEPT
_FACEGENDERIDENTITY.fields_by_name['concepts'].message_type = proto_dot_clarifai_dot_api_dot_concept__pb2._CONCEPT
_FACEMCAFFINITY.fields_by_name['concepts'].message_type = proto_dot_clarifai_dot_api_dot_concept__pb2._CONCEPT
_FACE.fields_by_name['identity'].message_type = _FACEIDENTITY
_FACE.fields_by_name['age_appearance'].message_type = _FACEAGE
_FACE.fields_by_name['gender_appearance'].message_type = _FACEGENDERIDENTITY
_FACE.fields_by_name['multicultural_appearance'].message_type = _FACEMCAFFINITY
DESCRIPTOR.message_types_by_name['FaceIdentity'] = _FACEIDENTITY
DESCRIPTOR.message_types_by_name['FaceAge'] = _FACEAGE
DESCRIPTOR.message_types_by_name['FaceGenderIdentity'] = _FACEGENDERIDENTITY
DESCRIPTOR.message_types_by_name['FaceMCAffinity'] = _FACEMCAFFINITY
DESCRIPTOR.message_types_by_name['Face'] = _FACE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FaceIdentity = _reflection.GeneratedProtocolMessageType('FaceIdentity', (_message.Message,), dict(
  DESCRIPTOR = _FACEIDENTITY,
  __module__ = 'proto.clarifai.api.face_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.FaceIdentity)
  ))
_sym_db.RegisterMessage(FaceIdentity)

FaceAge = _reflection.GeneratedProtocolMessageType('FaceAge', (_message.Message,), dict(
  DESCRIPTOR = _FACEAGE,
  __module__ = 'proto.clarifai.api.face_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.FaceAge)
  ))
_sym_db.RegisterMessage(FaceAge)

FaceGenderIdentity = _reflection.GeneratedProtocolMessageType('FaceGenderIdentity', (_message.Message,), dict(
  DESCRIPTOR = _FACEGENDERIDENTITY,
  __module__ = 'proto.clarifai.api.face_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.FaceGenderIdentity)
  ))
_sym_db.RegisterMessage(FaceGenderIdentity)

FaceMCAffinity = _reflection.GeneratedProtocolMessageType('FaceMCAffinity', (_message.Message,), dict(
  DESCRIPTOR = _FACEMCAFFINITY,
  __module__ = 'proto.clarifai.api.face_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.FaceMCAffinity)
  ))
_sym_db.RegisterMessage(FaceMCAffinity)

Face = _reflection.GeneratedProtocolMessageType('Face', (_message.Message,), dict(
  DESCRIPTOR = _FACE,
  __module__ = 'proto.clarifai.api.face_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.Face)
  ))
_sym_db.RegisterMessage(Face)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\033clarifai2.internal.grpc.apiZ\003api\242\002\004CAIP\252\002\026Clarifai.Internal.GRPC\302\002\001_\312\002\021Clarifai\\Internal'))
# @@protoc_insertion_point(module_scope)
