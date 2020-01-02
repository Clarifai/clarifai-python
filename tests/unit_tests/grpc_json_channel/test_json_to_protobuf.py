import json
import unittest

from google.protobuf.struct_pb2 import Struct

from clarifai.rest.grpc.custom_converters.custom_message_to_dict import protobuf_to_dict
from clarifai.rest.grpc.grpc_json_channel import dict_to_protobuf
from clarifai.rest.grpc.proto.clarifai.api.concept_pb2 import Concept as ConceptPB


class TestJsonToProtobuf(unittest.TestCase):

  def test_concept_with_no_value(self):
    converted = dict_to_protobuf(ConceptPB, {'id': 'some-id', 'name': 'Some Name'})
    assert converted.value == 1.0

  def test_concept_with_value_non_zero(self):
    converted = dict_to_protobuf(ConceptPB, {'id': 'some-id', 'name': 'Some Name', 'value': 0.5})
    assert converted.value == 0.5

  def test_concept_with_value_zero(self):
    converted = dict_to_protobuf(ConceptPB, {'id': 'some-id', 'name': 'Some Name', 'value': 0.0})
    assert converted.value == 0.0

  def test_concept_with_value_one(self):
    converted = dict_to_protobuf(ConceptPB, {'id': 'some-id', 'name': 'Some Name', 'value': 1.0})
    assert converted.value == 1.0

  def test_concept_with_new_field(self):
    converted = dict_to_protobuf(
        ConceptPB, {'id': 'some-id',
                    'name': 'Some Name',
                    'value': 1.0,
                    'new_field': 'new_value'},
        ignore_unknown_fields=True)
    assert not hasattr(converted, 'new_field')

  def test_struct_converts_primitive_fields(self):
    converted = dict_to_protobuf(Struct,
                                 json.loads("""
{
  "field1": "value1",
  "field2": 3,
  "field3": 1.0,
  "field4": true,
  "field5": null
} 
        """))

    s = Struct()
    s['field1'] = 'value1'
    s['field2'] = 3
    s['field3'] = 1.0
    s['field4'] = True
    s['field5'] = None

    assert converted == s

  def test_struct_converts_single_array_field(self):
    converted = dict_to_protobuf(Struct,
                                 json.loads("""
{
  "field1": ["value1", 3, 1.0, true, null]
} 
        """))

    s = Struct()
    s['field1'] = ['value1', 3, 1.0, True, None]

    assert converted == s

  def test_struct_converts_single_object_field(self):
    converted = dict_to_protobuf(Struct,
                                 json.loads("""
{
  "field1": {
    "key1": "str",
    "key2": 3,
    "key3": 1.0,
    "key4": true,
    "key5": null
  },
  "field2": "key2"
}
        """))

    s = Struct()
    s['field1'] = {'key1': 'str', 'key2': 3, 'key3': 1.0, 'key4': True, 'key5': None}
    s['field2'] = 'key2'

    assert converted == s

  def test_struct_serial(self):
    parameters = {
        'MAX_NITEMS': 1000000.0,
        'MIN_NITEMS': 1000,
        'N_EPOCHS': 5,
        'custom_training_cfg': 'custom_training_1layer',
        'custom_training_cfg_args': {},
        'this_is_none': None,
    }
    converted = dict_to_protobuf(Struct, parameters, ignore_unknown_fields=True)
    converted_parameters = protobuf_to_dict(converted)

    assert parameters == converted_parameters
