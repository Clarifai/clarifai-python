import json
import unittest

from google.protobuf.struct_pb2 import Struct

from clarifai.rest.grpc.grpc_json_channel import protobuf_to_dict
from clarifai.rest.grpc.proto.clarifai.api.concept_pb2 import Concept as ConceptPB
from clarifai.rest.grpc.proto.clarifai.api.model_pb2 import \
    MultiModelResponse as MultiModelResponsePB
from clarifai.rest.grpc.proto.clarifai.api.status.status_pb2 import Status as StatusPB
from clarifai.rest.grpc.proto.clarifai.api.workflow_pb2 import \
    MultiWorkflowResponse as MultiWorkflowResponsePB


class TestProtobufToJson(unittest.TestCase):

  def test_concept_with_no_value(self):
    concept = ConceptPB()
    concept.id = 'some-id'
    concept.name = 'Some Name'
    converted = protobuf_to_dict(concept)
    assert converted['value'] == 0.0

  def test_concept_with_value_non_zero(self):
    concept = ConceptPB()
    concept.id = 'some-id'
    concept.name = 'Some Name'
    concept.value = 0.5
    converted = protobuf_to_dict(concept)
    assert converted['value'] == 0.5

  def test_concept_with_value_zero(self):
    concept = ConceptPB()
    concept.id = 'some-id'
    concept.name = 'Some Name'
    concept.value = 0.0
    converted = protobuf_to_dict(concept)
    assert converted['value'] == 0.0

  def test_concept_with_value_one(self):
    concept = ConceptPB()
    concept.id = 'some-id'
    concept.name = 'Some Name'
    concept.value = True
    converted = protobuf_to_dict(concept)
    assert converted['value'] == 1.0

  def test_show_workflows_list_if_empty(self):
    status = StatusPB()
    status.description = 'Some description'

    workflows_response = MultiWorkflowResponsePB()
    workflows_response.status.CopyFrom(status)

    converted = protobuf_to_dict(workflows_response)
    assert (converted == {'status': {'description': 'Some description'}, 'workflows': []})

  def test_show_models_list_if_empty(self):
    status = StatusPB()
    status.description = 'Some description'

    models_response = MultiModelResponsePB()
    models_response.status.CopyFrom(status)

    converted = protobuf_to_dict(models_response)
    assert (converted == {'status': {'description': 'Some description'}, 'models': []})

  def test_json_converts_primitive_fields(self):
    s = Struct()
    s['field1'] = 'value1'
    s['field2'] = 3.0
    s['field3'] = 1.0
    s['field4'] = True
    s['field5'] = None

    converted = protobuf_to_dict(s)
    assert (converted == json.loads("""
{
  "field1": "value1",
  "field2": 3.0,
  "field3": 1.0,
  "field4": true,
  "field5": null
}
    """))

  def test_json_converts_single_array_field(self):
    s = Struct()
    s['field1'] = ['value1', 3.0, 1.0, True, None]

    converted = protobuf_to_dict(s)
    assert converted == json.loads("""
{
  "field1": ["value1", 3.0, 1.0, true, null]
}
    """)

  def test_json_converts_single_object(self):
    s = Struct()
    s['field1'] = {
        'key1': 'value1',
        'key2': 3.0,
        'key3': 1.0,
        'key4': True,
        'key5': None,
    }
    s['field2'] = 'key2'

    converted = protobuf_to_dict(s)
    assert (converted == json.loads("""
{
  "field1": {
    "key1": "value1",
    "key2": 3.0,
    "key3": 1.0,
    "key4": true,
    "key5": null
  },
  "field2": "key2"
}
    """))
