import glob
import re

import pytest
import yaml
from schema import SchemaError

from clarifai.workflows.validate import validate


@pytest.mark.parametrize("filename", glob.glob("tests/workflow/fixtures/*.yml"))
def test_validate_fixtures(filename):
  with open(filename, "r") as file:
    validate(yaml.safe_load(file))


def test_validate_invalid_id():
  with pytest.raises(
      SchemaError, match="Key 'id' error:\nRegex(.*) does not match 'id with spaces'"):
    validate({"workflow": {"id": "id with spaces"}})


def test_validate_empty_nodes():
  with pytest.raises(
      SchemaError, match=re.escape("Key 'nodes' error:\nlen([]) should evaluate to True")):
    validate({"workflow": {"id": "workflow-id", "nodes": []}})


def test_validate_invalid_hex_id():
  with pytest.raises(
      SchemaError, match="Key 'model_version_id' error:\nRegex(.*) does not match 'not-a-hex-id'"):
    validate({
        "workflow": {
            "id":
                "workflow-id",
            "nodes": [{
                "id": "node-id",
                "model": {
                    "model_id": "model-id",
                    "model_version_id": "not-a-hex-id",
                },
            }],
        }
    })


def test_validate_upper_hex_id():
  data = validate({
      "workflow": {
          "id":
              "workflow-id",
          "nodes": [{
              "id": "node-id",
              "model": {
                  "model_id": "model-id",
                  "model_version_id": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
              },
          }],
      }
  })
  assert data["workflow"]["nodes"][0]["model"][
      "model_version_id"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_validate_missing_input():
  with pytest.raises(SchemaError, match="missing input 'previous-node-id' for node 'node-id'"):
    validate({
        "workflow": {
            "id":
                "workflow-id",
            "nodes": [{
                "id": "node-id",
                "model": {
                    "model_id": "model-id",
                    "model_version_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                },
                "node_inputs": [{
                    "node_id": "previous-node-id",
                }],
            }],
        }
    })


def test_validate_model_has_model_version_id_and_other_model_fields():
  with pytest.raises(
      SchemaError, match="model should not set model_version_id and other model fields"):
    validate({
        "workflow": {
            "id":
                "workflow-id",
            "nodes": [{
                "id": "node-id",
                "model": {
                    "model_id": "model-id",
                    "model_version_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "description": "hello"
                },
            }],
        }
    })
