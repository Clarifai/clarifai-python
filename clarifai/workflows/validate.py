from schema import And, Optional, Regex, Schema, SchemaError, Use

# Non-empty, up to 32-character ASCII strings with internal dashes and underscores.
_id_validator = And(str, lambda s: 0 < len(s) <= 48, Regex(r'^[0-9A-Za-z]+([-_][0-9A-Za-z]+)*$'))

# 32-character hex string, converted to lower-case.
_hex_id_validator = And(str, Use(str.lower), Regex(r'^[0-9a-f]{32}'))


def _model_does_not_have_model_version_id_and_other_fields(m):
  """ Validate that model does not have model_version_id and other model fields."""
  if ('model_version_id' in m) and _model_has_other_fields(m):
    raise SchemaError(f"model should not set model_version_id and other model fields: {m};"
                      f" please remove model_version_id or other model fields.")
  return True


def _model_has_other_fields(m):
  return any(k not in ['model_id', 'model_version_id', 'user_id', 'app_id'] for k in m.keys())


def _workflow_nodes_have_valid_dependencies(nodes):
  """Validate that all inputs to a node are declared before it."""
  node_ids = set()
  for node in nodes:
    for node_input in node.get("node_inputs", []):
      if node_input["node_id"] not in node_ids:
        raise SchemaError(f"missing input '{node_input['node_id']}' for node '{node['id']}'")
    node_ids.add(node["id"])

  return True


_data_schema = Schema({
    "workflow": {
        "id":
            _id_validator,
        "nodes":
            And(
                len,
                [{
                    "id":
                        And(str, len),  # Node IDs are not validated as IDs by the API.
                    "model":
                        And({
                            "model_id": _id_validator,
                            Optional("app_id"): _id_validator,
                            Optional("user_id"): _id_validator,
                            Optional("model_version_id"): _hex_id_validator,
                            Optional("model_type_id"): _id_validator,
                            Optional("description"): str,
                            Optional("output_info"): {
                                Optional("params"): dict,
                            },
                        }, _model_does_not_have_model_version_id_and_other_fields),
                    Optional("node_inputs"):
                        And(len, [{
                            "node_id": And(str, len),
                        }]),
                }],
                _workflow_nodes_have_valid_dependencies),
    },
})


def validate(data):
  return _data_schema.validate(data)
