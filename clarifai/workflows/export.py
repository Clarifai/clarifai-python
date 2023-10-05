from typing import Any, Dict

import yaml
from google.protobuf.json_format import MessageToDict

VALID_YAML_KEYS = ["workflow", "id", "nodes", "node_inputs", "node_id", "model"]


def clean_up_unused_keys(wf: dict):
  """Removes unused keys from dict before exporting to yaml. Supports nested dicts."""
  new_wf = dict()
  for key, val in wf.items():
    if key not in VALID_YAML_KEYS:
      continue
    if key == "model":
      new_wf["model"] = {
          "model_id": wf["model"]["id"],
          "model_version_id": wf["model"]["model_version"]["id"]
      }
      # If the model is not from clarifai main, add the app_id and user_id to the model dict.
      if wf["model"]["user_id"] != "clarifai" and wf["model"]["app_id"] != "main":
        new_wf["model"].update({
            "app_id": wf["model"]["app_id"],
            "user_id": wf["model"]["user_id"]
        })
    elif isinstance(val, dict):
      new_wf[key] = clean_up_unused_keys(val)
    elif isinstance(val, list):
      new_list = []
      for i in val:
        new_list.append(clean_up_unused_keys(i))
      new_wf[key] = new_list
    else:
      new_wf[key] = val
  return new_wf


class Exporter:

  def __init__(self, workflow):
    self.wf = workflow

  def __enter__(self):
    return self

  def parse(self) -> Dict[str, Any]:
    """Reads a resources_pb2.Workflow object (e.g. from a GetWorkflow response)

    Returns:
        dict: A dict representation of the workflow.
    """
    if isinstance(self.wf, list):
      self.wf = self.wf[0]
    wf = {"workflow": MessageToDict(self.wf, preserving_proto_field_name=True)}
    clean_wf = clean_up_unused_keys(wf)
    self.wf_dict = clean_wf
    return clean_wf

  def export(self, out_path):
    with open(out_path, 'w') as out_file:
      yaml.dump(self.wf_dict["workflow"], out_file, default_flow_style=False)

  def __exit__(self, *args):
    self.close()

  def close(self):
    del self.wf
    del self.wf_dict
