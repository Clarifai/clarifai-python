import datetime
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install
from rich.tree import Tree

install()

# The default logger to use throughout the SDK is defined at bottom of this file.

# For the json logger.
JSON_LOGGER_NAME = "clarifai-json"
JSON_LOG_KEY = 'msg'
JSON_DEFAULT_CHAR_LENGTH = 400
FIELD_BLACKLIST = [
    'msg', 'message', 'account', 'levelno', 'created', 'threadName', 'name', 'processName',
    'module', 'funcName', 'msecs', 'relativeCreated', 'pathname', 'args', 'thread', 'process'
]

# Create thread local storage that the format() call below uses.
# This is only used by the json_logger in the appropriate CLARIFAI_DEPLOY levels.
thread_log_info = threading.local()


def get_logger_context():
  return thread_log_info.__dict__


def set_logger_context(**kwargs):
  thread_log_info.__dict__.update(kwargs)


def clear_logger_context():
  thread_log_info.__dict__.clear()


def restore_logger_context(context):
  thread_log_info.__dict__.clear()
  thread_log_info.__dict__.update(context)


def get_req_id_from_context():
  ctx = get_logger_context()
  return ctx.get('req_id', '')


def display_workflow_tree(nodes_data: List[Dict]) -> None:
  """Displays a tree of the workflow nodes."""
  # Create a mapping of node_id to the list of node_ids that are connected to it.
  node_adj_mapping = defaultdict(list)
  # Create a mapping of node_id to the node data info.
  nodes_data_dict = {}
  for node in nodes_data:
    nodes_data_dict[node["id"]] = node
    if node.get("node_inputs", "") == "":
      node_adj_mapping["Input"].append(node["id"])
    else:
      for node_input in node["node_inputs"]:
        node_adj_mapping[node_input["node_id"]].append(node["id"])

  # Get all leaf nodes.
  leaf_node_ids = set()
  for node_id in list(nodes_data_dict.keys()):
    if node_adj_mapping.get(node_id, "") == "":
      leaf_node_ids.add(node_id)

  def build_node_tree(node_id="Input"):
    """Recursively builds a rich tree of the workflow nodes."""
    # Set the style of the current node.
    style_str = "green" if node_id in leaf_node_ids else "white"

    # Create a Tree object for the current node.
    if node_id != "Input":
      node_table = table_from_dict(
          [nodes_data_dict[node_id]["model"]],
          column_names=["id", "model_type_id", "app_id", "user_id"],
          title="Node: " + node_id)

      tree = Tree(node_table, style=style_str, guide_style="underline2 white")
    else:
      tree = Tree(f"[green] {node_id}", style=style_str, guide_style="underline2 white")

    # Recursively add the child nodes of the current node to the tree.
    for child in node_adj_mapping.get(node_id, []):
      tree.add(build_node_tree(child))

    # Return the tree.
    return tree

  tree = build_node_tree("Input")
  rprint(tree)


def table_from_dict(data: List[Dict], column_names: List[str], title: str = "") -> Table:
  """Use this function for printing tables from a list of dicts."""
  table = Table(title=title, show_lines=False, show_header=True, header_style="blue")
  for column_name in column_names:
    table.add_column(column_name)
  for row in data:
    req_row = [row.get(column_name, "") for column_name in column_names]
    table.add_row(*req_row)
  return table


def _get_library_name() -> str:
  return __name__.split(".")[0]


def _configure_logger(name: str, logger_level: Union[int, str] = logging.NOTSET) -> None:
  """Configure the logger with the specified name."""

  logger = logging.getLogger(name)
  logger.setLevel(logger_level)

  # Remove existing handlers
  for handler in logger.handlers[:]:
    logger.removeHandler(handler)

  # If ENABLE_JSON_LOGGER is 'true' then definitely use json logger.
  # If ENABLE_JSON_LOGGER is 'false' then definitely don't use json logger.
  # If ENABLE_JSON_LOGGER is not set, then use json logger if in k8s.
  enabled_json = os.getenv('ENABLE_JSON_LOGGER', None)
  in_k8s = 'KUBERNETES_SERVICE_HOST' in os.environ
  if enabled_json == 'true' or (in_k8s and enabled_json != 'false'):
    # Add the json handler and formatter
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  else:
    # Add the new rich handler and formatter
    handler = RichHandler(
        rich_tracebacks=True, log_time_format="%Y-%m-%d %H:%M:%S", console=Console(width=255))
    formatter = logging.Formatter('%(name)s:  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(logger_level: Union[int, str] = logging.NOTSET,
               name: Optional[str] = None) -> logging.Logger:
  """Return a logger with the specified name."""

  if name is None:
    name = _get_library_name()

  _configure_logger(name, logger_level)
  return logging.getLogger(name)


def add_file_handler(logger: logging.Logger, file_path: str, log_level: str = 'WARNING') -> None:
  """Add a file handler to the logger."""
  file_handler = logging.FileHandler(file_path)
  file_handler.setLevel(log_level)
  logger.addHandler(file_handler)


def process_log_files(log_file_path: str,) -> tuple:
  """Processes log files to get failed inputs and annotations.

    Args:
        log_file_path (str): path to the log file
    """
  import re
  duplicate_input_ids = []
  failed_input_ids = []
  pattern = re.compile(r'\| +(\d+) +\| +(\S+) +\| +(.+?) +\| +(.+?) +\| +(.+?) +\| +(.+?) \|')
  try:
    with open(log_file_path, 'r') as file:
      log_content = file.read()
      matches = pattern.findall(log_content)
      for match in matches:
        index = int(match[0])
        input_id = match[1]
        status = match[2]
        if status == "Input has a duplicate ID.":
          duplicate_input_ids.append({"Index": index, "Input_ID": input_id})
        else:
          failed_input_ids.append({"Index": index, "Input_ID": input_id})

  except Exception as e:
    print(f"Error Processing log file {log_file_path}:{e}")
    return [], []

  return duplicate_input_ids, failed_input_ids


def display_concept_relations_tree(relations_dict: Dict[str, Any]) -> None:
  """Print all the concept relations of the app in rich tree format.

    Args:
        relations_dict (dict): A dict of concept relations info.
    """
  for parent, children in relations_dict.items():
    tree = Tree(parent)
    for child in children:
      tree.add(child)
    rprint(tree)


def _default_json_default(obj):
  """
  Handle objects that could not be serialized to JSON automatically.

  Coerce everything to strings.
  All objects representing time get output as ISO8601.
  """
  if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
    return obj.isoformat()
  else:
    return _object_to_string_with_truncation(obj)


def _object_to_string_with_truncation(obj) -> str:
  """
  Truncate object string.

  It's preferred to not log objects that could cause triggering this function,
  It's better to extract important parts form them and log them as regular Python types,
  like str or int, which won't be passed to this functon.

  This message brings additional information to the logs
  that could help to find and fix truncation cases.
  - hardcoded part of the message could be used for the looking all entries in logs
  - obj class could help with detail investigation
  """

  objstr = str(obj)
  if len(objstr) > JSON_DEFAULT_CHAR_LENGTH:
    type_name = type(obj).__name__
    truncated = objstr[:JSON_DEFAULT_CHAR_LENGTH]
    objstr = f"{truncated}...[{type_name} was truncated, len={len(objstr)} chars]"
  return objstr


class JsonFormatter(logging.Formatter):

  def __init__(self,
               fmt=None,
               datefmt=None,
               style='%',
               json_cls=None,
               json_default=_default_json_default):
    """
    :param fmt: Config as a JSON string, allowed fields;
           extra: provide extra fields always present in logs
           source_host: override source host name
    :param datefmt: Date format to use (required by logging.Formatter
        interface but not used)
    :param json_cls: JSON encoder to forward to json.dumps
    :param json_default: Default JSON representation for unknown types,
                         by default coerce everything to a string
    """

    if fmt is not None:
      self._fmt = json.loads(fmt)
    else:
      self._fmt = {}
    self.json_default = json_default
    self.json_cls = json_cls
    if 'extra' not in self._fmt:
      self.defaults = {}
    else:
      self.defaults = self._fmt['extra']
    if 'source_host' in self._fmt:
      self.source_host = self._fmt['source_host']
    else:
      try:
        self.source_host = socket.gethostname()
      except Exception:
        self.source_host = ""

  def _build_fields(self, defaults, fields):
    """Return provided fields including any in defaults
    """
    return dict(list(defaults.get('@fields', {}).items()) + list(fields.items()))

  # Override the format function to fit Clarifai
  def format(self, record):
    fields = record.__dict__.copy()

    # logger.info({...}) directly.
    if isinstance(record.msg, dict):
      fields.update(record.msg)
      fields.pop('msg')
      msg = ""
    else:  # logger.info("message", {...})
      if isinstance(record.args, dict):
        fields.update(record.args)
      msg = record.getMessage()
    for k in FIELD_BLACKLIST:
      fields.pop(k, None)
    # Rename 'levelname' to 'level' and make the value lowercase to match Go logs
    level = fields.pop('levelname', None)
    if level:
      fields['level'] = level.lower()

    # Get the thread local data
    req_id = getattr(thread_log_info, 'req_id', None)
    if req_id:
      fields['req_id'] = req_id
    orig_req_id = getattr(thread_log_info, 'orig_req_id', None)
    if orig_req_id:
      fields['orig_req_id'] = orig_req_id
    # Get the thread local data
    requester = getattr(thread_log_info, 'requester', None)
    if requester:
      fields['requester'] = requester

    user_id = getattr(thread_log_info, 'user_id', None)
    if requester:
      fields['user_id'] = user_id

    if hasattr(thread_log_info, 'start_time'):
      #pylint: disable=no-member
      fields['duration_ms'] = (time.time() - thread_log_info.start_time) * 1000

    if 'exc_info' in fields:
      if fields['exc_info']:
        formatted = traceback.format_exception(*fields['exc_info'])
        fields['exception'] = formatted

      fields.pop('exc_info')

    if 'exc_text' in fields and not fields['exc_text']:
      fields.pop('exc_text')

    logr = self.defaults.copy()

    logr.update({
        JSON_LOG_KEY: msg,
        '@timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    })

    logr.update(fields)

    try:
      return json.dumps(logr, default=self.json_default, cls=self.json_cls)
    except Exception:

      type, value, tb = sys.exc_info()
      return json.dumps(
          {
              "msg": f"Fail to format log {type.__name__}({value}), {logr}",
              "formatting_traceback": "\n".join(traceback.format_tb(tb)),
          },
          default=self.json_default,
          cls=self.json_cls,
      )


# the default logger for the SDK.
logger = get_logger(logger_level=os.environ.get("LOG_LEVEL", "INFO"), name="clarifai")
