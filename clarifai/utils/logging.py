import logging
from collections import defaultdict
from typing import Dict, List, Optional

from rich import print as rprint
from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install
from rich.tree import Tree

install()


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


def _configure_logger(logger_level: str = "ERROR") -> None:
  logging.basicConfig(
      level=logger_level,
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[RichHandler(rich_tracebacks=True)])


def get_logger(logger_level: str = "ERROR", name: Optional[str] = None) -> logging.Logger:
  """Return a logger with the specified name."""

  if name is None:
    name = _get_library_name()

  _configure_logger(logger_level)
  return logging.getLogger(name)
