import typing

import pytest
from rich.console import Console
from rich.tree import Tree


def get_workflow_tree_test_data() -> typing.List[typing.Dict]:
  test_data = [
        {   # Single Branch Single Node
          "adjacency_dict": {"Input": [1]},
          "expected_pattern": r"""
Input
└── 1
"""
        },
        {   # Multi Branch Multiple Nodes
            "adjacency_dict": {"Input": [1, 2], 2: [3, 4, 5], 4: [6, 7], 6: [8]},
            "expected_pattern": r"""
Input
├── 1
└── 2
    ├── 3
    ├── 4
    │   ├── 6
    │   │   └── 8
    │   └── 7
    └── 5
"""
      },
      {   # Single Branch Multiple Nodes
          "adjacency_dict": {"Input": [1], 1: [2], 2: [3]},
          "expected_pattern": r"""
Input
└── 1
    └── 2
        └── 3
"""
      },

  ]

  return test_data


class TestDisplayWorkflowTree:

  def setup_method(self):
    self.console = Console()

  def build_node_tree(self, adj, node_id="Input"):
    """Recursively builds a rich tree of the workflow nodes. Simplified version of the function in clarifai/utils/logging.py"""
    tree = Tree(str(node_id))
    for child in adj.get(node_id, []):
      tree.add(self.build_node_tree(adj, child))
    return tree

  @pytest.mark.parametrize("test_data", get_workflow_tree_test_data())
  def test_display_workflow_tree(self, test_data: typing.Dict):
    tree = self.build_node_tree(test_data["adjacency_dict"])
    with self.console.capture() as capture:
      self.console.print(tree)

    actual_pattern = capture.get()
    assert actual_pattern.strip() == test_data["expected_pattern"].strip()
