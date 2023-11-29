import os

import pytest
import yaml

from clarifai.client.workflow import Workflow


@pytest.mark.requires_secrets
def test_export_workflow_general():

  workflow = Workflow("https://clarifai.com/clarifai/main/workflows/General")

  workflow.export('tests/workflow/fixtures/export_general.yml')
  # assert this to the reader result
  with open('tests/workflow/fixtures/general.yml', 'r') as file:
    expected_data = yaml.safe_load(file)
  with open('tests/workflow/fixtures/export_general.yml', 'r') as file:
    actual_data = yaml.safe_load(file)
  assert actual_data == expected_data, f"dicts did not match: actual: {actual_data}"

  # cleanup
  os.remove('tests/workflow/fixtures/export_general.yml')
