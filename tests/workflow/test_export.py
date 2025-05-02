import os

import pytest
import yaml

from clarifai.client.workflow import Workflow

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


@pytest.mark.requires_secrets
def test_export_workflow_general():
  workflow = Workflow(
    workflow_id="General", user_id="clarifai", app_id="main", base_url=CLARIFAI_API_BASE
  )

  workflow.export('tests/workflow/fixtures/export_general.yml')
  # assert this to the reader result
  with open('tests/workflow/fixtures/general.yml', 'r') as file:
    expected_data = yaml.safe_load(file)
  with open('tests/workflow/fixtures/export_general.yml', 'r') as file:
    actual_data = yaml.safe_load(file)
  assert actual_data == expected_data, f"dicts did not match: actual: {actual_data}"

  # cleanup
  os.remove('tests/workflow/fixtures/export_general.yml')
