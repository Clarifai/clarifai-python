import os

import yaml

from clarifai.client.workflow import Workflow


def test_export_workflow_general():

  workflow = Workflow("https://clarifai.com/clarifai/main/workflows/General")

  workflow.export('tests/workflow/fixtures/export_general.yml')
  # assert this to the reader result
  with open('tests/workflow/fixtures/general.yml', 'r') as file:
    data = yaml.safe_load(file)
  with open('tests/workflow/fixtures/export_general.yml', 'r') as file:
    export_data = yaml.safe_load(file)
  assert export_data == data, f"dicts did not match: actual: {export_data}"

  # cleanup
  os.remove('tests/workflow/fixtures/export_general.yml')
