import importlib
import inspect
import os
from typing import Union

from clarifai.datasets.upload.base import ClarifaiDataLoader


def load_module_dataloader(module_dir: Union[str, os.PathLike]) -> ClarifaiDataLoader:
  """Validate and import dataset module data generator.
  Args:
    `module_dir`: relative path to the module directory
    The directory must contain a `dataset.py` script and the data itself.
  Module Directory Structure:
  ---------------------------
      <folder_name>/
      ├──__init__.py
      ├──<Your local dir dataset>/
      └──dataset.py
  dataset.py must implement a class named following the convention,
  <dataset_name>DataLoader and this class must inherit from base ClarifaiDataLoader()
  """
  module_path = os.path.join(module_dir, "dataset.py")
  spec = importlib.util.spec_from_file_location("dataset", module_path)

  if not spec:
    raise ImportError(f"Module not found at {module_path}")

  # Load the module using the spec
  dataset = importlib.util.module_from_spec(spec)
  # Execute the module to make its contents available
  spec.loader.exec_module(dataset)

  # get main module class
  main_module_cls = None
  for name, obj in dataset.__dict__.items():
    if inspect.isclass(obj) and "DataLoader" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls()
