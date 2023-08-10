import importlib
import inspect
import os
import sys
from typing import Union

from .base import ClarifaiDataLoader


def load_module_dataloader(module_dir: Union[str, os.PathLike], split: str) -> ClarifaiDataLoader:
  """Validate and import dataset module data generator.
  Args:
    `module_dir`: relative path to the module directory
    The directory must contain a `dataset.py` script and the data itself.
    `split`: "train" or "val"/"test" dataset split
  Module Directory Structure:
  ---------------------------
      <folder_name>/
      ├──__init__.py
      ├──<Your local dir dataset>/
      └──dataset.py
  dataset.py must implement a class named following the convention,
  <dataset_name>Dataset and this class must have a dataloader()
  generator method
  """
  sys.path.append(str(module_dir))

  if not os.path.exists(os.path.join(module_dir, "__init__.py")):
    with open(os.path.join(module_dir, "__init__.py"), "w"):
      pass

  import dataset  # dataset module

  # get main module class
  main_module_cls = None
  for name, obj in dataset.__dict__.items():
    if inspect.isclass(obj) and "DataLoader" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls(split)


def load_dataloader(name: str, split: str) -> ClarifaiDataLoader:
  """Get dataset generator object from dataset loaders.
  Args:
    `name`: dataset module name in datasets/upload/loaders/.
    `split`: "train" or "val"/"test" dataset split
  Returns:
    Data generator object
  """
  loader_dataset = importlib.import_module(f"clarifai.datasets.upload.loaders.{name}")
  # get main module class
  main_module_cls = None
  for name, obj in loader_dataset.__dict__.items():
    if inspect.isclass(obj) and "DataLoader" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls(split)
