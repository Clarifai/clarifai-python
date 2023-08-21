#! Cifar10 Dataset

import csv
import os

from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.datasets.upload.features import VisualClassificationFeatures


class Cifar10DataLoader(ClarifaiDataLoader):
  """Cifar10 Dataset."""

  def __init__(self, split: str = "train"):
    """Initialize dataset params.
    Args:
      split: "train" or "test"
    """
    self.split = split
    self.data_dirs = {
        "train": os.path.join(os.path.dirname(__file__), "cifar_small_train.csv"),
        "test": os.path.join(os.path.dirname(__file__), "cifar_small_test.csv")
    }
    self.data = self.load_data()

  def load_data(self):
    data = []
    with open(self.data_dirs[self.split]) as _file:
      reader = csv.reader(_file)
      next(reader, None)  # skip header
      for review in reader:
        data.append((review[0], review[1]))
    return data

  def __getitem__(self, index):
    item = self.data[index]
    return VisualClassificationFeatures(
        image_path=os.path.join(os.path.dirname(__file__), item[0]),
        label=item[1],
        id=os.path.basename(item[0]).split(".")[0])

  def __len__(self):
    return len(self.data)
