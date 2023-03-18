#! Cifar10 Dataset

import csv
import os

from clarifai.data_upload.datasets.features import VisualClassificationFeatures


class Cifar10Dataset:
  """Cifar10 Dataset."""

  def __init__(self, split: str = "train"):
    """
    Initialize dataset params.
    Args:
      data_dir: the local dataset directory.
      split: "train" or "test"
    """
    self.split = split
    self.data_dirs = {
        "train": os.path.join(os.path.dirname(__file__), "cifar_small_train.csv"),
        "test": os.path.join(os.path.dirname(__file__), "cifar_small_test.csv")
    }

  def dataloader(self):
    """
    Transform text data into clarifai proto compatible
    format for upload.
    Returns:
      TextFeatures type generator.
    """
    ## Your preprocessing code here
    with open(self.data_dirs[self.split]) as _file:
      reader = csv.reader(_file)
      next(reader, None)  # skip header
      for review in reader:
        yield VisualClassificationFeatures(
            image_path='examples/image_classification/cifar10/' + review[0],
            label=review[1],
            id=None)
