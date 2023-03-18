#! IMDB 50k Movie Reviews dataset

import csv
import os

from clarifai.data_upload.datasets.features import TextFeatures


class IMDBMovieReviewsDataset:
  """IMDB 50K Movie Reviews Dataset."""

  def __init__(self, split: str = "train"):
    """
    Initialize dataset params.
    Args:
      data_dir: the local dataset directory.
      split: "train" or "test"
    """
    self.split = split
    self.data_dirs = {
        "train": os.path.join(os.path.dirname(__file__), "train.csv"),
        "test": os.path.join(os.path.dirname(__file__), "test.csv")
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
        yield TextFeatures(
            text=review[0],  # text,
            labels=review[1],  # sentiment,
            id=None)
