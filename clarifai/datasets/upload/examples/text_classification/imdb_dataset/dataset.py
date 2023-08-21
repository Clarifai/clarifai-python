import csv
import os

from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.datasets.upload.features import TextFeatures


class IMDBMovieReviewsDataLoader(ClarifaiDataLoader):
  """IMDB 50K Movie Reviews Dataset."""

  def __init__(self, split: str = "train"):
    """Initialize dataset params.
    Args:
        split: "train" or "test"
    """
    self.split = split
    self.data_dirs = {
        "train": os.path.join(os.path.dirname(__file__), "train.csv"),
        "test": os.path.join(os.path.dirname(__file__), "test.csv")
    }
    self.data = []

    self.load_data()

  def load_data(self):
    with open(self.data_dirs[self.split]) as _file:
      reader = csv.reader(_file)
      next(reader, None)  # skip header
      for review in reader:
        self.data.append({"text": review[0], "labels": review[1], "id": None})

  def __getitem__(self, idx):
    item = self.data[idx]
    return TextFeatures(text=item["text"], labels=item["labels"], id=item["id"])

  def __len__(self):
    return len(self.data)
