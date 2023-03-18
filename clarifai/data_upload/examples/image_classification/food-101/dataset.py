#! Food-101 image classification dataset

import os

from clarifai.data_upload.datasets.features import VisualClassificationFeatures


class Food101Dataset:
  """Food-101 Image Classification Dataset.
  url: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
  """

  def __init__(self, split: str = "train"):
    """
    Initialize dataset params.
    Args:
      data_dir: the local dataset directory.
      split: "train" or "test"
    """
    self.split = split
    self.image_dir = {"train": os.path.join(os.path.dirname(__file__), "images")}

  def dataloader(self):
    """
    Transform food-101 dataset into clarifai proto compatible
    format for upload.
    Returns:
      VisualClassificationFeatures type generator.
    """
    ## Your preprocessing code here
    class_names = os.listdir(self.image_dir[self.split])
    for class_name in class_names:
      for image in os.listdir(os.path.join(self.image_dir[self.split], class_name)):
        image_path = os.path.join(self.image_dir[self.split], class_name, image)
        yield VisualClassificationFeatures(
            image_path=image_path,
            label=class_name,
            id=None  # or image_id
        )
