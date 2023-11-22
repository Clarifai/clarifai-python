#! ImageNet Classification dataset

import os

from clarifai.datasets.upload.base import ClarifaiDataLoader
from ..features import VisualClassificationFeatures


class ImageNetDataLoader(ClarifaiDataLoader):
  """ImageNet Dataset."""

  def __init__(self, data_dir, split: str = "train"):
    """
    Initialize dataset params.
    Args:
      data_dir: the local dataset directory.
      split: "train" or "test"
    """
    self.split = split
    self.data_dir = data_dir
    self.label_map = dict()
    self.concepts = []
    self.image_paths = []

    self.load_data()

  def load_data(self):
    #Creating label map
    with open(os.path.join(self.data_dir, "LOC_synset_mapping.txt")) as _file:
      for _id in _file:
        #Removing the spaces,upper quotes and Converting to set to remove repetitions. Then converting to list for compatibility.
        self.label_map[_id.split(" ")[0]] = list({
            "".join(("".join((label.rstrip().lstrip().split(" ")))).split("'"))
            for label in _id[_id.find(" ") + 1:].split(",")
        })

    for _folder in os.listdir(os.path.join(self.data_dir, self.split)):
      try:
        concept = self.label_map[_folder]  #concepts
      except Exception:
        continue
      folder_path = os.path.join(self.data_dir, self.split) + "/" + _folder
      for _img in os.listdir(folder_path):
        if _img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
          self.concepts.append(concept)
          self.image_paths.append(folder_path + "/" + _img)

    assert len(self.concepts) == len(self.image_paths)
    "Number of concepts and images are not equal"

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    return VisualClassificationFeatures(
        image_path=self.image_paths[idx],
        label=self.concepts[idx],
        id=self.image_paths[idx].split('.')[0].split('/')[-1])
