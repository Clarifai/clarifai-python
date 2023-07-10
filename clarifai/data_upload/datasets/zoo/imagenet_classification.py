#! ImageNet Classification dataset

import os

from clarifai.data_upload.datasets.features import VisualClassificationFeatures


class ImageNetDataset:
  """ImageNet Dataset."""

  def __init__(self, split: str = "train"):
    """
    Initialize dataset params.
    Args:
      data_dir: the local dataset directory.
      split: "train" or "test"
    """
    self.split = split
    self.data_dir = os.path.join(os.curdir, "data")  # data storage directory
    self.label_map = dict()

  def dataloader(self):
    """
    Transform text data into clarifai proto compatible
    format for upload.
    Returns:
      VisualClassificationFeatures type generator.
    """
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
      except:
        continue
      folder_path = os.path.join(self.data_dir, self.split) + "/" + _folder
      for _img in os.listdir(folder_path):
        if _img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
          yield VisualClassificationFeatures(
              image_path=folder_path + "/" + _img, label=concept, id=None)
