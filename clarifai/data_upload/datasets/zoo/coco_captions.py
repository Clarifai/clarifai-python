#! COCO 2017 image captioning dataset

import os
import zipfile
from glob import glob

import requests
from pycocotools.coco import COCO
from tqdm import tqdm

from ..features import VisualClassificationFeatures


class COCOCaptionsDataset:
  """COCO 2017 Image Captioning Dataset."""

  def __init__(self, split: str = "train"):
    """
    Initialize coco dataset.
    Args:
      filenames: the coco zip filenames: Dict[str, str] to be downloaded if download=True,
      data_dir: the local coco dataset directory.
      split: "train" or "val"
    """
    self.filenames = {
        "train": "train2017.zip",
        "val": "val2017.zip",
        "annotations": "annotations_trainval2017.zip"
    }
    self.split = split
    self.url = "http://images.cocodataset.org/zips/"  # coco base image-zip url
    self.data_dir = os.path.join(os.curdir, "data")  # data storage directory
    self.extracted_coco_dirs = {"train": None, "val": None, "annotations": None}

  def coco_download(self, save_dir):
    """Download coco dataset."""
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    #check if train, val and annotation dirs exist
    #so that the coco2017 data isn't downloaded
    for key, filename in self.filenames.items():
      existing_files = glob(f"{save_dir}/{key}*")
      if existing_files:
        print(f"{key} dataset already downloded and extracted")
        continue

      print("-" * 80)
      print(f"Downloading {filename}")
      print("-" * 80)

      if "annotations" in filename:
        self.url = "http://images.cocodataset.org/annotations/"

      response = requests.get(self.url + filename, stream=True)
      response.raise_for_status()
      with open(os.path.join(save_dir, filename), "wb") as _file:
        for chunk in tqdm(response.iter_content(chunk_size=5124000)):
          if chunk:
            _file.write(chunk)
      print("Data download complete...")

      #extract files
      zf = zipfile.ZipFile(os.path.join(save_dir, filename))
      print(f" Extracting {filename} file")
      zf.extractall(path=save_dir)
      # Delete coco zip
      print(f" Deleting {filename}")
      os.remove(path=os.path.join(save_dir, filename))

  def dataloader(self):
    """
    Transform coco image captioning data into clarifai proto compatible
    format for upload.
    Returns:
      VisualClassificationFeatures type generator.
    """
    if isinstance(self.filenames, dict) and len(self.filenames) == 3:  #train, val, annotations
      self.coco_download(self.data_dir)
      self.extracted_coco_dirs["train"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "train" in i][0]
      self.extracted_coco_dirs["val"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "val" in i][0]

      self.extracted_coco_dirs["annotations"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "annotations" in i][0]
    else:
      raise Exception(f"`filenames` must be a dict of atleast 3 coco zip file names; \
      train, val and annotations. Found {len(self.filenames)} items instead.")

    annot_file = glob(self.extracted_coco_dirs["annotations"] + "/" + f"captions_{self.split}*")[0]
    coco = COCO(annot_file)
    annot_ids = coco.getAnnIds()
    annotations = coco.loadAnns(annot_ids)
    for annot in annotations:
      image_path = glob(self.extracted_coco_dirs[self.split]+"/"+\
      f"{str(annot['image_id']).zfill(12)}*")[0]
      # image_captioning and image classification datasets have the same
      # image-label input feature formats
      yield VisualClassificationFeatures(image_path, annot["caption"], id=annot["image_id"])
