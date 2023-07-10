#! COCO 2017 Image Segmentation dataset

import gc
import os
import zipfile
from functools import reduce
from glob import glob

import cv2
import numpy as np
import requests
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm

from ..features import VisualSegmentationFeatures


class COCOSegmentationDataset:
  """COCO 2017 Image Segmentation Dataset."""

  def __init__(self, split: str = "train"):
    """
    Initialize coco dataset.
    Args:
      filenames: the coco zip filenames: Dict[str, str] to be downloaded if download=True,
      data_dir: the local coco dataset directory
      split: "train" or "val"
    """
    self.filenames = {
        "train": "train2017.zip",
        "val": "val2017.zip",
        "annotations": "annotations_trainval2017.zip"
    }
    self.split = split
    self.url = "http://images.cocodataset.org/zips/"  # coco base image-zip url
    self.data_dir = os.path.join(os.curdir, "data")  # data storage dir
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
      print("Coco data download complete...")

      #extract files
      zf = zipfile.ZipFile(os.path.join(save_dir, filename))
      print(f" Extracting {filename} file")
      zf.extractall(path=save_dir)
      # Delete coco zip
      print(f" Deleting {filename}")
      os.remove(path=os.path.join(save_dir, filename))

  def dataloader(self):
    """
    Transform coco data into clarifai proto compatible format for upload.
    Returns:
      VisualSegmentationFeatures type generator.
    """
    if isinstance(self.filenames, dict) and len(self.filenames) == 3:
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

    annot_file = glob(self.extracted_coco_dirs["annotations"] + "/" +\
     f"instances_{self.split}*")[0]
    coco = COCO(annot_file)
    categories = coco.loadCats(coco.getCatIds())
    cat_id_map = {category["id"]: category["name"] for category in categories}
    cat_img_ids = {}
    for cat_id in list(cat_id_map.keys()):
      cat_img_ids[cat_id] = coco.getImgIds(catIds=[cat_id])

    img_ids = []
    for i in list(cat_img_ids.values()):
      img_ids.extend(i)

    #get annotations for each image id
    for _id in set(img_ids):
      annots = []  # polygons
      class_names = []
      labels = [i for i in list(filter(lambda x: _id in cat_img_ids[x], cat_img_ids))]
      image_path = glob(self.extracted_coco_dirs[self.split]+"/"+\
      f"{str(_id).zfill(12)}*")[0]

      image_height, image_width = cv2.imread(image_path).shape[:2]
      for cat_id in labels:
        annot_ids = coco.getAnnIds(imgIds=_id, catIds=[cat_id])
        if len(annot_ids) > 0:
          img_annotations = coco.loadAnns(annot_ids)
          for ann in img_annotations:
            # get polygons
            if type(ann['segmentation']) == list:
              for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                poly[:, 0], poly[:, 1] = poly[:, 0] / image_width, poly[:, 1] / image_height
                annots.append(poly.tolist())  #[[x=col, y=row],...]
                class_names.append(cat_id_map[cat_id])
            else:  # seg: {"counts":[...]}
              if type(ann['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects([ann['segmentation']], image_height, image_width)
              else:
                rle = ann['segmentation']
              mask = maskUtils.decode(rle)  #binary mask
              #convert mask to polygons and add to annots
              contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
              polygons = []
              for cont in contours:
                if cont.size >= 6:
                  polygons.append(cont.astype(float).flatten().tolist())
              # store polygons in (x,y) pairs
              polygons_flattened = reduce(lambda x, y: x + y, polygons)
              del polygons
              del contours
              del mask
              gc.collect()

              polygons = np.array(polygons_flattened).reshape((int(len(polygons_flattened) / 2),
                                                               2))
              polygons[:, 0] = polygons[:, 0] / image_width
              polygons[:, 1] = polygons[:, 1] / image_height

              annots.append(polygons.tolist())  #[[x=col, y=row],...,[x=col, y=row]]
              class_names.append(cat_id_map[cat_id])
        else:  # if no annotations for given image_id-cat_id pair
          continue
      assert len(class_names) == len(annots), f"Num classes must match num annotations\
      for a single image. Found {len(class_names)} classes and {len(annots)} polygons."

      yield VisualSegmentationFeatures(image_path, class_names, annots, id=_id)
