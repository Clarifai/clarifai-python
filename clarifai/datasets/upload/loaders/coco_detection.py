#! COCO 2017 detection dataset

import os
import zipfile
from glob import glob

import cv2
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

from clarifai.datasets.upload.base import ClarifaiDataLoader

from ..features import VisualDetectionFeatures


class COCODetectionDataLoader(ClarifaiDataLoader):
  """COCO 2017 Image Detection Dataset."""

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
    self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data")  # data storage directory
    self.extracted_coco_dirs = {"train": None, "val": None, "annotations": None}

    self.load_data()

  def coco_download(self, save_dir):
    """Download coco dataset."""
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    #check if train*, val* and annotation* dirs exist
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

  def load_data(self):
    if isinstance(self.filenames, dict) and len(self.filenames) == 3:
      self.coco_download(self.data_dir)
      self.extracted_coco_dirs["train"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "train" in i][0]
      self.extracted_coco_dirs["val"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "val" in i][0]

      self.extracted_coco_dirs["annotations"] = [os.path.join(self.data_dir, i) \
      for i in os.listdir(self.data_dir) if "annotations" in i][0]
    else:
      raise Exception(f"`filenames` must be a dict of atleast 2 coco zip file names; \
      train, val and annotations. Found {len(self.filenames)} items instead.")

    annot_file = glob(self.extracted_coco_dirs["annotations"] + "/" +\
     f"instances_{self.split}*")[0]
    self.coco = COCO(annot_file)
    categories = self.coco.loadCats(self.coco.getCatIds())
    self.cat_id_map = {category["id"]: category["name"] for category in categories}
    self.cat_img_ids = {}
    for cat_id in list(self.cat_id_map.keys()):
      self.cat_img_ids[cat_id] = self.coco.getImgIds(catIds=[cat_id])

    img_ids = []
    for i in list(self.cat_img_ids.values()):
      img_ids.extend(i)

    self.img_ids = list(set(img_ids))

  def __len__(self):
    return len(self.img_ids)

  def __getitem__(self, idx):
    _id = self.img_ids[idx]
    annots = []  # bboxes
    class_names = []
    labels = [i for i in list(filter(lambda x: _id in self.cat_img_ids[x], self.cat_img_ids))]
    image_path = glob(self.extracted_coco_dirs[self.split]+"/"+\
    f"{str(_id).zfill(12)}*")[0]

    image_height, image_width = cv2.imread(image_path).shape[:2]
    for cat_id in labels:
      annot_ids = self.coco.getAnnIds(imgIds=_id, catIds=[cat_id])
      if len(annot_ids) > 0:
        img_annotations = self.coco.loadAnns(annot_ids)
        for ann in img_annotations:
          class_names.append(self.cat_id_map[cat_id])
          x_min = ann['bbox'][0] / image_width  #left_col
          y_min = ann['bbox'][1] / image_height  #top_row
          x_max = (ann['bbox'][0] + ann['bbox'][2]) / image_width  #right_col
          y_max = (ann['bbox'][1] + ann['bbox'][3]) / image_height  #bottom_row
          annots.append([x_min, y_min, x_max, y_max])
      else:  # if no annotations for given image_id-cat_id pair
        continue
    assert len(class_names) == len(annots), f"Num classes must match num bbox annotations\
    for a single image. Found {len(class_names)} classes and {len(annots)} bboxes."

    return VisualDetectionFeatures(image_path, class_names, annots, id=str(_id))
