#! COCO 2017 Image Segmentation dataset

import gc
import os
from functools import reduce

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from clarifai.data_upload.datasets.features import VisualSegmentationFeatures


class COCOSegmentationDataset:
  """COCO 2017 Image Segmentation Dataset.
  url: https://cocodataset.org/#download
  """

  def __init__(self, split: str = "train"):
    """
    Inititalize dataset params.
    Args:
      split: "train" or "test"
    """
    self.split = split
    self.image_dir = {"train": os.path.join(os.path.dirname(__file__), "images")}
    self.annotations_file = {
        "train":
            os.path.join(os.path.dirname(__file__), "annotations/instances_val2017_subset.json")
    }

  def dataloader(self):
    """
    Transform COCO 2017 segmentation dataset into clarifai proto compatible
    format to uplaod
    Returns:
      VisualSegmentationFeatures type generator.
    """
    coco = COCO(self.annotations_file[self.split])
    categories = coco.loadCats(coco.getCatIds())
    cat_id_map = {category["id"]: category["name"] for category in categories}
    cat_img_ids = {}
    for cat_id in list(cat_id_map.keys()):
      cat_img_ids[cat_id] = coco.getImgIds(catIds=[cat_id])

    img_ids = []
    for i in list(cat_img_ids.values()):
      img_ids.extend(i)

    # Get the image information for the specified image IDs
    image_info = coco.loadImgs(img_ids)
    # Extract the file names from the image information
    image_filenames = {img_id: info['file_name'] for info, img_id in zip(image_info, img_ids)}
    #get annotations for each image id
    for _id in set(img_ids):
      annots = []  # polygons
      class_names = []
      labels = [i for i in list(filter(lambda x: _id in cat_img_ids[x], cat_img_ids))]
      image_path = os.path.join(self.image_dir[self.split], image_filenames[_id])

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
