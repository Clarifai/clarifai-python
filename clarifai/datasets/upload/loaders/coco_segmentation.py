#! COCO 2017 Image Segmentation dataset

import gc
import os
from functools import reduce

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from clarifai.datasets.upload.base import ClarifaiDataLoader

from ..features import VisualSegmentationFeatures


class COCOSegmentationDataLoader(ClarifaiDataLoader):
  """COCO Image Segmentation Dataset."""

  def __init__(self, images_dir, label_filepath):
    """
    Args:
      images_dir: Directory containing the images.
      label_filepath: Path to the COCO annotation file.
    """
    self.images_dir = images_dir
    self.label_filepath = label_filepath

    self.map_ids = {}
    self.load_data()

  @property
  def task(self):
    return "visual_segmentation"

  def load_data(self) -> None:
    self.coco = COCO(self.label_filepath)
    self.map_ids = {i: img_id for i, img_id in enumerate(list(self.coco.imgs.keys()))}

  def __len__(self):
    return len(self.coco.imgs)

  def __getitem__(self, index):
    """Get image and annotations for a given index."""
    value = self.coco.imgs[self.map_ids[index]]
    image_path = os.path.join(self.images_dir, value['file_name'])
    annots = []  # polygons
    concept_ids = []

    input_ann_ids = self.coco.getAnnIds(imgIds=[value['id']])
    input_anns = self.coco.loadAnns(input_ann_ids)

    for ann in input_anns:
      # get concept info
      # note1: concept_name can be human readable
      # note2: concept_id can only be alphanumeric, up to 32 characters, with no special chars except `-` and `_`
      concept_name = self.coco.cats[ann['category_id']]['name']
      concept_id = concept_name.lower().replace(' ', '-')

      # get polygons
      if isinstance(ann['segmentation'], list):
        poly = np.array(ann['segmentation']).reshape((int(len(ann['segmentation'][0]) / 2),
                                                      2)).astype(float)
        poly[:, 0], poly[:, 1] = poly[:, 0] / value['width'], poly[:, 1] / value['height']
        poly = np.clip(poly, 0, 1)
        annots.append(poly.tolist())  #[[x=col, y=row],...]
        concept_ids.append(concept_id)
      else:  # seg: {"counts":[...]}
        if isinstance(ann['segmentation']['counts'], list):
          rle = maskUtils.frPyObjects([ann['segmentation']], value['height'], value['width'])
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
                                                         2)).astype(float)
        polygons[:, 0] = polygons[:, 0] / value['width']
        polygons[:, 1] = polygons[:, 1] / value['height']
        polygons = np.clip(polygons, 0, 1)
        annots.append(polygons.tolist())  #[[x=col, y=row],...,[x=col, y=row]]
        concept_ids.append(concept_id)

    assert len(concept_ids) == len(annots), f"Num concepts must match num bbox annotations\
    for a single image. Found {len(concept_ids)} concepts and {len(annots)} bboxes."

    return VisualSegmentationFeatures(image_path, concept_ids, annots, id=str(value['id']))
