#! COCO image captioning dataset

import os

from pycocotools.coco import COCO

from clarifai.datasets.upload.base import ClarifaiDataLoader

from ..features import VisualClassificationFeatures


class COCOCaptionsDataLoader(ClarifaiDataLoader):
  """COCO Image Captioning Dataset."""

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
    return "visual_captioning"

  def load_data(self) -> None:
    self.coco = COCO(self.label_filepath)
    self.map_ids = {i: img_id for i, img_id in enumerate(list(self.coco.imgs.keys()))}

  def __len__(self):
    return len(self.coco.imgs)

  def __getitem__(self, index):
    value = self.coco.imgs[self.map_ids[index]]
    image_path = os.path.join(self.images_dir, value['file_name'])
    annots = []

    input_ann_ids = self.coco.getAnnIds(imgIds=[value['id']])
    input_anns = self.coco.loadAnns(input_ann_ids)

    for ann in input_anns:
      annots.append(ann['caption'])

    return VisualClassificationFeatures(image_path, labels=annots[0], id=str(value['id']))
