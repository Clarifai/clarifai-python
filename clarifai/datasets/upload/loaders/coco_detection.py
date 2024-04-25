#! COCO detection dataset

import os

from ..base import ClarifaiDataLoader

from ..features import VisualDetectionFeatures

#pycocotools is a dependency for this loader
try:
  from pycocotools.coco import COCO
except ImportError:
  raise ImportError("Could not import pycocotools package. "
                    "Please do `pip install 'clarifai[all]'` to import pycocotools.")


class COCODetectionDataLoader(ClarifaiDataLoader):

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
    return "visual_detection"

  def load_data(self) -> None:
    self.coco = COCO(self.label_filepath)
    self.map_ids = {i: img_id for i, img_id in enumerate(list(self.coco.imgs.keys()))}

  def __getitem__(self, index: int):
    value = self.coco.imgs[self.map_ids[index]]
    image_path = os.path.join(self.images_dir, value['file_name'])
    annots = []  # bboxes
    concept_ids = []

    input_ann_ids = self.coco.getAnnIds(imgIds=[value['id']])
    input_anns = self.coco.loadAnns(input_ann_ids)

    for ann in input_anns:
      # get concept info
      # note1: concept_name can be human readable
      # note2: concept_id can only be alphanumeric, up to 32 characters, with no special chars except `-` and `_`
      concept_name = self.coco.cats[ann['category_id']]['name']
      concept_id = concept_name.lower().replace(' ', '-')

      # get bbox information
      # note1: coco bboxes are `[x_min, y_min, width, height]` in pixels
      # note2: clarifai bboxes are `[x_min, y_min, x_max, y_max]` normalized between 0-1.0
      coco_bbox = ann['bbox']
      clarifai_bbox = {
          'left_col': max(0, coco_bbox[0] / value['width']),
          'top_row': max(0, coco_bbox[1] / value['height']),
          'right_col': min(1, (coco_bbox[0] + coco_bbox[2]) / value['width']),
          'bottom_row': min(1, (coco_bbox[1] + coco_bbox[3]) / value['height'])
      }
      if (clarifai_bbox['left_col'] >=
          clarifai_bbox['right_col']) or (clarifai_bbox['top_row'] >= clarifai_bbox['bottom_row']):
        continue
      annots.append([
          clarifai_bbox['left_col'], clarifai_bbox['top_row'], clarifai_bbox['right_col'],
          clarifai_bbox['bottom_row']
      ])
      concept_ids.append(concept_id)

    assert len(concept_ids) == len(annots), f"Num concepts must match num bbox annotations\
        for a single image. Found {len(concept_ids)} concepts and {len(annots)} bboxes."

    return VisualDetectionFeatures(image_path, concept_ids, annots, id=str(value['id']))

  def __len__(self):
    return len(self.coco.imgs)
