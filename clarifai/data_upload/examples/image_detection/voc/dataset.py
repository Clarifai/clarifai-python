#! Pascal VOC Object Detection dataset.

import os

try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

from clarifai.data_upload.datasets.features import VisualDetectionFeatures


class VOCDetectionDataset:
  """Pascal VOC Image Detection Dataset.
  url: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
  """

  voc_concepts = [
      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
      'train', 'tvmonitor'
  ]

  def __init__(self, split: str = "train") -> None:
    """
    Inititalize dataset params.
    Args:
      split: "train" or "test"
    """

    self.split = split
    self.image_dir = {"train": os.path.join(os.path.dirname(__file__), "images")}
    self.annotations_dir = {"train": os.path.join(os.path.dirname(__file__), "annotations")}

  def dataloader(self):
    """
    Transform Pascal VOC detection dataset into clarifai proto compatible
    format to uplaod
    Returns:
      VisualDetectionFeatures type generator.
    """

    all_imgs = os.listdir(self.image_dir[self.split])
    img_ids = [img_filename.split('.')[0] for img_filename in all_imgs]

    for _id in img_ids:
      image_path = os.path.join(self.image_dir[self.split], _id + ".jpg")
      annot_path = os.path.join(self.annotations_dir[self.split], _id + ".xml")

      root = ET.parse(annot_path).getroot()
      size = root.find('size')
      width = float(size.find('width').text)
      height = float(size.find('height').text)

      annots = []
      class_names = []
      for obj in root.iter('object'):
        concept = obj.find('name').text.strip().lower()
        if concept not in self.voc_concepts:
          continue
        xml_box = obj.find('bndbox')
        #Making bounding box to be 0-1
        x_min = max(min((float(xml_box.find('xmin').text) - 1) / width, 1.0), 0.0)  #left_col
        y_min = max(min((float(xml_box.find('ymin').text) - 1) / height, 1.0), 0.0)  #top_row
        x_max = max(min((float(xml_box.find('xmax').text) - 1) / width, 1.0), 0.0)  #right_col
        y_max = max(min((float(xml_box.find('ymax').text) - 1) / height, 1.0), 0.0)  #bottom_row

        if (x_min >= x_max) or (y_min >= y_max):
          continue
        annots.append([x_min, y_min, x_max, y_max])
        class_names.append(concept)

      assert len(class_names) == len(annots), f"Num classes must match num bbox annotations\
            for a single image. Found {len(class_names)} classes and {len(annots)} bboxes."

      yield VisualDetectionFeatures(image_path, class_names, annots, id=_id)
