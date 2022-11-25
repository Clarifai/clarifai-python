import gc
import os
from typing import List, Tuple
from xml.etree import ElementTree as et
import numpy as np
import pandas as pd


def validate_bboxes(bboxes: List) -> List:
  """
  Validate bounding boxes for correctness of dimensions.
  Args:
  	`bboxes`: a list holding a single bounding box in the format,
		 [x_min, y_min, x_max, y_max]
  """
  # Some bbox annotations may have their min dims > max dims
  # Check for these as clarifai annotation post requests will throw errors
  x_min, y_min, x_max, y_max = bboxes

  #assert max_dims > min_dims
  assert x_max > x_min, f"x_max must be greater than x_min. Got x_max:{x_max} x_min:{x_min}"
  assert y_max > y_min, f"y_max must be greater than y_min. Got y_max:{y_max} y_min:{y_min}"

  top_row = [x_min, y_min]
  bottom_row = [x_max, y_max]

  # bound bbox coords between 0 and 1
  top_row = list(np.clip(top_row, 0., 1.))
  bottom_row = list(np.clip(bottom_row, 0., 1.))

  assert bottom_row[0] > top_row[0], f"x_max must be greater than x_min. Got x_max:{bottom_row[0]} x_min:{top_row[0]}"
  assert bottom_row[1] > top_row[1], f"y_max must be greater than y_min. Got y_max:{bottom_row[1]} y_min:{top_row[1]}"

  return top_row + bottom_row


def load_xml_annotations(xml_file: str) -> Tuple:
  """
  Read image, labels and annotations.
  """
  tree = et.parse(xml_file)
  root = tree.getroot()

  all_bboxes = []
  labels = []

  for boxes in root.iter('object'):
    labels.append(boxes.find('name').text)
    img_name = root.find('filename').text
    # image dims relative to the annotator
    image_width = int(root.find('size').find('width').text)
    image_height = int(root.find('size').find('height').text)

    ymin, xmin, ymax, xmax = None, None, None, None

    ymin = int(boxes.find("bndbox/ymin").text)
    xmin = int(boxes.find("bndbox/xmin").text)
    ymax = int(boxes.find("bndbox/ymax").text)
    xmax = int(boxes.find("bndbox/xmax").text)

    xmin_corr = float(xmin / image_width)
    xmax_corr = float(xmax / image_width)
    ymin_corr = float(ymin / image_height)
    ymax_corr = float(ymax / image_height)

    unit_bbox = [xmin_corr, ymin_corr, xmax_corr, ymax_corr]
    valid_bbox = validate_bboxes(unit_bbox)
    all_bboxes.append(valid_bbox)

  assert len(labels) == len(all_bboxes)  # num. of labels must match num. of bboxes
  return img_name, labels, all_bboxes


def load_txt_annotations(csv_file: str) -> Tuple:
  """
  Load annotations and class labels from text files.
  Expected file structure: [class_name, x_min, y_min, x_max, y_max]
  x_min,y_min,x_max,y_max are bounding box coordinates already normalized
  relative to an image's respective width and height dimensions so the values
  expected should range between 0 and 1.
  """
  labels = []
  bboxes = []
  img_name = csv_file.split("/")[-1]  # get filename from path

  with open(csv_file, "r") as annot_file:
    lines = annot_file.readlines()
    for line in lines:
      data = line.split()
      labels.append(data[0])  # class label should be a string
      y_min = float(data[2])
      x_min = float(data[1])
      y_max = float(data[4])
      x_max = float(data[3])
      unit_bbox = [x_min, y_min, x_max, y_max]
      valid_bbox = validate_bboxes(unit_bbox)
      bboxes.append(valid_bbox)

  assert len(labels) == len(bboxes)  # num. of labels must match num. of bboxes
  return img_name, labels, bboxes


def create_image_df(data_dir, labels_dir, from_text_file=False):
  """
  Create image-label-annotations dataframe
  Args:
  	data_dir: image directory
  	labels_dir: labels/annotations directory
  	from_csv: indicates whether csv annotations or xml style annotations
  """
  data_dict = {"id": [], "image": [], "label": [], "bboxes": []}
  if from_text_file == False:

    img_annot_dict = {}
    image_paths = [data_dir + i for i in os.listdir(data_dir)]

    ## Create an image:annot mapping
    for i in image_paths:
      img_annot_dict[str(i)] = labels_dir + f"{i.split('/')[-1][:-4]}.xml"

    ## Extract labels and annotations from xml respective files
    for img_path, annot_path in img_annot_dict.items():
      image_name, labels, annotations = load_xml_annotations(annot_path)

      # create output image-label-annotation dataframe
      data_dict["id"].append(image_name[:-4])
      data_dict["image"].append(img_path)
      data_dict["label"].append(labels)
      data_dict["bboxes"].append(annotations)  # [xmin, ymin, xmax, ymax]

  else:
    img_annot_dict = {}
    image_paths = [data_dir + i for i in os.listdir(data_dir)]

    ## Create an image:annot mapping
    for i in image_paths:
      img_annot_dict[str(
          i
      )] = labels_dir + f"{i.split('/')[-1][:-4]}.txt"  # extract filename and link it to corresp. labels file

    ## Extract labels and annotations from xml respective files
    for img_path, annot_path in img_annot_dict.items():
      image_name, labels, annotations = load_txt_annotations(annot_path)

      # create output image-label-annotation dataframe
      data_dict["id"].append(image_name[:-4])
      data_dict["image"].append(img_path)
      data_dict["label"].append(labels)
      data_dict["bboxes"].append(annotations)  # [xmin, ymin, xmax, ymax]

  image_df = pd.DataFrame(data_dict)

  del data_dict
  gc.collect()

  return image_df


def create_segmentation_df(image_dir: str, masks_dir: str) -> pd.DataFrame:
  """
  Create an image, masked_image and labels dataframe for image
  segmentation data upload.
  Returns:
  	A dataframe with the id, image, label and mask columns.
  Customization Guide:
  	Update `data_dict` by appending the corresponding data to the values lists.
  	id: (str) used to uniquely identify each image.
  	image: (str) holds a list of image paths
  	label: (str) holds a list of labels,.i.e ['car', 'tv', 'car']
  	mask: (str) is a list of paths to images containing masks corresponding to each label
  	in the labels key of an image.i.e.
  		['car_masked_image_path', 'tv_masked_image_path', 'car_masked_image_path']
  	The order of the masks and labels lists matters and they must be of the same length.
  	Each masked image should therefore contain a single mask mapping to a single label.
  """
  data_dict = {"id": [], "image": [], "label": [], "mask": []}

  ## Your image processing code here

  segmentation_df = pd.DataFrame(data_dict)
  return segmentation_df
