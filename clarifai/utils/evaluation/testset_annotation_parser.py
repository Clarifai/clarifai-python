from typing import List, Tuple
import numpy as np
from clarifai_grpc.grpc.api import resources_pb2 as respb2


def parse_eval_annotation_classifier(
    eval_metrics: respb2.EvalMetrics) -> Tuple[np.array, np.array, list, List[respb2.Input]]:
  test_set = eval_metrics.test_set
  # get concept ids
  concept_ids = [each.id for each in test_set[0].predicted_concepts]
  concept_ids.sort()
  # get test set
  y_preds = []
  y = []
  inputs = []
  for data in test_set:

    def _to_array(_data):
      cps = [0] * len(concept_ids)
      for each in _data:
        cps[concept_ids.index(each.id)] = each.value
      return np.asarray(cps)

    y_preds.append(_to_array(data.predicted_concepts))
    y.append(_to_array(data.ground_truth_concepts))
    inputs.append(data.input)

  return np.asarray(y), np.asarray(y_preds), concept_ids, inputs


def parse_eval_annotation_detector(
    eval_metrics: respb2.EvalMetrics) -> Tuple[np.array, np.array, list, List[respb2.Input]]:

  concept_ids = [each.id for each in eval_metrics.metrics_by_class]
  concept_ids.sort()

  def _get_box_annot(field, img_height, img_width):
    xyxy_concept_score = []
    for each in field:
      box = each.region_info.bounding_box
      x1 = box.left_col * img_width
      y1 = box.top_row * img_height
      x2 = box.right_col * img_width
      y2 = box.bottom_row * img_height
      score = each.data.concepts[0].value
      concept = each.data.concepts[0].id
      concept_index = concept_ids.index(concept)
      xyxy_concept_score.append([x1, y1, x2, y2, concept_index, score])
    return xyxy_concept_score

  inputs = []
  pred_xyxy_concept_score = []
  gt_xyxy_concept_score = []
  for input_data in eval_metrics.test_set:
    _input = input_data.input
    img_height = _input.data.image.image_info.height
    img_width = _input.data.image.image_info.height
    _pred_xyxy_concept_score = _get_box_annot(
        input_data.predicted_annotations.data.region, img_height=img_height, img_width=img_width)
    _gt_xyxy_concept_score = _get_box_annot(
        input_data.ground_truth_annotations.data.region,
        img_height=img_height,
        img_width=img_width)
    pred_xyxy_concept_score.append(_pred_xyxy_concept_score)
    gt_xyxy_concept_score.append(_gt_xyxy_concept_score)
    inputs.append(_input)

  return np.asarray(gt_xyxy_concept_score), np.asarray(
      pred_xyxy_concept_score), concept_ids, inputs
