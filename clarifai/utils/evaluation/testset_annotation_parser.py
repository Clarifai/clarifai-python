from typing import List, Tuple

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2 as respb2


def parse_eval_annotation_classifier(
    eval_metrics: respb2.EvalMetrics,
) -> Tuple[np.array, np.array, list, List[respb2.Input]]:
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
    eval_metrics: respb2.EvalMetrics, normalized_box: bool = False, box_style: str = "xyxy"
) -> Tuple[np.array, np.array, list, List[respb2.Input]]:
    BOX_STYLES = ["xyxy", "xywh"]
    assert box_style in BOX_STYLES, ValueError(f"Expected box_style in {BOX_STYLES}")

    concept_ids = list(set([each.concept.id for each in eval_metrics.metrics_by_class]))
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
            if box_style == "xyxy":
                xyxy_concept_score.append([x1, y1, x2, y2, concept_index, score])
            else:
                w = abs(x1 - x2)
                h = abs(y1 - y2)
                xyxy_concept_score.append([x1, y1, w, h, concept_index, score])

        return np.asarray(xyxy_concept_score)

    inputs = []
    pred_xyxy_concept_score = []
    gt_xyxy_concept_score = []
    for input_data in eval_metrics.test_set:
        _input = input_data.input
        img_height = _input.data.image.image_info.height if not normalized_box else 1.0
        img_width = _input.data.image.image_info.height if not normalized_box else 1.0
        _pred_xyxy_concept_score = _get_box_annot(
            input_data.predicted_annotation.data.regions,
            img_height=img_height,
            img_width=img_width,
        )
        _gt_xyxy_concept_score = _get_box_annot(
            input_data.ground_truth_annotation.data.regions,
            img_height=img_height,
            img_width=img_width,
        )

        pred_xyxy_concept_score.append(_pred_xyxy_concept_score)
        gt_xyxy_concept_score.append(_gt_xyxy_concept_score)
        inputs.append(_input)

    return (
        np.asarray(gt_xyxy_concept_score),
        np.asarray(pred_xyxy_concept_score),
        concept_ids,
        inputs,
    )


def parse_eval_annotation_detector_coco(
    eval_metrics: respb2.EvalMetrics,
) -> Tuple[np.array, np.array, list, List[respb2.Input]]:
    gts, preds, concept_ids, inputs = parse_eval_annotation_detector(
        eval_metrics=eval_metrics, normalized_box=False, box_style="xywh"
    )

    def _make_box_annot(data, input_data, accum_id, is_pred=True):
        img_id = input_data.id
        img_url = input_data.data.image.url
        img_height = input_data.data.image.image_info.height
        img_width = input_data.data.image.image_info.height
        image = {
            "id": img_id,
            "file_name": img_url,
            "width": img_width,
            "height": img_height,
        }
        annotations = []
        for i, d in enumerate(data.tolist()):
            area = d[2] * d[3]
            box = {
                "iscrowd": 0,
                "ignore": 0,
                "image_id": img_id,
                "bbox": d[:4],
                "area": area,
                "segmentation": [],
                "category_id": d[4],
                "id": accum_id + i,
            }
            if is_pred:
                box["score"] = d[5]
            annotations.append(box)

        return image, annotations

    categories = [
        {"supercategory": "none", "id": i, "name": label} for i, label in enumerate(concept_ids)
    ]

    accum_pred_ids = 0
    accum_gt_ids = 0
    pred_images = []
    pred_boxes = []
    gt_images = []
    gt_boxes = []
    for ith, input_proto in enumerate(inputs):
        pred = preds[ith]
        gt = gts[ith]
        pred_img, pred_box = _make_box_annot(
            pred, accum_id=accum_pred_ids, input_data=input_proto, is_pred=True
        )
        gt_img, gt_box = _make_box_annot(
            gt, accum_id=accum_gt_ids, input_data=input_proto, is_pred=False
        )

        accum_pred_ids += len(pred)
        pred_images.append(pred_img)
        pred_boxes += pred_box

        accum_gt_ids += len(gt)
        gt_images.append(gt_img)
        gt_boxes += gt_box

    pred_annots = {"images": pred_images, "annotations": pred_boxes, "categories": categories}
    gt_annots = {"images": gt_images, "annotations": gt_boxes, "categories": categories}

    return gt_annots, pred_annots
