import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.dataset import Dataset
from clarifai.client.model import Model

try:
  import pandas as pd
except ImportError:
  raise ImportError("Can not import pandas. Please run `pip install pandas` to install it")

try:
  from loguru import logger
except ImportError:
  from ..logging import logger

MACRO_AVG = "macro_avg"


class EvalType(Enum):
  UNDEFINED = 0
  CLASSIFICATION = 1
  DETECTION = 2
  CLUSTERING = 3
  SEGMENTATION = 4
  TRACKER = 5


def get_eval_type(model_type):
  if "classifier" in model_type:
    return EvalType.CLASSIFICATION
  elif "visual-detector" in model_type:
    return EvalType.DETECTION
  elif "segmenter" in model_type:
    return EvalType.SEGMENTATION
  elif "embedder" in model_type:
    return EvalType.CLUSTERING
  elif "tracker" in model_type:
    return EvalType.TRACKER
  else:
    return EvalType.UNDEFINED


def to_file_name(x) -> str:
  return x.replace('/', '--')


@dataclass
class _BaseEvalResultHandler:
  model: Model
  eval_data: List[resources_pb2.EvalMetrics] = field(default_factory=list)

  def evaluate_and_wait(self, dataset: Dataset, eval_info: dict = None):
    from tqdm import tqdm
    dataset_id = dataset.id
    dataset_app_id = dataset.app_id
    dataset_user_id = dataset.user_id
    _ = self.model.evaluate(
        dataset_id=dataset_id,
        dataset_app_id=dataset_app_id,
        dataset_user_id=dataset_user_id,
        eval_info=eval_info)
    latest_eval = self.model.list_evaluations()[0]
    excepted = 10
    desc = f"Please wait for the evaluation process between model {self.get_model_name()} and dataset {dataset_user_id}/{dataset_app_id}/{dataset_id} to complete."
    bar = tqdm(total=excepted, desc=desc, leave=False, ncols=0)
    while latest_eval.status.code in [
        status_code_pb2.MODEL_EVALUATING, status_code_pb2.MODEL_QUEUED_FOR_EVALUATION
    ]:
      latest_eval = self.model.list_evaluations()[0]
      time.sleep(1)
      bar.update(1)

    if latest_eval.status.code == status_code_pb2.MODEL_EVALUATED:
      return latest_eval
    else:
      raise Exception(
          f"Model has failed to evaluate \n {latest_eval.status}.\nPlease check your dataset inputs!"
      )

  def find_eval_id(self,
                   datasets: List[Dataset] = [],
                   attempt_evaluate: bool = False,
                   eval_info: dict = None):
    list_eval_outputs = self.model.list_evaluations()
    self.eval_data = []
    for dataset in datasets:
      dataset.app_id = dataset.app_id or self.model.auth_helper.app_id
      dataset.user_id = dataset.user_id or self.model.auth_helper.user_id
      dataset_assert_msg = dataset.dataset_info
      # checking if dataset exists
      out = dataset.list_versions()
      try:
        next(iter(out))
      except Exception as e:
        if any(["CONN_DOES_NOT_EXIST" in _e for _e in e.args]):
          raise Exception(
              f"Dataset {dataset_assert_msg} does not exists. Please check datasets args")
        else:
          # caused by sdk failure
          pass
      # checking if model is evaluated with this dataset
      _is_found = False
      for each in list_eval_outputs:
        if each.status.code == status_code_pb2.MODEL_EVALUATED:
          eval_dataset = each.ground_truth_dataset
          # if version_id is empty -> get latest eval result of dataset,app,user id
          if dataset.app_id == eval_dataset.app_id and dataset.id == eval_dataset.id and dataset.user_id == eval_dataset.user_id and (
              not dataset.version.id or dataset.version.id == eval_dataset.version.id):
            # append to eval_data
            self.eval_data.append(each)
            _is_found = True
            break

      # if not evaluated, but user wants to proceed it
      if not _is_found:
        if attempt_evaluate:
          self.eval_data.append(self.evaluate_and_wait(dataset, eval_info=eval_info))
        # otherwise raise error
        else:
          raise Exception(
              f"Model {self.model.model_info.name} in app {self.model.model_info.app_id} is not evaluated yet with dataset {dataset_assert_msg}"
          )

  @staticmethod
  def proto_to_dict(value):
    return MessageToDict(value, preserving_proto_field_name=True)

  @staticmethod
  def _f1(x: float, y: float):
    z = x + y
    return 2 * x * y / z if z else 0.

  def _get_eval(self, index=0, **kwargs):
    logger.info(
        f"Model {self.get_model_name(pretify=True)}: retrieving {kwargs} metrics of dataset: {self.get_dataset_name_by_index(index)}"
    )
    result = self.model.get_eval_by_id(eval_id=self.eval_data[index].id, **kwargs)
    for k, v in kwargs.items():
      if v:
        getattr(self.eval_data[index], k).MergeFrom(getattr(result, k))

  def get_eval_data(self, metric_name: str, index=0):
    if metric_name == 'binary_metrics':
      if len(self.eval_data[index].binary_metrics) == 0:
        self._get_eval(index, binary_metrics=True)
    elif metric_name == 'label_counts':
      if self.proto_to_dict(self.eval_data[index].label_counts) == {}:
        self._get_eval(index, label_counts=True)
    elif metric_name == 'confusion_matrix':
      if self.eval_data[index].confusion_matrix.ByteSize() == 0:
        self._get_eval(index, confusion_matrix=True)
    elif metric_name == 'metrics_by_class':
      if len(self.eval_data[index].metrics_by_class) == 0:
        self._get_eval(index, metrics_by_class=True)
    elif metric_name == 'metrics_by_area':
      if len(self.eval_data[index].metrics_by_area) == 0:
        self._get_eval(index, metrics_by_area=True)

    return getattr(self.eval_data[index], metric_name)

  def get_threshold_index(self, threshold_list: list, selected_value: float = 0.5) -> int:
    assert 0 <= selected_value <= 1 and isinstance(selected_value, float)
    threshold_list = [round(each, 2) for each in threshold_list]

    def parse_precision(x):
      return len(str(x).split(".")[1])

    precision = parse_precision(selected_value)
    if precision > 2:
      selected_value = round(selected_value, 2)
      logger.warning("Round the selected value to .2 decimals")
    return threshold_list.index(selected_value)

  def get_dataset_name_by_index(self, index=0, pretify=True):
    out = self.eval_data[index].ground_truth_dataset
    if pretify:
      app_id = out.app_id
      dataset = out.id
      #out = f"{app_id}/{dataset}/{ver[:5]}" if ver else f"{app_id}/{dataset}"
      if self.model.model_info.app_id == app_id:
        out = dataset
      else:
        out = f"{app_id}/{dataset}"

    return out

  def get_model_name(self, pretify=True):
    model = self.model.model_info
    if pretify:
      app_id = model.app_id
      name = model.id
      ver = model.model_version.id
      model = f"{app_id}/{name}/{ver[:5]}" if ver else f"{app_id}/{name}"

    return model

  def _process_curve(self, data: resources_pb2.BinaryMetrics, metric_name: str, x: str,
                     y: str) -> Dict[str, Dict[str, np.array]]:
    """ Postprocess curve
    """
    x_arr = []
    y_arr = []
    threshold = []
    outputs = []

    def _make_df(xcol, ycol, concept_col, th_col):
      return pd.DataFrame({x: xcol, y: ycol, 'concept': concept_col, 'threshold': th_col})

    for bd in data:
      concept_id = bd.concept.id
      metric = eval(f'bd.{metric_name}')
      if metric.ByteSize() == 0:
        continue
      _x = np.array(eval(f'metric.{x}'))
      _y = np.array(eval(f'metric.{y}'))
      threshold = np.array(metric.thresholds)
      x_arr.append(_x)
      y_arr.append(_y)
      concept_cols = [concept_id for _ in range(len(_x))]
      outputs.append(_make_df(_x, _y, concept_cols, threshold))

    avg_x = np.mean(x_arr, axis=0)
    avg_y = np.mean(y_arr, axis=0)
    if np.isnan(avg_x).all():
      return None
    else:
      avg_cols = [MACRO_AVG for _ in range(len(avg_x))]
    outputs.append(_make_df(avg_x, avg_y, avg_cols, threshold))

    return pd.concat(outputs, axis=0)

  def parse_concept_ids(self, *args, **kwargs) -> List[str]:
    raise NotImplementedError

  def detailed_summary(self, *args, **kwargs):
    raise NotImplementedError

  def pr_curve(self, *args, **kwargs):
    raise NotImplementedError

  def roc_curve(self, *args, **kwargs):
    raise NotImplementedError

  def confusion_matrix(self, *args, **kwargs):
    raise NotImplementedError


@dataclass
class PlaceholderHandler(_BaseEvalResultHandler):

  def parse_concept_ids(self, *args, **kwargs) -> List[str]:
    return None

  def detailed_summary(self, *args, **kwargs):
    return None

  def pr_curve(self, *args, **kwargs):
    return None


@dataclass
class ClassificationResultHandler(_BaseEvalResultHandler):

  def parse_concept_ids(self, index=0) -> List[str]:
    eval_data = self.get_eval_data(metric_name='label_counts', index=index)
    concept_ids = [temp.concept.id for temp in eval_data.positive_label_counts]
    return concept_ids

  def detailed_summary(self, index=0, confidence_threshold: float = 0.5,
                       **kwargs) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Making detailed table per concept and for total concept

    Args:
        index (int, optional): Index of eval dataset. Defaults to 0.
        confidence_threshold (float, optional): confidence threshold. Defaults to 0.5.

    Returns:
        tuple: concepts dataframe, total dataframe
    """
    eval_data = self.get_eval_data('binary_metrics', index=index)
    summary = self.get_eval_data('summary', index=index)

    total_labeled = 0
    total_predicted = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0
    metrics = []

    for bd in eval_data:
      concept_id = bd.concept.id
      if bd.precision_recall_curve.ByteSize() == 0:
        continue
      pr_th_index = self.get_threshold_index(
          list(bd.precision_recall_curve.thresholds), selected_value=confidence_threshold)
      roc_th_index = self.get_threshold_index(
          list(bd.roc_curve.thresholds), selected_value=confidence_threshold)
      if pr_th_index is None or roc_th_index is None:
        continue
      num_pos_labeled = bd.num_pos
      num_neg_labeled = bd.num_neg
      # TP/(TP+FP)
      precision = bd.precision_recall_curve.precision[pr_th_index]
      # TP/(TP+FN)
      recall = bd.precision_recall_curve.recall[pr_th_index]
      # FP/(FP+TN)
      fpr = bd.roc_curve.fpr[roc_th_index]
      # TP/(TP+FN)
      tpr = bd.roc_curve.tpr[roc_th_index]
      # TP+FN
      tp = int(tpr * num_pos_labeled)
      fn = num_pos_labeled - tp
      fp = int(fpr * num_neg_labeled)
      num_pos_pred = tp + fp
      f1 = self._f1(recall, precision)

      total_labeled += num_pos_labeled
      total_predicted += num_pos_pred
      total_fn += fn
      total_tp += tp
      total_fp += fp
      # roc auc, total labelled, predicted, tp, fn, fp, recall, precision, f1
      _d = OrderedDict({
          "Concept": concept_id,
          "Accuracy (ROC AUC)": round(bd.roc_auc, 3),
          "Total Labeled": num_pos_labeled,
          "Total Predicted": num_pos_pred,
          "True Positives": tp,
          "False Negatives": fn,
          "False Positives": fp,
          "Recall": recall,
          "Precision": precision,
          "F1": f1
      })
      metrics.append(pd.DataFrame(_d, index=[0]))

    # If no valid data is found, return None
    if not metrics:
      return None
    # Make per concept df
    df = pd.concat(metrics, axis=0)
    # Make total df
    sum_df_total = sum(df["Total Labeled"])
    precision = sum(df.Precision * df["Total Labeled"]) / sum_df_total if sum_df_total else 0.
    recall = sum(df.Recall * df["Total Labeled"]) / sum_df_total if sum_df_total else 0.
    f1 = self._f1(recall, precision)
    df_total = pd.DataFrame(
        [
            [
                'Total', summary.macro_avg_roc_auc, total_labeled, total_predicted, total_tp,
                total_fn, total_fp, recall, precision, f1
            ],
        ],
        columns=df.columns,
        index=[0])

    return df, df_total

  def pr_curve(self, index=0, **kwargs) -> Union[None, pd.DataFrame]:
    """Making PR curve

    Args:
        index (int, optional): Index of eval dataset. Defaults to 0.

    Returns:
        dictionary: Keys are concept ids and 'macro_avg'. Values are dictionaries of {precision: np.array, recall: np.array}
    """
    eval_data = self.get_eval_data(metric_name='binary_metrics', index=index)
    outputs = self._process_curve(
        eval_data, metric_name='precision_recall_curve', x='recall', y='precision')
    return outputs

  def roc_curve(self, index=0, **kwargs) -> Union[None, pd.DataFrame]:
    eval_data = self.get_eval_data(metric_name='binary_metrics', index=index)
    outputs = self._process_curve(eval_data, metric_name='roc_curve', x='tpr', y='fpr')
    return outputs

  def confusion_matrix(self, index=0, **kwargs):
    eval_data = self.get_eval_data(metric_name='confusion_matrix', index=index)
    concept_ids = self.parse_concept_ids(index)
    concept_ids.sort()
    data = np.zeros((len(concept_ids), len(concept_ids)), np.float32)
    for entry in eval_data.matrix:
      p = entry.predicted_concept.id
      a = entry.actual_concept.id
      if p in concept_ids and a in concept_ids:
        data[concept_ids.index(a), concept_ids.index(p)] = np.around(entry.value, decimals=3)
      else:
        continue
    rownames = pd.MultiIndex.from_arrays([concept_ids], names=['Actual'])
    colnames = pd.MultiIndex.from_arrays([concept_ids], names=['Predicted'])
    df = pd.DataFrame(data, columns=colnames, index=rownames)

    return df


@dataclass
class DetectionResultHandler(_BaseEvalResultHandler):
  AREA_LIST = ["all", "medium", "small"]
  IOU_LIST = list(np.arange(0.5, 1., 0.1))

  def parse_concept_ids(self, index=0) -> List[str]:
    eval_data = self.get_eval_data(metric_name='metrics_by_class', index=index)
    concept_ids = [temp.concept.id for temp in eval_data]
    return concept_ids

  def detailed_summary(self,
                       index=0,
                       confidence_threshold: float = 0.5,
                       iou_threshold: float = 0.5,
                       area: str = "all",
                       bypass_const: bool = False,
                       **kwargs):
    if not bypass_const:
      assert iou_threshold in self.IOU_LIST, f"Expected iou_threshold in {self.IOU_LIST}, got {iou_threshold}"
      assert area in self.AREA_LIST, f"Expected area in {self.AREA_LIST}, got {area}"

    eval_data = self.get_eval_data('metrics_by_class', index=index)
    #summary = self.get_eval_data('summary', index=index)
    metrics = []
    for bd in eval_data:
      # total label
      _iou = round(bd.iou, 1)
      if not (area and bd.area_name == area) or not (iou_threshold and iou_threshold == _iou):
        continue
      concept_id = bd.concept.id
      total = round(bd.num_tot, 3)
      # TP / (TP + FP)
      if len(bd.precision_recall_curve.precision) > 0:
        pr_th_index = self.get_threshold_index(
            list(bd.precision_recall_curve.thresholds), selected_value=confidence_threshold)
        p = round(bd.precision_recall_curve.precision[pr_th_index], 3)
      else:
        p = 0
      # TP / (TP + FN)
      if len(bd.precision_recall_curve.recall) > 0:
        pr_th_index = self.get_threshold_index(
            list(bd.precision_recall_curve.thresholds), selected_value=confidence_threshold)
        r = round(bd.precision_recall_curve.recall[pr_th_index], 3)
      else:
        r = 0
      tp = int(round(r * total, 0))
      fn = total - tp
      fp = float(tp) / p - tp if p else 0
      fp = int(round(fp, 1))
      f1 = self._f1(r, p)
      _d = {
          "Concept": concept_id,
          "Average Precision": round(float(bd.avg_precision), 3),
          "Total Labeled": total,
          "True Positives": tp,
          "False Positives": fp,
          "False Negatives": fn,
          "Recall": r,
          "Precision": p,
          "F1": f1,
      }
      metrics.append(pd.DataFrame(_d, index=[0]))

    if not metrics:
      return None

    df = pd.concat(metrics, axis=0)
    df_total = defaultdict()
    sum_df_total = df["Total Labeled"].sum()
    df_total["Concept"] = "Total"
    df_total["Average Precision"] = df["Average Precision"].mean()
    df_total["Total Labeled"] = sum_df_total
    df_total["True Positives"] = df["True Positives"].sum()
    df_total["False Positives"] = df["False Positives"].sum()
    df_total["False Negatives"] = df["False Negatives"].sum()
    df_total["Recall"] = sum(
        df.Recall * df["Total Labeled"]) / sum_df_total if sum_df_total else 0.
    df_total["Precision"] = df_total["True Positives"] / (
        df_total["True Positives"] + df_total["False Positives"]) if sum_df_total else 0.
    df_total["F1"] = self._f1(df_total["Recall"], df_total["Precision"])
    df_total = pd.DataFrame(df_total, index=[0])

    return [df, df_total]

  def pr_curve(self,
               index=0,
               iou_threshold: float = 0.5,
               area: str = "all",
               bypass_const=False,
               **kwargs):

    if not bypass_const:
      assert iou_threshold in self.IOU_LIST, f"Expected iou_threshold in {self.IOU_LIST}, got {iou_threshold}"
      assert area in self.AREA_LIST, f"Expected area in {self.AREA_LIST}, got {area}"

    eval_data = self.get_eval_data(metric_name='metrics_by_class', index=index)
    _valid_eval_data = []
    for bd in eval_data:
      _iou = round(bd.iou, 1)
      if not (area and bd.area_name == area) or not (iou_threshold and iou_threshold == _iou):
        continue
      _valid_eval_data.append(bd)

    outputs = self._process_curve(
        _valid_eval_data, metric_name='precision_recall_curve', x='recall', y='precision')
    return outputs

  def roc_curve(self, index=0, **kwargs) -> None:
    return None

  def confusion_matrix(self, index=0, **kwargs) -> None:
    return None


def make_handler_by_type(model_type: str) -> _BaseEvalResultHandler:
  _eval_type = get_eval_type(model_type)
  if _eval_type == EvalType.CLASSIFICATION:
    return ClassificationResultHandler
  elif _eval_type == EvalType.DETECTION:
    return DetectionResultHandler
  else:
    return PlaceholderHandler
