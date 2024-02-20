import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.dataset import Dataset
from clarifai.client.model import Model

try:
  import seaborn as sns
except ImportError:
  raise ImportError("Can not import seaborn. Please run `pip install seaborn` to install it")

try:
  import matplotlib.pyplot as plt
except ImportError:
  raise ImportError("Can not import matplotlib. Please run `pip install matplotlib` to install it")

try:
  from loguru import logger
except ImportError:
  from .logging import get_logger
  logger = get_logger(logger_level="INFO", name=__name__)


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


@dataclass
class _BaseEvalResultHandler:
  model: Model
  eval_data: List[resources_pb2.EvalMetrics] = field(default_factory=list)

  def find_eval_id(self, dataset_info: List[Dict] = [{}]):
    list_eval_outputs = self.model.list_evaluations()
    self.eval_data = []
    for info in dataset_info:
      app_id = info.get("app_id", self.model.auth_helper.app_id)
      dataset_id = info.get("dataset_id", "")
      user_id = info.get("user_id", self.model.auth_helper.user_id)
      version_id = info.get("version_id", "")
      dataset_assert_msg = f"{user_id}/{app_id}/{dataset_id}/{version_id}" if version_id else f"{user_id}/{app_id}/{dataset_id}"
      # checking if dataset exists
      out = Dataset(dataset_id=dataset_id, user_id=user_id, app_id=app_id).list_versions()
      try:
        next(iter(out))
      except Exception as e:
        if any(["CONN_DOES_NOT_EXIST" in _e for _e in e.args]):
          raise Exception(
              f"Dataset {dataset_assert_msg} does not exists. Please check dataset_info args")
        else:
          # caused by sdk failure
          pass
      # checking if model is evaluated with this dataset
      _is_found = False
      for each in list_eval_outputs:
        if each.status.code == status_code_pb2.MODEL_EVALUATED:
          _dataset = each.ground_truth_dataset
          # if version_id is empty -> get latest eval result of dataset,app,user id
          if app_id == _dataset.app_id and dataset_id == _dataset.id and user_id == _dataset.user_id and (
              not version_id or version_id == _dataset.version):
            self.eval_data.append(each)
            _is_found = True
            break
      assert _is_found, f"Model {self.model.model_info.name} in app {self.model.model_info.app_id} is not evaluated successfully with dataset {dataset_assert_msg}"

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
      avg_cols = ["macro_avg" for _ in range(len(avg_x))]
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


########################## Comparison ###########################


def make_handler_by_type(model_type: str) -> _BaseEvalResultHandler:
  _eval_type = get_eval_type(model_type)
  if _eval_type == EvalType.CLASSIFICATION:
    return ClassificationResultHandler
  elif _eval_type == EvalType.DETECTION:
    return DetectionResultHandler
  else:
    return PlaceholderHandler


def to_file_name(x) -> str:
  return x.replace('/', '--')


class CompareMode(Enum):
  MANY_MODELS_TO_ONE_DATA = 0
  ONE_MODEL_TO_MANY_DATA = 1


class EvalResultCompare:
  """Compare evaluation result of models against datasets.
  Note: The module will pick latest result on the datasets.
  and models must be same model type

  Args:
  ---
    models (Union[List[Model], List[str]]): List of Model or urls of models
    dataset_info (Union[Dict, List[Dict[str, str]]]): Dict or list of dict.
      - app_id (str): dataset app id, if None will take model app id
      - dataset_id (str): dataset id
      - user_id (str): dataset user id, if None will take model user id
      - version_id (str): dataset version id, if None will use latest in `list_eval`
  """

  def __init__(self, models: Union[List[Model], List[str]],
               dataset_info: Union[Dict, List[Dict[str, str]]]):
    assert isinstance(models, list), ValueError("Expected list")

    if len(models) > 1:
      self.mode = CompareMode.MANY_MODELS_TO_ONE_DATA
      self.comparator = "Model"
      assert isinstance(dataset_info, dict) or (
          isinstance(dataset_info, list) and len(dataset_info) == 1
      ), f"When comparing multiple models, must provide only one `dataset_info`. However got {dataset_info}"
    else:
      self.mode = CompareMode.ONE_MODEL_TO_MANY_DATA
      self.comparator = "Dataset"

    if not isinstance(dataset_info, list):
      dataset_info = [
          dataset_info,
      ]
    if all(map(lambda x: isinstance(x, str), models)):
      models = [Model(each) for each in models]
    elif not all(map(lambda x: isinstance(x, Model), models)):
      raise ValueError(
          f"Expected all models are list of string or list of Model, got {[type(each) for each in models]}"
      )

    self._eval_handlers: List[_BaseEvalResultHandler] = []
    self.model_type = None
    logger.info("Initializing models...")
    for model in models:
      model.load_info()
      model_type = model.model_info.model_type_id
      if not self.model_type:
        self.model_type = model_type
      else:
        assert self.model_type == model_type, f"Can not compare when model types are different, {self.model_type} != {model_type}"
      m = make_handler_by_type(model_type)(model=model)
      logger.info(f"* {m.get_model_name(pretify=True)}")
      m.find_eval_id(dataset_info=dataset_info)
      self._eval_handlers.append(m)

  @property
  def eval_handlers(self):
    return self._eval_handlers

  def _loop_eval_handlers(self, func_name: str, **kwargs) -> Tuple[list, list]:
    """ Run methods of `eval_handlers[...].model`

    Args:
      func_name (str): method name, see `_BaseEvalResultHandler` child classes
      kwargs: keyword arguments of the method

    Return:
      tuple:
        - list of outputs
        - list of comparator names

    """
    outs = []
    comparators = []
    logger.info(f'Running `{func_name}`')
    for _, each in enumerate(self.eval_handlers):
      for ds_index, _ in enumerate(each.eval_data):
        func = eval(f'each.{func_name}')
        out = func(index=ds_index, **kwargs)

        if self.mode == CompareMode.MANY_MODELS_TO_ONE_DATA:
          name = each.get_model_name(pretify=True)
        else:
          name = each.get_dataset_name_by_index(ds_index, pretify=True)
        if out is None:
          logger.warning(f'{self.comparator}:{name} does not have valid data for `{func_name}`')
          continue
        comparators.append(name)
        outs.append(out)

    # remove app_id if models a
    if self.mode == CompareMode.MANY_MODELS_TO_ONE_DATA:
      apps = set([comp.split('/')[0] for comp in comparators])
      if len(apps) == 1:
        comparators = ['/'.join(comp.split('/')[1:]) for comp in comparators]

    if not outs:
      logger.warning(f'Model type {self.model_type} does not support `{func_name}`')

    return outs, comparators

  def detailed_summary(self,
                       confidence_threshold: float = .5,
                       iou_threshold: float = .5,
                       area: str = "all",
                       bypass_const=False) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
    """
    Retrieve and compute popular metrics of model.

    Args:
      confidence_threshold (float): confidence threshold, applicable for classification and detection. Default is 0.5
      iou_threshold (float): iou threshold, support in range(0.5, 1., step=0.1) applicable for detection
      area (float): size of area, support {all, small, medium}, applicable for detection

    Return:
      None or tuple of dataframe: df summary per concept and total concepts

    """
    df = []
    total = []
    # loop over all eval_handlers/dataset and call its method
    outs, comparators = self._loop_eval_handlers(
        'detailed_summary',
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        area=area,
        bypass_const=bypass_const)
    for indx, out in enumerate(outs):
      _df, _total = out
      _df[self.comparator] = [comparators[indx] for _ in range(len(_df))]
      _total['Concept'].replace(
          to_replace=['Total'], value=f'{self.comparator}:{comparators[indx]}', inplace=True)
      _total.rename({'Concept': 'Total Concept'}, axis=1, inplace=True)
      df.append(_df)
      total.append(_total)

    if df:
      df = pd.concat(df, axis=0)
      total = pd.concat(total, axis=0)
      return df, total
    else:
      return None

  def confusion_matrix(self, show=True, save_path: str = None,
                       cm_kwargs: dict = {}) -> Union[pd.DataFrame, None]:
    """Return dataframe of confusion matrix
    Args:
        show (bool, optional): Show the chart. Defaults to True.
        save_path (str): path to save rendered chart.
        cm_kwargs (dict): keyword args of `eval_handler[...].model.cm_kwargs` method.
    Returns:
        None or pd.Dataframe, If models don't have confusion matrix, return None
    """
    outs, comparators = self._loop_eval_handlers("confusion_matrix", **cm_kwargs)
    all_dfs = []
    for _, (df, anchor) in enumerate(zip(outs, comparators)):
      df[self.comparator] = [anchor for _ in range(len(df))]
      all_dfs.append(df)

    if all_dfs:
      all_dfs = pd.concat(all_dfs, axis=0)
      if save_path or show:

        def _facet_heatmap(data, **kws):
          data = data.dropna(axis=1)
          data = data.drop(self.comparator, axis=1)
          concepts = data.columns
          colnames = pd.MultiIndex.from_arrays([concepts], names=['Predicted'])
          data.columns = colnames
          ax = sns.heatmap(data, cmap='Blues', annot=True, annot_kws={"fontsize": 8}, **kws)
          ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)
          ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)

        temp = all_dfs.copy()
        temp.columns = ["_".join(pair) for pair in temp.columns]
        with sns.plotting_context(font_scale=5.5):
          g = sns.FacetGrid(
              temp,
              col=self.comparator,
              col_wrap=3,
              aspect=1,
              height=3,
              sharex=False,
              sharey=False,
          )
          cbar_ax = g.figure.add_axes([.92, .3, .02, .4])
          g = g.map_dataframe(
              _facet_heatmap, cbar_ax=cbar_ax, vmin=0, vmax=1, cbar=True, square=True)
          g.set_titles(col_template=str(self.comparator) + ':{col_name}', fontsize=5)
          if show:
            plt.show()
          if save_path:
            g.savefig(save_path)

    return all_dfs if isinstance(all_dfs, pd.DataFrame) else None

  def roc_curve_plot(self,
                     show=True,
                     save_path: str = None,
                     roc_curve_kwargs: dict = {},
                     relplot_kwargs: dict = {}) -> Union[pd.DataFrame, None]:
    """Return dataframe of ROC curve
    Args:
        show (bool, optional): Show the chart. Defaults to True.
        save_path (str): path to save rendered chart.
        pr_curve_kwargs (dict): keyword args of `eval_handler[...].model.roc_curve` method.
        relplot_kwargs (dict): keyword args of `sns.relplot` except {data,x,y,hue,kind,col}.
    Returns:
        None or pd.Dataframe, If models don't have ROC curve, return None
    """
    sns.color_palette("Paired")
    outs, comparator = self._loop_eval_handlers("roc_curve", **roc_curve_kwargs)
    all_dfs = []
    for _, (df, anchor) in enumerate(zip(outs, comparator)):
      df[self.comparator] = [anchor for _ in range(len(df))]
      all_dfs.append(df)

    if all_dfs:
      all_dfs = pd.concat(all_dfs, axis=0)
      if save_path or show:
        g = sns.relplot(
            data=all_dfs,
            x="fpr",
            y="tpr",
            hue='concept',
            kind="line",
            col=self.comparator,
            **relplot_kwargs)
        g.set_titles(col_template=str(self.comparator) + ':{col_name}', fontsize=5)
        if show:
          plt.show()
        if save_path:
          g.savefig(save_path)

    return all_dfs if isinstance(all_dfs, pd.DataFrame) else None

  def pr_plot(self,
              show=True,
              save_path: str = None,
              pr_curve_kwargs: dict = {},
              relplot_kwargs: dict = {}) -> Union[pd.DataFrame, None]:
    """Return dataframe of PR curve
    Args:
        show (bool, optional): Show the chart. Defaults to True.
        save_path (str): path to save rendered chart.
        pr_curve_kwargs (dict): keyword args of `eval_handler[...].model.pr_curve` method.
        relplot_kwargs (dict): keyword args of `sns.relplot` except {data,x,y,hue,kind,col}.
    Returns:
        None or pd.Dataframe, If models don't have PR curve, return None
    """
    sns.color_palette("Paired")
    outs, comparator = self._loop_eval_handlers("pr_curve", **pr_curve_kwargs)
    all_dfs = []
    for _, (df, anchor) in enumerate(zip(outs, comparator)):
      df[self.comparator] = [anchor for _ in range(len(df))]
      all_dfs.append(df)

    if all_dfs:
      all_dfs = pd.concat(all_dfs, axis=0)
      if save_path or show:
        g = sns.relplot(
            data=all_dfs,
            x="recall",
            y="precision",
            hue='concept',
            kind="line",
            col=self.comparator,
            **relplot_kwargs)
        g.set_titles(col_template=str(self.comparator) + ':{col_name}', fontsize=5)
        if show:
          plt.show()
        if save_path:
          g.savefig(save_path)

    return all_dfs if isinstance(all_dfs, pd.DataFrame) else None

  def all(
      self,
      output_folder: str,
      confidence_threshold: float = 0.5,
      iou_threshold: float = 0.5,
      overwrite: bool = False,
      metric_kwargs: dict = {},
      pr_plot_kwargs: dict = {},
      roc_plot_kwargs: dict = {},
  ):
    """Run all comparison methods one by one:
    - detailed_summary
    - pr_curve (if applicable)
    - pr_plot
    - confusion_matrix (if applicable)
    And save to output_folder

    Args:
      output_folder (str): path to output
      confidence_threshold (float): confidence threshold, applicable for classification and detection. Default is 0.5.
      iou_threshold (float): iou threshold, support in range(0.5, 1., step=0.1) applicable for detection.
      overwrite (bool): overwrite result of output_folder.
      metric_kwargs (dict): keyword args for `eval_handler[...].model.{method}`, except for {confidence_threshold, iou_threshold}.
      roc_plot_kwargs (dict): for relplot_kwargs of `roc_curve_plot` method.
      pr_plot_kwargs (dict): for relplot_kwargs of `pr_plot` method.
    """
    eval_type = get_eval_type(self.model_type)
    area = metric_kwargs.pop("area", "all")
    bypass_const = metric_kwargs.pop("bypass_const", False)

    fname = f"conf-{confidence_threshold}"
    if eval_type == EvalType.DETECTION:
      fname = f"{fname}_iou-{iou_threshold}_area-{area}"

    def join_root(*args):
      return os.path.join(output_folder, *args)

    output_folder = join_root(fname)
    if os.path.exists(output_folder) and not overwrite:
      raise RuntimeError(f"{output_folder} exists. If you want to overwrite, set `overwrite=True`")

    os.makedirs(output_folder, exist_ok=True)

    logger.info("Making summary tables...")
    dfs = self.detailed_summary(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        area=area,
        bypass_const=bypass_const)
    if dfs is not None:
      concept_df, total_df = dfs
      concept_df.to_csv(join_root("concepts_summary.csv"))
      total_df.to_csv(join_root("total_summary.csv"))

    curve_metric_kwargs = dict(
        confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)
    curve_metric_kwargs.update(metric_kwargs)

    self.roc_curve_plot(
        show=False,
        save_path=join_root("roc.jpg"),
        roc_curve_kwargs=curve_metric_kwargs,
        relplot_kwargs=roc_plot_kwargs)

    self.pr_plot(
        show=False,
        save_path=join_root("pr.jpg"),
        pr_curve_kwargs=curve_metric_kwargs,
        relplot_kwargs=pr_plot_kwargs)

    self.confusion_matrix(
        show=False, save_path=join_root("confusion_matrix.jpg"), cm_kwargs=curve_metric_kwargs)
