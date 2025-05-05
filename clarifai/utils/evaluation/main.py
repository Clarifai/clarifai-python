import os
from enum import Enum
from typing import List, Tuple, Union

from clarifai.client.dataset import Dataset
from clarifai.client.model import Model

from .helpers import (
    MACRO_AVG,
    EvalType,
    _BaseEvalResultHandler,
    get_eval_type,
    make_handler_by_type,
)

try:
    import seaborn as sns
except ImportError:
    raise ImportError("Can not import seaborn. Please run `pip install seaborn` to install it")

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "Can not import matplotlib. Please run `pip install matplotlib` to install it"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError("Can not import pandas. Please run `pip install pandas` to install it")

try:
    from loguru import logger
except ImportError:
    from ..logging import logger

__all__ = ['EvalResultCompare']


class CompareMode(Enum):
    MANY_MODELS_TO_ONE_DATA = 0
    ONE_MODEL_TO_MANY_DATA = 1


class EvalResultCompare:
    """Compare evaluation result of models against datasets.
    Note: The module will pick latest result on the datasets.
    and models must be same model type

    Args:
    ---
      models (Union[List[Model], List[str]]): List of Model or urls of models.
      datasets (Union[Dataset, List[Dataset], str, List[str]]): A single or List of Url or Dataset
      attempt_evaluate (bool): Evaluate when model is not evaluated with the datasets.
      auth_kwargs (dict): Additional auth keyword arguments to be passed to the Dataset and Model if using url(s)
    """

    def __init__(
        self,
        models: Union[List[Model], List[str]],
        datasets: Union[Dataset, List[Dataset], str, List[str]],
        attempt_evaluate: bool = False,
        eval_info: dict = None,
        auth_kwargs: dict = {},
    ):
        assert isinstance(models, list), ValueError("Expected list")

        if len(models) > 1:
            self.mode = CompareMode.MANY_MODELS_TO_ONE_DATA
            self.comparator = "Model"
            assert isinstance(datasets, Dataset) or (
                isinstance(datasets, list) and len(datasets) == 1
            ), (
                f"When comparing multiple models, must provide only one `datasets`. However got {datasets}"
            )
        else:
            self.mode = CompareMode.ONE_MODEL_TO_MANY_DATA
            self.comparator = "Dataset"

        # validate models
        if all(map(lambda x: isinstance(x, str), models)):
            models = [Model(each, **auth_kwargs) for each in models]
        elif not all(map(lambda x: isinstance(x, Model), models)):
            raise ValueError(
                f"Expected all models are list of string or list of Model, got {[type(each) for each in models]}"
            )
        # validate datasets
        if not isinstance(datasets, list):
            datasets = [
                datasets,
            ]
        if all(map(lambda x: isinstance(x, str), datasets)):
            datasets = [Dataset(each, **auth_kwargs) for each in datasets]
        elif not all(map(lambda x: isinstance(x, Dataset), datasets)):
            raise ValueError(
                f"Expected datasets must be str, list of string or Dataset, list of Dataset, got {[type(each) for each in datasets]}"
            )
        # Validate models vs datasets together
        self._eval_handlers: List[_BaseEvalResultHandler] = []
        self.model_type = None
        logger.info("Initializing models...")
        for model in models:
            model.load_info()
            model_type = model.model_info.model_type_id
            if not self.model_type:
                self.model_type = model_type
            else:
                assert self.model_type == model_type, (
                    f"Can not compare when model types are different, {self.model_type} != {model_type}"
                )
            m = make_handler_by_type(model_type)(model=model)
            logger.info(f"* {m.get_model_name(pretify=True)}")
            m.find_eval_id(
                datasets=datasets, attempt_evaluate=attempt_evaluate, eval_info=eval_info
            )
            self._eval_handlers.append(m)

    @property
    def eval_handlers(self):
        return self._eval_handlers

    def _loop_eval_handlers(self, func_name: str, **kwargs) -> Tuple[list, list]:
        """Run methods of `eval_handlers[...].model`

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
                    logger.warning(
                        f'{self.comparator}:{name} does not have valid data for `{func_name}`'
                    )
                    continue
                comparators.append(name)
                outs.append(out)

        if self.mode == CompareMode.MANY_MODELS_TO_ONE_DATA:
            apps = set([comp.split('/')[0] for comp in comparators])
            if len(apps) == 1:
                comparators = ['/'.join(comp.split('/')[1:]) for comp in comparators]

        if not outs:
            logger.warning(f'Model type {self.model_type} does not support `{func_name}`')

        return outs, comparators

    def detailed_summary(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        area: str = "all",
        bypass_const=False,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
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
            bypass_const=bypass_const,
        )
        for indx, out in enumerate(outs):
            _df, _total = out
            _df[self.comparator] = [comparators[indx] for _ in range(len(_df))]
            _total['Concept'].replace(
                to_replace=['Total'], value=f'{self.comparator}:{comparators[indx]}', inplace=True
            )
            _total.rename({'Concept': 'Total Concept'}, axis=1, inplace=True)
            df.append(_df)
            total.append(_total)

        if df:
            df = pd.concat(df, axis=0)
            total = pd.concat(total, axis=0)
            return df, total
        else:
            return None

    def confusion_matrix(
        self, show=True, save_path: str = None, cm_kwargs: dict = {}
    ) -> Union[pd.DataFrame, None]:
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
                    ax = sns.heatmap(
                        data, cmap='Blues', annot=True, annot_kws={"fontsize": 8}, **kws
                    )
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
                    cbar_ax = g.figure.add_axes([0.92, 0.3, 0.02, 0.4])
                    g = g.map_dataframe(
                        _facet_heatmap, cbar_ax=cbar_ax, vmin=0, vmax=1, cbar=True, square=True
                    )
                    g.set_titles(col_template=str(self.comparator) + ':{col_name}', fontsize=5)
                    if show:
                        plt.show()
                    if save_path:
                        g.savefig(save_path)

        return all_dfs if isinstance(all_dfs, pd.DataFrame) else None

    @staticmethod
    def _set_default_kwargs(kwargs: dict, var_name: str, value):
        if var_name not in kwargs:
            kwargs.update({var_name: value})
        return kwargs

    @staticmethod
    def _setup_default_lineplot(df: pd.DataFrame, kwargs: dict):
        hue_order = df["concept"].unique().tolist()
        hue_order.remove(MACRO_AVG)
        hue_order.insert(0, MACRO_AVG)
        EvalResultCompare._set_default_kwargs(kwargs, "hue_order", hue_order)

        sizes = {}
        for each in hue_order:
            s = 1.5
            if each == MACRO_AVG:
                s = 4.0
            sizes.update({each: s})
        EvalResultCompare._set_default_kwargs(kwargs, "sizes", sizes)
        EvalResultCompare._set_default_kwargs(kwargs, "size", "concept")

        EvalResultCompare._set_default_kwargs(kwargs, "errorbar", None)
        EvalResultCompare._set_default_kwargs(kwargs, "height", 5)

        return kwargs

    def roc_curve_plot(
        self,
        show=True,
        save_path: str = None,
        roc_curve_kwargs: dict = {},
        relplot_kwargs: dict = {},
    ) -> Union[pd.DataFrame, None]:
        """Return dataframe of ROC curve
        Args:
            show (bool, optional): Show the chart. Defaults to True.
            save_path (str): path to save rendered chart.
            pr_curve_kwargs (dict): keyword args of `eval_handler[...].model.roc_curve` method.
            relplot_kwargs (dict): keyword args of `sns.relplot` except {data,x,y,hue,kind,col}. where x="fpr", y="tpr", hue="concept"
        Returns:
            None or pd.Dataframe, If models don't have ROC curve, return None
        """
        sns.set_palette("Paired")
        outs, comparator = self._loop_eval_handlers("roc_curve", **roc_curve_kwargs)
        all_dfs = []
        for _, (df, anchor) in enumerate(zip(outs, comparator)):
            df[self.comparator] = [anchor for _ in range(len(df))]
            all_dfs.append(df)

        if all_dfs:
            all_dfs = pd.concat(all_dfs, axis=0)
            if save_path or show:
                relplot_kwargs = self._setup_default_lineplot(all_dfs, relplot_kwargs)
                g = sns.relplot(
                    data=all_dfs,
                    x="fpr",
                    y="tpr",
                    hue='concept',
                    kind="line",
                    col=self.comparator,
                    **relplot_kwargs,
                )
                g.set_titles(col_template=str(self.comparator) + ':{col_name}', fontsize=5)
                if show:
                    plt.show()
                if save_path:
                    g.savefig(save_path)

        return all_dfs if isinstance(all_dfs, pd.DataFrame) else None

    def pr_plot(
        self,
        show=True,
        save_path: str = None,
        pr_curve_kwargs: dict = {},
        relplot_kwargs: dict = {},
    ) -> Union[pd.DataFrame, None]:
        """Return dataframe of PR curve
        Args:
            show (bool, optional): Show the chart. Defaults to True.
            save_path (str): path to save rendered chart.
            pr_curve_kwargs (dict): keyword args of `eval_handler[...].model.pr_curve` method.
            relplot_kwargs (dict): keyword args of `sns.relplot` except {data,x,y,hue,kind,col} where x="recall", y="precision", hue="concept"
        Returns:
            None or pd.Dataframe, If models don't have PR curve, return None
        """
        sns.set_palette("Paired")
        outs, comparator = self._loop_eval_handlers("pr_curve", **pr_curve_kwargs)
        all_dfs = []
        for _, (df, anchor) in enumerate(zip(outs, comparator)):
            df[self.comparator] = [anchor for _ in range(len(df))]
            all_dfs.append(df)

        if all_dfs:
            all_dfs = pd.concat(all_dfs, axis=0)
            if save_path or show:
                relplot_kwargs = self._setup_default_lineplot(all_dfs, relplot_kwargs)
                g = sns.relplot(
                    data=all_dfs,
                    x="recall",
                    y="precision",
                    hue='concept',
                    kind="line",
                    col=self.comparator,
                    **relplot_kwargs,
                )
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
            raise RuntimeError(
                f"{output_folder} exists. If you want to overwrite, set `overwrite=True`"
            )

        os.makedirs(output_folder, exist_ok=True)

        logger.info("Making summary tables...")
        dfs = self.detailed_summary(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            area=area,
            bypass_const=bypass_const,
        )
        if dfs is not None:
            concept_df, total_df = dfs
            concept_df.to_csv(join_root("concepts_summary.csv"))
            total_df.to_csv(join_root("total_summary.csv"))

        curve_metric_kwargs = dict(
            confidence_threshold=confidence_threshold, iou_threshold=iou_threshold
        )
        curve_metric_kwargs.update(metric_kwargs)

        self.roc_curve_plot(
            show=False,
            save_path=join_root("roc.jpg"),
            roc_curve_kwargs=curve_metric_kwargs,
            relplot_kwargs=roc_plot_kwargs,
        )

        self.pr_plot(
            show=False,
            save_path=join_root("pr.jpg"),
            pr_curve_kwargs=curve_metric_kwargs,
            relplot_kwargs=pr_plot_kwargs,
        )

        self.confusion_matrix(
            show=False, save_path=join_root("confusion_matrix.jpg"), cm_kwargs=curve_metric_kwargs
        )

        logger.info(f"Done. Your outputs are saved at {output_folder}")
