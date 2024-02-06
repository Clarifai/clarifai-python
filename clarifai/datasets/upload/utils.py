import importlib
import inspect
import os
from typing import Dict, Tuple, Type, Union

from clarifai_grpc.grpc.api.service_pb2 import MultiDatasetVersionMetricsGroupResponse
from google.protobuf.json_format import MessageToDict

from clarifai.constants.dataset import TASK_TO_ANNOTATION_TYPE
from clarifai.datasets.upload.base import ClarifaiDataLoader
from clarifai.errors import UserError


def load_module_dataloader(module_dir: Union[str, os.PathLike], **kwargs) -> ClarifaiDataLoader:
  """Validate and import dataset module data generator.
  Args:
    `module_dir`: relative path to the module directory
    The directory must contain a `dataset.py` script and the data itself.
    kwargs: keyword arguments to be passed to the dataloader class
  Module Directory Structure:
  ---------------------------
      <folder_name>/
      ├──__init__.py
      ├──<Your local dir dataset>/
      └──dataset.py
  dataset.py must implement a class named following the convention,
  <dataset_name>DataLoader and this class must inherit from base ClarifaiDataLoader()
  """
  module_path = os.path.join(module_dir, "dataset.py")
  spec = importlib.util.spec_from_file_location("dataset", module_path)

  if not spec:
    raise ImportError(f"Module not found at {module_path}")

  # Load the module using the spec
  dataset = importlib.util.module_from_spec(spec)
  # Execute the module to make its contents available
  spec.loader.exec_module(dataset)

  # get main module class
  main_module_cls = None
  for name, obj in dataset.__dict__.items():
    if inspect.isclass(obj) and "DataLoader" in name:
      main_module_cls = obj
    else:
      continue

  return main_module_cls(**kwargs)


class DisplayUploadStatus:
  """Class to display dataset upload status."""

  def __init__(self, dataloader: ClarifaiDataLoader,
               dataset_metrics_response: Type[MultiDatasetVersionMetricsGroupResponse],
               dataset_info_dict: Dict[str, str],
               pre_upload_stats: Tuple[Dict[str, int], Dict[str, int]]) -> None:
    """Initialize the class.
    Args:
      dataloader: ClarifaiDataLoader object
      dataset_metrics_response: The dataset version metrics response from the server.
      dataset_info_dict: The dataset info dictionary.
      pre_upload_stats: The pre upload stats for the dataset.
    """
    self.dataloader = dataloader
    self.dataset_metrics_response = dataset_metrics_response
    self.dataset_info_dict = dataset_info_dict
    self.pre_upload_stats = pre_upload_stats

    self.display()

  def display(self) -> None:
    """Display the upload status."""
    from rich.console import Console

    local_inputs_count, local_annotations_dict = self.get_dataloader_stats()
    uploaded_inputs_dict, uploaded_annotations_dict = self.get_dataset_version_stats(
        self.dataset_metrics_response)

    # Subtract the pre upload stats from the uploaded stats
    uploaded_inputs_dict = {
        key: int(uploaded_inputs_dict[key]) - int(self.pre_upload_stats[0].get(key, 0))
        for key in uploaded_inputs_dict
    }
    uploaded_annotations_dict = {
        key: uploaded_annotations_dict[key] - self.pre_upload_stats[1].get(key, 0)
        for key in uploaded_annotations_dict
    }

    self.local_annotations_count = sum(local_annotations_dict.values())
    self.uploaded_annotations_count = sum(uploaded_annotations_dict.values())

    local_dataset_dict = {
        "Inputs Count": str(local_inputs_count),
        "Annotations Count": str(local_annotations_dict)
    }
    uploaded_dataset_dict = {
        "Inputs Count": str(uploaded_inputs_dict["total"]),
        "Annotations Count": str(uploaded_annotations_dict)
    }

    panel_layout = self.get_display_layout(local_dataset_dict, uploaded_dataset_dict)

    console = Console()
    console.print(panel_layout)

  def get_dataloader_stats(self) -> Tuple[int, Dict[str, int]]:
    """Get the number of inputs and annotations in a dataloader.

    Returns:
      local_inputs_count (int): total number of inputs in the dataloader
      local_annotations_dict (Dict[str, int]): total number of annotations in the dataloader
    """
    from clarifai.constants.dataset import DATASET_UPLOAD_TASKS

    task = self.dataloader.task
    if task not in DATASET_UPLOAD_TASKS:
      raise UserError(
          "Invalid task, please use one of the following: {}".format(DATASET_UPLOAD_TASKS))
    local_inputs_count = len(self.dataloader)
    local_annotations_dict = dict(concepts=0, bboxes=0, polygons=0)
    for i in range(local_inputs_count):
      key, attr = [(k, v) for k, v in TASK_TO_ANNOTATION_TYPE.get(task).items()][0]
      local_annotations_dict[key] += len(getattr(self.dataloader[i], attr))
    return local_inputs_count, local_annotations_dict

  @staticmethod
  def get_dataset_version_stats(
      dataset_metrics_response: Type[MultiDatasetVersionMetricsGroupResponse]
  ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Parse the response from the server for the dataset version metrics groups.
    Args:
      dataset_metrics_response: The dataset version metrics response from the server.

    Returns:
      uploaded_inputs_dict (Dict[str, int]): The input statistics for the dataset.
      uploaded_annotations_dict (Dict[str, int]): The annotation statistics for the dataset.
    """
    dataset_statistics = []
    uploaded_inputs_dict = {}
    uploaded_annotations_dict = dict(concepts=0, bboxes=0, polygons=0)
    dict_response = MessageToDict(dataset_metrics_response)

    for data in dict_response["datasetVersionMetricsGroups"]:
      if isinstance(data["value"], str):
        if data["value"].startswith("id-"):
          data["metrics"].update({"Concept": data["value"]})
          data["metrics"].pop("regionLocationMatrix", None)
          dataset_statistics.append(data["metrics"])
        else:
          uploaded_inputs_dict[data["value"]] = data["metrics"]["inputsCount"]

    for ds in dataset_statistics:
      uploaded_annotations_dict["bboxes"] += int(ds["boundingBoxesCount"])
      uploaded_annotations_dict["concepts"] += int(ds["positiveInputTagsCount"])
      uploaded_annotations_dict["polygons"] += int(ds["polygonsCount"])

    return uploaded_inputs_dict, uploaded_annotations_dict

  def _create_layout(self):
    from rich.layout import Layout
    from rich.progress import BarColumn, Progress, TextColumn

    # Create a Layout
    layout = Layout()

    # Add a new task to the progress bar
    progress = Progress(
        "{task.description}",
        BarColumn(bar_width=100),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )

    # Split the layout into top and bottom rows
    layout.split(Layout(name="Progress"), Layout(name="Tables"))

    # Add the progress bar to the top layout
    layout["Progress"].update(progress)

    # Split the bottom layout into two columns
    layout["Tables"].split_row(
        Layout(name="Local Dataset"),
        Layout(name="Uploaded Dataset"),
    )

    # Create a new layout for the panels
    panel_layout = Layout(size=18)

    # Split the panel layout into top and bottom rows
    panel_layout.split(Layout(name="Progress Panel", size=9), Layout(name="Tables Panel", size=9))

    return layout, panel_layout, progress

  def get_display_layout(self, local_dataset_dict: Dict[str, str],
                         uploaded_dataset_dict: Dict[str, str]):
    """Create a layout for the display.

    Args:
      local_dataset_dict (dict): The local dataset stats info dict.
      uploaded_dataset_dict (dict): The uploaded dataset stats info dict.

    Returns:
      panel_layout (Layout): The panel layout for the display.
    """
    from rich.console import Group
    from rich.panel import Panel

    from clarifai.utils.logging import table_from_dict

    local_dataset_table = table_from_dict(
        [local_dataset_dict],
        column_names=["Inputs Count", "Annotations Count"],
        title="[cyan]Local Dataset")
    uploaded_dataset_table = table_from_dict(
        [uploaded_dataset_dict],
        column_names=["Inputs Count", "Annotations Count"],
        title="[cyan]Uploaded Dataset")
    dataset_info_table = table_from_dict(
        [self.dataset_info_dict], column_names=["dataset_id", "user_id", "app_id"])

    layout, panel_layout, progress = self._create_layout()

    # Add a new task to the progress bar
    progress.add_task(
        "[cyan]Inputs Progress:",
        completed=int(uploaded_dataset_dict["Inputs Count"]),
        total=int(local_dataset_dict["Inputs Count"]))
    progress.add_task(
        "[cyan]Annotations Progress:",
        completed=self.uploaded_annotations_count,
        total=self.local_annotations_count)

    # Add the tables to the respective layouts
    layout["Local Dataset"].update(local_dataset_table)
    layout["Uploaded Dataset"].update(uploaded_dataset_table)

    # Create a render group for the progress bar and the additional data
    progress_group = Group(progress, dataset_info_table)

    # Create a panel for the progress bar with a blue border and a suitable heading
    progress_panel = Panel(progress_group, title="[b] Dataset Upload Summary", border_style="blue")

    # Create a panel for the tables comparison with a blue border and a suitable heading
    tables_panel = Panel(
        layout["Tables"],
        title="[b] Dataset Metrics Comparison",
        border_style="blue",
        expand=False)

    # Add the panels to the respective layouts
    panel_layout["Progress Panel"].update(progress_panel)
    panel_layout["Tables Panel"].update(tables_panel)

    return panel_layout
