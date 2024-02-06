import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, List

import yaml

from .inference_parameter import InferParam
from .output import *  # noqa: F403
from .triton import DType  # noqa
from .triton import Device, DynamicBatching, InputConfig, OutputConfig, TritonModelConfig

logger = logging.getLogger(__name__)

__all__ = ["ModelTypes", "ModelConfigClass", "MODEL_TYPES", "get_model_config", "load_user_config"]


# Clarifai model types
@dataclass
class ModelTypes:
  """ All supported Clarifai model type names
  """
  visual_detector: str = "visual-detector"
  visual_classifier: str = "visual-classifier"
  text_classifier: str = "text-classifier"
  text_to_text: str = "text-to-text"
  text_embedder: str = "text-embedder"
  text_to_image: str = "text-to-image"
  visual_embedder: str = "visual-embedder"
  visual_segmenter: str = "visual-segmenter"
  multimodal_embedder: str = "multimodal-embedder"

  def __post_init__(self):
    self.all = list(asdict(self).values())

  @property
  def image_input_models(self):
    """ Return list of model types having image as input or one of inputs"""
    _visual = [each for each in self.all if each.startswith("visual")]

    return _visual + [self.multimodal_embedder]


@dataclass
class FieldMapsConfig:
  input_fields_map: dict = field(default_factory=dict)
  output_fields_map: dict = field(default_factory=dict)


@dataclass
class ServingBackendConfig:
  """
  """
  triton: TritonModelConfig = None


@dataclass
class ClarifaiModelConfig:
  """Clarifai necessary configs for building/uploading/creation

  Args:
    field_maps (FieldMapsConfig): Field maps config
    output_type (dataclass): model output type
    labels (List[str]): list of concept names
    inference_parameters (List[InferParam]): list of inference parameters
    clarifai_model_id (str): Clarifai model id on the platform
    type (str): one of `MODEL_TYPES`
    clarifai_user_app_id (str): User ID and App ID separated by '/', e.g., <user_id>/<app_id>
    description (str): model description

  """
  field_maps: FieldMapsConfig = None
  output_type: str = None
  labels: List[str] = field(default_factory=list)
  inference_parameters: List[InferParam] = field(default_factory=list)
  clarifai_model_id: str = ""
  type: str = ""
  clarifai_user_app_id: str = ""
  description: str = ""

  def _checking(self, var_name: str, var_value: Any):
    if var_name == "type":
      _model_types = MODEL_TYPES + [""]
      assert self.type in _model_types
    elif var_name == "clarifai_model_id" and var_value:
      # TODO: Must ensure name is valid
      pass
    elif var_name == "clarifai_user_app_id" and var_value:
      _user_app = var_value.split("/")
      assert len(_user_app) == 2, ValueError(
          f"id must be combination of user_id and app_id separated by `/`, e.g. <user_id>/<app_id>. Got {var_value}"
      )
    elif var_name == "labels":
      if var_value:
        assert isinstance(var_value, tuple) or isinstance(
            var_value, list), f"labels must be tuple or list, got {type(var_value)}"
        var_value = [str(each) for each in var_value]

    return var_value

  def __setattr__(self, __name: str, __value: Any) -> None:
    __value = self._checking(__name, __value)

    super().__setattr__(__name, __value)


@dataclass
class ModelConfigClass():
  """All config of model
  Args:
    clarifai_model (ClarifaiModelConfig): Clarifai model config
    serving_backend (ServingBackendConfig): Custom serving backend config. Only support triton for now
  """
  clarifai_model: ClarifaiModelConfig
  serving_backend: ServingBackendConfig

  def make_triton_model_config(
      self,
      model_name: str,
      model_version: str,
      image_shape: List = None,
      instance_group: Device = Device(),
      dynamic_batching: DynamicBatching = DynamicBatching(),
      max_batch_size: int = 1,
      backend: str = "python",
  ) -> TritonModelConfig:

    return TritonModelConfig(
        model_name=model_name,
        model_version=model_version,
        image_shape=image_shape,
        instance_group=instance_group,
        dynamic_batching=dynamic_batching,
        max_batch_size=max_batch_size,
        backend=backend,
        input=self.serving_backend.triton.input,
        output=self.serving_backend.triton.output)

  def dump_to_user_config(self):
    data = asdict(self)
    _self = deepcopy(self)
    # dump backend
    if hasattr(_self.serving_backend, "triton"):
      dict_triton_config = asdict(_self.serving_backend.triton)
      for k, v in dict_triton_config.items():
        if (k == "max_batch_size" and v > 1) \
        or (k == "image_shape" and v != [-1, -1] and self.clarifai_model.type in ModelTypes().image_input_models):
          continue
        else:
          data["serving_backend"]["triton"].pop(k, None)

      if not data["serving_backend"]["triton"]:
        data["serving_backend"].pop("triton", None)
    if not data["serving_backend"]:
      data.pop("serving_backend", None)

    # dump clarifai model
    data["clarifai_model"].pop("field_maps", None)
    data["clarifai_model"].pop("output_type", None)

    return data

  @classmethod
  def custom_doc(cls):
    msg = f"{cls.__doc__}\nWhere: \n\n"
    for k, v in cls.__annotations__.items():
      msg += f"* {k}:\n------\n {v.__doc__}\n"
    return msg


def read_yaml(path: str) -> dict:
  with open(path, encoding="utf-8") as f:
    config = yaml.safe_load(f)  # model dict
  return config


def parse_config(config: dict):
  clarifai_model = config.get("clarifai_model", {})
  serving_backend = config.get("serving_backend", {})
  if serving_backend:
    if serving_backend.get("triton", {}):
      # parse triton input/output
      triton = serving_backend["triton"]
      input_triton_configs = triton.pop("input", {})
      triton.update(
          dict(input=[
              InputConfig(
                  name=input["name"],
                  data_type=eval(f"DType.{input['data_type']}") if isinstance(
                      input['data_type'], str) else input['data_type'],
                  dims=input["dims"],
                  optional=input.get("optional", False),
              ) for input in input_triton_configs
          ]))
      output_triton_configs = triton.pop("output", {})
      triton.update(
          dict(output=[
              OutputConfig(
                  name=output["name"],
                  data_type=eval(f"DType.{output['data_type']}") if isinstance(
                      output['data_type'], str) else output['data_type'],
                  dims=output["dims"],
                  label_filename=output["label_filename"],
              ) for output in output_triton_configs
          ]))
      serving_backend.update(dict(triton=TritonModelConfig(**triton)))
    serving_backend = ServingBackendConfig(**serving_backend)

  # parse field maps for deployment
  field_maps = clarifai_model.pop("field_maps", {})
  clarifai_model.update(dict(field_maps=FieldMapsConfig(**field_maps)))
  # parse inference_parameters
  inference_parameters = clarifai_model.pop("inference_parameters", [])
  if inference_parameters is None:
    inference_parameters = []
  clarifai_model.update(
      dict(inference_parameters=[InferParam(**each) for each in inference_parameters]))
  # parse output type
  output_type = clarifai_model.pop("output_type", None)
  if output_type:
    #if isinstance(output_type, str):
    #  output_type = eval(output_type)
    clarifai_model.update(dict(output_type=output_type))

  clarifai_model = ClarifaiModelConfig(**clarifai_model)

  return ModelConfigClass(clarifai_model=clarifai_model, serving_backend=serving_backend)


def get_model_config(model_type: str) -> ModelConfigClass:
  """
  Get model config by model type

  Args:

    model_type (str): One of field value of ModelTypes

  Return:
    ModelConfigClass

  ### Example:
  >>> from clarifai.models.model_serving.model_config import get_model_config, ModelTypes
  >>> cfg = get_model_config(ModelTypes.text_classifier)
  >>> custom_triton_config = cfg.make_triton_model_config(**kwargs)

  """
  if model_type == "MODEL_TYPE_PLACEHOLDER":
    logger.warning(
        "Warning: A placeholder value has been detected for obtaining the model configuration. This will result in empty `ModelConfigClass` object."
    )
    return ModelConfigClass(clarifai_model=None, serving_backend=None)

  import os
  assert model_type in MODEL_TYPES, f"`model_type` must be in {MODEL_TYPES}"
  cfg = read_yaml(
      os.path.join(os.path.dirname(__file__), "model_types_config", f"{model_type}.yaml"))
  cfg = parse_config(cfg)
  cfg.clarifai_model.type = model_type
  return cfg


_model_types = ModelTypes()
MODEL_TYPES = _model_types.all
del _model_types


def load_user_config(cfg_path: str) -> ModelConfigClass:
  """Read `clarifai_config.yaml` in user working dir

  Args:
      cfg_path (str): path to config

  Returns:
      ModelConfigClass
  """
  cfg = read_yaml(cfg_path)
  return _ensure_user_config(cfg)


def _ensure_user_config(user_config: dict) -> ModelConfigClass:
  """Ensure user config with default one

  Args:
      user_config (dict): ModelConfigClass as dict

  Raises:
      e: Exception when loading user config

  Returns:
      ModelConfigClass
  """

  try:
    user_config_obj: ModelConfigClass = parse_config(user_config)
  except Exception as e:
    raise e

  default_config = get_model_config(user_config_obj.clarifai_model.type)

  for _model_cfg, value in asdict(user_config_obj.clarifai_model).items():

    if value and _model_cfg != "field_maps":
      setattr(default_config.clarifai_model, _model_cfg, value)

  if hasattr(user_config_obj, "serving_backend"):
    if hasattr(user_config_obj.serving_backend, "triton"):
      if user_config_obj.serving_backend.triton.max_batch_size > 1:
        default_config.serving_backend.triton.max_batch_size = user_config_obj.serving_backend.triton.max_batch_size
      if user_config_obj.serving_backend.triton.image_shape != [-1, -1]:
        default_config.serving_backend.triton.image_shape = user_config_obj.serving_backend.triton.image_shape

  return default_config
