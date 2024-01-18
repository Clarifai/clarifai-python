import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import List

import yaml

from .inference_parameter import InferParam
from .output import EmbeddingOutput  # noqa # pylint: disable=unused-import
from .triton import DynamicBatching  # noqa # pylint: disable=unused-import
from .triton import Device, InputConfig, OutputConfig, TritonModelConfig

logger = logging.getLogger(__name__)

__all__ = ["ModelTypes", "ModelConfigClass", "MODEL_TYPES"]


# Clarifai model types
@dataclass
class ModelTypes:
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


@dataclass
class FieldMapsConfig:
  input_fields_map: dict = field(default_factory=dict)
  output_fields_map: dict = field(default_factory=dict)


@dataclass
class ServingBackendConfig:
  triton: TritonModelConfig = None


@dataclass
class ClarfaiModelConfig:

  field_maps: FieldMapsConfig = None
  output_type: dataclass = None
  labels: list = None
  inference_parameters: List[InferParam] = None
  clarifai_model_id: str = ""
  type: str = ""

  def __post_init__(self):
    _model_types = MODEL_TYPES + [""]
    assert self.type in _model_types


@dataclass
class ModelConfigClass():
  clarifai_model: ClarfaiModelConfig
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
      for k, v in asdict(_self.serving_backend.triton).items():
        if (k == "max_batch_size" and v > 1) or (k == "image_shape" and v != [-1, -1]):
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
                  labels=output["labels"],
              ) for output in output_triton_configs
          ]))
      serving_backend.update(dict(triton=TritonModelConfig(**triton)))
    serving_backend = ServingBackendConfig(**serving_backend)

  # parse field maps for deployment
  field_maps = clarifai_model.pop("field_maps", {})
  clarifai_model.update(dict(field_maps=FieldMapsConfig(**field_maps)))
  output_type = clarifai_model.pop("output_type", None)
  if output_type:
    if isinstance(output_type, str):
      output_type = eval(output_type)
    clarifai_model.update(dict(output_type=output_type))

  clarifai_model = ClarfaiModelConfig(**clarifai_model)

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
