import pytest
import yaml

from clarifai.models.model_serving import (MODEL_TYPES, InferParam, InferParamManager, ModelTypes,
                                           get_model_config)
from clarifai.models.model_serving.model_config.config import _ensure_user_config
from clarifai.models.model_serving.model_config.inference_parameter import InferParamType


def init_config(type,
                clarifai_model_id='',
                clarifai_user_app_id='',
                description='',
                inference_parameters=[],
                labels=[],
                with_be=True,
                max_batch_size=1,
                image_shape=[-1, -1]):
  if with_be:
    config = f"""
  clarifai_model:
    clarifai_model_id: {clarifai_model_id}
    clarifai_user_app_id: {clarifai_user_app_id}
    description: {description}
    inference_parameters: {inference_parameters}
    labels: {labels}
    type: {type}
  serving_backend:
    triton:
      image_shape: {image_shape}
      max_batch_size: {max_batch_size}
  """

  else:
    config = f"""
  clarifai_model:
    clarifai_model_id: {clarifai_model_id}
    clarifai_user_app_id: {clarifai_user_app_id}
    description: {description}
    inference_parameters: {inference_parameters}
    labels: {labels}
    type: {type}
  """

  return yaml.safe_load(config)


def all_types_infer_param_samples() -> list:
  return [
      InferParam(
          path="number_var",
          default_value=1,
          field_type=InferParamType.NUMBER,
          description="number"),
      InferParam(
          path="str_var",
          default_value="string",
          field_type=InferParamType.STRING,
          description="a string"),
      InferParam(
          path="bool_var",
          default_value=False,
          field_type=InferParamType.BOOL,
          description="boolean"),
      InferParam(
          path="secret_var",
          default_value="key",
          field_type=InferParamType.ENCRYPTED_STRING,
          description="a secret"),
  ]


def test_load_config():
  infer_param = InferParamManager(params=all_types_infer_param_samples()).get_list_params()
  kwargs = dict(
      clarifai_model_id="my model",
      clarifai_user_app_id="xyz/abc",
      description="some desc",
      inference_parameters=infer_param,
      labels=["a", "b", "c"],
  )

  def _check_clarifai_cfg(cfg, kwargs):
    cfg = config.clarifai_model
    for k, v in kwargs.items():
      _loaded = getattr(cfg, k)
      assert _loaded == v, f"{m}: loaded {k} has {_loaded} which is different value from input {v}"
    assert cfg.field_maps == get_model_config(m).clarifai_model.field_maps

  for m in MODEL_TYPES:
    # no be
    config = _ensure_user_config(init_config(type=m, **kwargs))
    _check_clarifai_cfg(config.clarifai_model, kwargs)
    # be
    config = _ensure_user_config(init_config(type=m, with_be=True, max_batch_size=4, **kwargs))
    _check_clarifai_cfg(config.clarifai_model, kwargs)
    assert config.serving_backend.triton.max_batch_size == 4

  # image shape
  image_shape = [100, 200]
  for m in MODEL_TYPES:
    # load
    config = _ensure_user_config(
        init_config(type=m, with_be=True, image_shape=image_shape, max_batch_size=4, **kwargs))

    # triton
    _triton_cfg = config.serving_backend.triton
    _loaded_image_shape = getattr(_triton_cfg, "image_shape")

    if m in ModelTypes().image_input_models:
      assert _loaded_image_shape == image_shape, f"{m} triton: loaded image_shape has {_loaded_image_shape} which is different value from input {image_shape}"
      for _input in _triton_cfg.input:
        if "image" in _input.name:
          assert _input.dims[0] == image_shape[0], "Height is not set for image input"
          assert _input.dims[1] == image_shape[1], "Width is not set for image input"
          assert _input.dims[2] == 3, "Image input requires Channel = 3"
    else:
      assert _loaded_image_shape != image_shape, f"{m} triton: can not set image_shape for non-image input model"

  triton_kwargs = dict(image_shape=[2000, 2000], max_batch_size=4)
  for m in ModelTypes().image_input_models:
    with pytest.raises(Exception):
      config = _ensure_user_config(init_config(type=m, with_be=True, **kwargs, **triton_kwargs))


def test_inference_params():
  with pytest.raises(Exception):
    InferParam(path="x", field_type=InferParamType.BOOL, default_value=1, description="")
  with pytest.raises(Exception):
    InferParam(path="x", field_type=InferParamType.NUMBER, default_value="1", description="")
  with pytest.raises(Exception):
    InferParam(path="x", field_type=InferParamType.STRING, default_value=1, description="")
  with pytest.raises(Exception):
    InferParam(
        path="x", field_type=InferParamType.ENCRYPTED_STRING, default_value=True, description="")
  with pytest.raises(Exception):
    InferParam(path="x", field_type=InferParamType.NUMBER, default_value=[], description="")

  kwargs = dict(integer=1, float_num=0.1, text="xyz", _secret="abc", binary=True)
  m = InferParamManager.from_kwargs(**kwargs)
  for param in m.params:
    if param.path == "integer":
      assert param.field_type == InferParamType.NUMBER
      assert param.default_value == kwargs["integer"]
    elif param.path == "float_num":
      assert param.default_value == kwargs["float_num"]
      assert param.field_type == InferParamType.NUMBER
    elif param.path == "text":
      assert param.field_type == InferParamType.STRING
      assert param.default_value == kwargs["text"]
    elif param.path == "_secret":
      assert param.field_type == InferParamType.ENCRYPTED_STRING
      assert param.default_value == kwargs["_secret"]


def test_clarifai_config():

  for m in MODEL_TYPES:
    # Test with wrong user app format
    _ensure_user_config(init_config(type=m, clarifai_user_app_id=""))
    with pytest.raises(Exception):
      _ensure_user_config(init_config(type=m, clarifai_user_app_id="xyz abc"))
    # Test with Non iterable labels
    with pytest.raises(Exception):
      _ensure_user_config(init_config(type=m, labels="12"))
    with pytest.raises(Exception):
      _ensure_user_config(init_config(type=m, labels=12))
