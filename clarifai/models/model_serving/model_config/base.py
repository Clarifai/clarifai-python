from typing import Dict, Iterable, List, TypedDict, Union

import numpy as np

from .config import ModelConfigClass, ModelTypes, get_model_config
from .output import (ClassifierOutput, EmbeddingOutput, ImageOutput, MasksOutput, TextOutput,
                     VisualDetectorOutput)
from .triton import wrappers as triton_wrapper


class _TypeCheckModelOutput(type):

  def __new__(cls, name, bases, attrs):
    """
    Override child `predict` function with parent._output_type_check(child.predict).
    Aim to check if child.predict returns valid output type
    """

    def wrap_function(fn_name, base, base_fn, other_fn):

      def new_fn(_self, input_data, inference_paramters: Dict[str, Union[str, float, int]] = {}):
        # Run child class
        out = other_fn(_self, input_data, inference_paramters=inference_paramters)
        # Run type check
        return base_fn(base, out)

      new_fn.__name__ = "wrapped_%s" % fn_name
      new_fn.__doc__ = other_fn.__doc__
      return new_fn

    if name != "_BaseClarifaiModel":
      attrs["predict"] = wrap_function("predict", bases[0],
                                       getattr(bases[0], "_output_type_check", lambda: None),
                                       attrs.setdefault("predict", lambda: None))

    return type.__new__(cls, name, bases, attrs)


class _BaseClarifaiModel(metaclass=_TypeCheckModelOutput):
  _config: ModelConfigClass = None

  @property
  def config(self):
    return self._config

  def _output_type_check(self, x):
    output_type = self._config.clarifai_model.output_type
    if not isinstance(x, Iterable):
      #assert isinstance(x, output_type), f"Expected output is instance of `{output_type}` type, got `{x}`"
      raise ValueError(f"Expected output is iteration of `{output_type}` type, got `{x}`")
    else:
      assert all(each.__class__.__name__ == output_type for each in
                 x), f"Expected output is iteration of `{output_type}` type, got iteration `{x}`"
    return x

  def predict(self,
              input_data: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
              inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable:
    """
    Prediction method.

    Args:
    -----
    - input_data: A list of input data item to predict on. The type depends on model input type:
      * `image`: List[np.ndarray]
      * `text`: List[str]
      * `multimodal`:
        input_data is list of dict where key is input type name e.i. `image`, `text` and value is list.
        {"image": List[np.ndarray], "text": List[str]}

    - inference_paramters (Dict[str, Union[str, float, int]]): your inference parameterss.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.model_config.output` types. Refer to the README/docs
    """
    raise NotImplementedError

  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    raise NotImplementedError


_MultiModalEmbdderInputTypeDict = TypedDict("_MultiModalEmbdderInputTypeDict", {
    "image": List[np.ndarray],
    "text": List[str]
})


class MultiModalEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.multimodal_embedder)

  def predict(
      self,
      input_data: _MultiModalEmbdderInputTypeDict,
      inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `multimodal-embedder` model.

    Args:
      input_data (_MultiModalEmbdderInputTypeDict): dict of key-value: `image`(List[np.ndarray]) and `text` (List[str])
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.multimodal_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)


class TextClassifier(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_classifier)

  def predict(self,
              input_data: List[str],
              inference_paramters: Dict[str, Union[str, float, int]] = {}
             ) -> Iterable[ClassifierOutput]:
    """ Custom prediction function for `text-classifier` model.

    Args:
      input_data (List[str]): List of text
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_classifier
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    return self.predict(input_data, inference_paramters=inference_paramters)


class TextEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_embedder)

  def predict(self,
              input_data: List[str],
              inference_paramters: Dict[str, Union[str, float, int]] = {}
             ) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `text-embedder` model.

    Args:
      input_data (List[str]): List of text
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    return self.predict(input_data, inference_paramters=inference_paramters)


class TextToImage(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_to_image)

  def predict(self,
              input_data: List[str],
              inference_paramters: Dict[str, Union[str, float, int]] = {}
             ) -> Iterable[ImageOutput]:
    """ Custom prediction function for `text-classifier` model.

    Args:
      input_data (List[str]): List of text
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ImageOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_to_image
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    return self.predict(input_data, inference_paramters=inference_paramters)


class TextToText(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_to_text)

  def predict(self,
              input_data: List[str],
              inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable[TextOutput]:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of TextOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_to_text
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)


class VisualClassifier(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_classifier)

  def predict(
      self,
      input_data: List[np.ndarray],
      inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable[ClassifierOutput]:
    """ Custom prediction function for `visual-classifier` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_classifier
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)


class VisualDetector(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_detector)

  def predict(self,
              input_data: List[np.ndarray],
              inference_paramters: Dict[str, Union[str, float, int]] = {}
             ) -> Iterable[VisualDetectorOutput]:
    """ Custom prediction function for `visual-detector` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of VisualDetectorOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_detector
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)


class VisualEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_embedder)

  def predict(
      self,
      input_data: List[np.ndarray],
      inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `visual-embedder` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)


class VisualSegmenter(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_segmenter)

  def predict(
      self,
      input_data: List[np.ndarray],
      inference_paramters: Dict[str, Union[str, float, int]] = {}) -> Iterable[MasksOutput]:
    """ Custom prediction function for `visual-segmenter` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_paramters (Dict[str, Union[str, float, int]]): your inference parameters

    Returns:
      list of MasksOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_segmenter
  def _tritonserver_predict(self,
                            input_data,
                            inference_paramters: Dict[str, Union[str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_paramters=inference_paramters)