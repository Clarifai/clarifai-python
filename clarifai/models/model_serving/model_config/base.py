from typing import Dict, Iterable, List, TypedDict, Union

import numpy as np

from ..constants import IMAGE_TENSOR_NAME, TEXT_TENSOR_NAME
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

      def new_fn(_self,
                 input_data,
                 inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
        # Run child class
        out = other_fn(_self, input_data, inference_parameters=inference_parameters)
        # Run type check
        return base_fn(base, input_data, out)

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

  def _output_type_check(self, input, output):
    output_type = self._config.clarifai_model.output_type
    if isinstance(output, Iterable):
      assert all(
          each.__class__.__name__ == output_type for each in output
      ), f"Expected output is iteration of `{output_type}` type, got iteration `{output}`"
      assert len(output) == len(
          input
      ), f"Input length and output length must be equal, but got input length of {len(input)} and output length of {len(output)}"
    else:
      raise ValueError(f"Expected output is iteration of `{output_type}` type, got `{output}`")
    return output

  def predict(self,
              input_data: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}) -> Iterable:
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

    - inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters.

    Returns:
    --------
      List of one of the `clarifai.models.model_serving.model_config.output` types. Refer to the README/docs
    """
    raise NotImplementedError

  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    raise NotImplementedError


_MultiModalInputTypeDict = TypedDict("_MultiModalInputTypeDict", {
    IMAGE_TENSOR_NAME: np.ndarray,
    TEXT_TENSOR_NAME: str
})


class MultiModalEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.multimodal_embedder)

  def predict(self,
              input_data: List[_MultiModalInputTypeDict],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `multimodal-embedder` model.

    Args:
      input_data (List[_MultiModalInputTypeDict]): List of dict of key-value: `image`(np.ndarray) and `text` (str)
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.multimodal_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)


class TextClassifier(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_classifier)

  def predict(self,
              input_data: List[str],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[ClassifierOutput]:
    """ Custom prediction function for `text-classifier` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_classifier
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    return self.predict(input_data, inference_parameters=inference_parameters)


class TextEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_embedder)

  def predict(self,
              input_data: List[str],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `text-embedder` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    return self.predict(input_data, inference_parameters=inference_parameters)


class TextToImage(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_to_image)

  def predict(
      self,
      input_data: List[str],
      inference_parameters: Dict[str, Union[bool, str, float, int]] = {}) -> Iterable[ImageOutput]:
    """ Custom prediction function for `text-to-image` model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of ImageOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_to_image
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    return self.predict(input_data, inference_parameters=inference_parameters)


class TextToText(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.text_to_text)

  def predict(
      self,
      input_data: List[str],
      inference_parameters: Dict[str, Union[bool, str, float, int]] = {}) -> Iterable[TextOutput]:
    """ Custom prediction function for `text-to-text` (also called as `text generation`) model.

    Args:
      input_data (List[str]): List of text
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of TextOutput
    """
    raise NotImplementedError

  @triton_wrapper.text_to_text
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)


class VisualClassifier(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_classifier)

  def predict(self,
              input_data: List[np.ndarray],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[ClassifierOutput]:
    """ Custom prediction function for `visual-classifier` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of ClassifierOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_classifier
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)


class VisualDetector(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_detector)

  def predict(self,
              input_data: List[np.ndarray],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[VisualDetectorOutput]:
    """ Custom prediction function for `visual-detector` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of VisualDetectorOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_detector
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)

  @staticmethod
  def postprocess(width: int,
                  height: int,
                  labels: list,
                  scores: list,
                  xyxy_boxes: list,
                  max_bbox_count: int = 500) -> VisualDetectorOutput:
    """Convert detection output to Clarifai detector output format

    Args:
        width (int): image width
        height (int): image height
        labels (list): list of labels
        scores (list): list of scores
        xyxy_boxes (list): list of bounding boxes in x_min, y_min, x_max, y_max format
        max_bbox_count (int, optional): Maximum detection result. Defaults to 500.

    Returns:
        VisualDetectorOutput
    """
    assert len(labels) == len(scores) == len(
        xyxy_boxes
    ), f"Length of `labels`, `scores` and `bboxes` must be equal, got {len(labels)}, {len(scores)} and {len(xyxy_boxes)} "
    labels = [[each] for each in labels]
    scores = [[each] for each in scores]
    bboxes = [[x[1] / height, x[0] / width, x[3] / height, x[2] / width]
              for x in xyxy_boxes]  # normalize the bboxes to [0,1] and [y1 x1 y2 x2]
    bboxes = np.clip(bboxes, 0, 1.)
    if len(bboxes) != 0:
      bboxes = np.concatenate((bboxes, np.zeros((max_bbox_count - len(bboxes), 4))))
      scores = np.concatenate((scores, np.zeros((max_bbox_count - len(scores), 1))))
      labels = np.concatenate((labels, np.zeros((max_bbox_count - len(labels), 1),
                                                dtype=np.int32)))
    else:
      bboxes = np.zeros((max_bbox_count, 4), dtype=np.float32)
      scores = np.zeros((max_bbox_count, 1), dtype=np.float32)
      labels = np.zeros((max_bbox_count, 1), dtype=np.int32)

    output = VisualDetectorOutput(
        predicted_bboxes=bboxes, predicted_labels=labels, predicted_scores=scores)

    return output


class VisualEmbedder(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_embedder)

  def predict(self,
              input_data: List[np.ndarray],
              inference_parameters: Dict[str, Union[bool, str, float, int]] = {}
             ) -> Iterable[EmbeddingOutput]:
    """ Custom prediction function for `visual-embedder` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of EmbeddingOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_embedder
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)


class VisualSegmenter(_BaseClarifaiModel):
  _config: ModelConfigClass = get_model_config(ModelTypes.visual_segmenter)

  def predict(
      self,
      input_data: List[np.ndarray],
      inference_parameters: Dict[str, Union[bool, str, float, int]] = {}) -> Iterable[MasksOutput]:
    """ Custom prediction function for `visual-segmenter` model.

    Args:
      input_data (List[np.ndarray]): List of image
      inference_parameters (Dict[str, Union[bool, str, float, int]]): your inference parameters

    Returns:
      list of MasksOutput
    """
    raise NotImplementedError

  @triton_wrapper.visual_segmenter
  def _tritonserver_predict(self,
                            input_data,
                            inference_parameters: Dict[str, Union[bool, str, float, int]] = {}):
    """ This method is invoked within tritonserver, specifically in the model.py of the Python backend. Attempting to execute it outside of the triton environment will result in failure."""
    return self.predict(input_data, inference_parameters=inference_parameters)
