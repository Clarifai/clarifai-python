# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parse inference model predictions to triton inference responses
per model type.
"""

from functools import wraps
from itertools import zip_longest
from typing import Callable, Dict
import numpy as np

try:
  import triton_python_backend_utils as pb_utils
except ModuleNotFoundError:
  pass


def visual_detector(func: Callable):
  """
  Visual detector type output parser.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_bboxes = []
    out_labels = []
    out_scores = []
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_bboxes.append(preds.predicted_bboxes)
      out_labels.append(preds.predicted_labels)
      out_scores.append(preds.predicted_scores)

    if len(out_bboxes) < 1 or len(out_labels) < 1:
      out_tensor_bboxes = pb_utils.Tensor("predicted_bboxes", np.zeros((0, 4), dtype=np.float32))
      out_tensor_labels = pb_utils.Tensor("predicted_labels", np.zeros((0, 1), dtype=np.int32))
      out_tensor_scores = pb_utils.Tensor("predicted_scores", np.zeros((0, 1), dtype=np.float32))
    else:
      out_tensor_bboxes = pb_utils.Tensor("predicted_bboxes",
                                          np.asarray(out_bboxes, dtype=np.float32))
      out_tensor_labels = pb_utils.Tensor("predicted_labels",
                                          np.asarray(out_labels, dtype=np.int32))
      out_tensor_scores = pb_utils.Tensor("predicted_scores",
                                          np.asarray(out_scores, dtype=np.float32))

    inference_response = pb_utils.InferenceResponse(
        output_tensors=[out_tensor_bboxes, out_tensor_labels, out_tensor_scores])

    return inference_response

  return parse_predictions


def visual_classifier(func: Callable):
  """
  Visual classifier type output parser.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_scores = []
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_scores.append(preds.predicted_scores)

    out_tensor_scores = pb_utils.Tensor("softmax_predictions",
                                        np.asarray(out_scores, dtype=np.float32))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_scores])

    return inference_response

  return parse_predictions


def text_classifier(func: Callable):
  """
  Text classifier type output parser.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_scores = []
    input_data = [in_elem[0].decode() for in_elem in input_data]
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_scores.append(preds.predicted_scores)

    out_tensor_scores = pb_utils.Tensor("softmax_predictions",
                                        np.asarray(out_scores, dtype=np.float32))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_scores])

    return inference_response

  return parse_predictions


def text_to_text(func: Callable):
  """
  Text to text type output parser.
  Convert a sequence of text into another e.g. text generation,
  summarization or translation.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_text = []
    input_data = [in_elem[0].decode() for in_elem in input_data]
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_text.append(preds.predicted_text)

    out_text_tensor = pb_utils.Tensor("text", np.asarray(out_text, dtype=object))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_text_tensor])

    return inference_response

  return parse_predictions


def text_embedder(func: Callable):
  """
  Text embedder type output parser.
  Generates embeddings for an input text.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_embeddings = []
    input_data = [in_elem[0].decode() for in_elem in input_data]
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_embeddings.append(preds.embedding_vector)

    out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

    return inference_response

  return parse_predictions


def visual_embedder(func: Callable):
  """
  Visual embedder type output parser.
  Generates embeddings for an input image.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_embeddings = []
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      out_embeddings.append(preds.embedding_vector)

    out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

    return inference_response

  return parse_predictions


def visual_segmenter(func: Callable):
  """
  Visual segmenter type output parser.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    masks = []
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      masks.append(preds.predicted_mask)

    out_mask_tensor = pb_utils.Tensor("predicted_mask", np.asarray(masks, dtype=np.int64))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_mask_tensor])

    return inference_response

  return parse_predictions


def text_to_image(func: Callable):
  """
  Text to image type output parser.
  """

  @wraps(func)
  def parse_predictions(self, input_data: np.ndarray, *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    gen_images = []
    input_data = [in_elem[0].decode() for in_elem in input_data]
    for item in input_data:
      preds = func(self, item, *args, **kwargs)
      gen_images.append(preds.image)

    out_image_tensor = pb_utils.Tensor("image", np.asarray(gen_images, dtype=np.uint8))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_image_tensor])

    return inference_response

  return parse_predictions


def multimodal_embedder(func: Callable):
  """
  Multimodal embedder type output parser.
  Generates embeddings for image or text input.
  """

  @wraps(func)
  def parse_predictions(self, input_data: Dict[str, np.ndarray], *args, **kwargs):
    """
    Format predictions and return clarifai compatible output.
    """
    out_embeddings = []
    for group in zip_longest(*input_data.values()):
      _input_data = dict(zip(input_data, group))
      for k, v in _input_data.items():
        # decode np.object to string
        if isinstance(v, np.ndarray) and v.dtype == np.object_:
          _input_data.update({k: v[0].decode()})
      preds = func(self, _input_data, *args, **kwargs)
      out_embeddings.append(preds.embedding_vector)

    out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
    inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

    return inference_response

  return parse_predictions
