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

from typing import Callable, Dict, List
import numpy as np

try:
  import triton_python_backend_utils as pb_utils
  IN_TRITON = True
except ModuleNotFoundError:
  IN_TRITON = False

from clarifai.models.model_serving.models.output import *


def outputs_to_triton_response(outputs: List[InferenceOutput], model_type: str) \
    -> pb_utils.InferenceResponse if IN_TRITON else None:
  """
  Convert list of InferenceOutput to triton inference response.
  """
  assert IN_TRITON, "converting to triton response format can only be called in triton server"
  return _OUTPUT_CONVERTERS[model_type](outputs)


_OUTPUT_CONVERTERS: Dict[str, Callable] = {}

def _register_output_converter(model_type: str):
  def decorator(func: Callable):
    _OUTPUT_CONVERTERS[model_type] = func
    return func
  return decorator


@_register_output_converter("visual-detector")
def _visual_detector(self, outputs: List[VisualDetectorOutput]):
  """
  Visual detector type output parser.
  """
  out_bboxes = []
  out_labels = []
  out_scores = []

  for pred in outputs:
    assert isinstance(pred, VisualDetectorOutput), 'Expected VisualDetectorOutput'
    out_bboxes.append(pred.predicted_bboxes)
    out_labels.append(pred.predicted_labels)
    out_scores.append(pred.predicted_scores)

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


@_register_output_converter("visual-classifier")
def _visual_classifier(self, outputs: List[ClassifierOutput]):
  """
  Visual classifier type output parser.
  """
  out_scores = []
  for pred in outputs:
    assert isinstance(pred, ClassifierOutput), 'Expected ClassifierOutput'
    out_scores.append(pred.predicted_scores)

  out_tensor_scores = pb_utils.Tensor("softmax_predictions",
                                      np.asarray(out_scores, dtype=np.float32))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_scores])

  return inference_response


@_register_output_converter("text-classifier")
def _text_classifier(self, outputs: List[ClassifierOutput]):
  """
  Text classifier type output parser.
  """
  out_scores = []
  for pred in outputs:
    assert isinstance(pred, ClassifierOutput), 'Expected ClassifierOutput'
    out_scores.append(pred.predicted_scores)

  out_tensor_scores = pb_utils.Tensor("softmax_predictions",
                                      np.asarray(out_scores, dtype=np.float32))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_scores])

  return inference_response


@_register_output_converter("text-to-text")
def _text_to_text(self, outputs: List[TextOutput]):
  """
  Text to text type output parser.
  Convert a sequence of text into another e.g. text generation,
  summarization or translation.
  """
  out_text = []
  for pred in outputs:
    assert isinstance(pred, TextOutput), 'Expected TextOutput'
    out_text.append(pred.predicted_text)

  out_text_tensor = pb_utils.Tensor("text", np.asarray(out_text, dtype=object))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_text_tensor])

  return inference_response


@_register_output_converter("text-embedder")
def _text_embedder(self, outputs: List[EmbeddingOutput]):
  """
  Text embedder type output parser.
  Generates embeddings for an input text.
  """
  out_embeddings = []
  for pred in outputs:
    assert isinstance(pred, EmbeddingOutput), 'Expected EmbeddingOutput'
    out_embeddings.append(pred.embedding_vector)

  out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

  return inference_response


@_register_output_converter("visual-embedder")
def _visual_embedder(self, outputs: List[EmbeddingOutput]):
  """
  Visual embedder type output parser.
  Generates embeddings for an input image.
  """
  out_embeddings = []
  for pred in outputs:
    assert isinstance(pred, EmbeddingOutput), 'Expected EmbeddingOutput'
    out_embeddings.append(pred.embedding_vector)

  out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

  return inference_response


@_register_output_converter("visual-segmenter")
def _visual_segmenter(self, outputs: List[MasksOutput]):
  """
  Visual segmenter type output parser.
  """
  masks = []
  for pred in outputs:
    assert isinstance(pred, MasksOutput), 'Expected MasksOutput'
    masks.append(pred.predicted_mask)

  out_mask_tensor = pb_utils.Tensor("predicted_mask", np.asarray(masks, dtype=np.int64))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_mask_tensor])

  return inference_response


@_register_output_converter("text-to-image")
def _text_to_image(self, outputs: List[ImageOutput]):
  """
  Text to image type output parser.
  """
  for pred in outputs:
    assert isinstance(pred, ImageOutput), 'Expected ImageOutput'
    gen_images.append(pred.image)

  out_image_tensor = pb_utils.Tensor("image", np.asarray(gen_images, dtype=np.uint8))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_image_tensor])

  return inference_response


@_register_output_converter("multimodal-embedder")
def _multimodal_embedder(self, outputs: List[EmbeddingOutput]):
  """
  Multimodal embedder type output parser.
  Generates embeddings for image or text input.
  """
  out_embeddings = []
  for pred in outputs:
    assert isinstance(pred, EmbeddingOutput), 'Expected EmbeddingOutput'
    out_embeddings.append(pred.embedding_vector)

  out_embed_tensor = pb_utils.Tensor("embeddings", np.asarray(out_embeddings, dtype=np.float32))
  inference_response = pb_utils.InferenceResponse(output_tensors=[out_embed_tensor])

  return inference_response
