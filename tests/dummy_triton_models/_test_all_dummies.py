import time

import numpy as np
import pytest as pytest
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

MAX_BATCH_SIZE = 4
MAX_TRIES = 5
INTERVAL = 10
count = 0
while count < MAX_TRIES:
  try:
    _ = InferenceServerClient('0.0.0.0:8001').is_server_live()
    break
  except Exception as e:
    print(e)
    count += 1
    time.sleep(INTERVAL)


@pytest.fixture
def triton_client():
  return InferenceServerClient('0.0.0.0:8001')


def make_input(name, inputs):
  model_input = InferInput(name, inputs.shape, np_to_triton_dtype(inputs.dtype))
  model_input.set_data_from_numpy(inputs)
  return model_input


def make_random_image_input(name="image", bs=1, size=256):
  image = np.random.rand(bs, size, size, 3) * 255
  image = image.astype("uint8")
  return make_input(name, image)


def make_text_input(name="text", text="this is text", bs=1):
  text = np.array([text] * bs, dtype=np.object_).reshape(-1, 1)
  return make_input(name, text)


def inference(triton_client, model_name, input_: list, output_names: list):
  res = triton_client.infer(
      model_name=model_name,
      inputs=input_,
      outputs=[InferRequestedOutput(each) for each in output_names])
  return {output_name: res.as_numpy(output_name) for output_name in output_names}


def execute_test_image_as_input(triton_client, model_name, input_name, output_names):
  single_input = make_random_image_input(name=input_name, bs=1, size=256)
  res = inference(triton_client, model_name, [single_input], output_names=output_names)
  outputs = [res[each] for each in output_names]

  if len(outputs) > 1:
    assert all(len(each[0]) == 1
               for each in outputs), f"[{model_name}], All predictions must have same length"
  elif model_name == "visual-classifier":
    assert outputs[0].all() <= 1.
  else:
    assert len(outputs[0].shape)

  # Test bs > 1
  multi_input = make_random_image_input(name=input_name, bs=2, size=256)
  res = inference(triton_client, model_name, [multi_input], output_names=output_names)
  outputs = [res[each] for each in output_names]

  if len(outputs) > 1:
    assert all(len(each[0]) == 1
               for each in outputs), f"[{model_name}], All predictions must have same length"
  elif model_name == "visual-classifier":
    assert outputs[0].all() <= 1.
  else:
    assert len(outputs[0].shape)

  # Test bs > max_batch_size
  with pytest.raises(Exception):
    multi_input = make_random_image_input(name=input_name, bs=10, size=256)
    res = inference(triton_client, model_name, [multi_input], output_names=output_names)


def execute_test_text_as_input(triton_client, model_name, input_name, output_names):
  single_input = make_text_input(name=input_name, bs=1)
  res = inference(triton_client, model_name, [single_input], output_names=output_names)
  outputs = [res[each] for each in output_names]

  if model_name == "text-to-image":
    assert len(outputs[0][0].shape) == 3
  elif model_name == "text-classifier":
    assert outputs[0].all() <= 1.
  else:
    assert len(outputs[0].shape)

  # Test bs > 1
  multi_input = make_text_input(name=input_name, bs=2)
  res = inference(triton_client, model_name, [multi_input], output_names=output_names)
  outputs = [res[each] for each in output_names]

  if model_name == "text-to-image":
    assert len(outputs[0][0].shape) == 3
  elif model_name == "text-classifier":
    assert outputs[0].all() <= 1.
  else:
    assert len(outputs[0].shape)

  # Test bs > max_batch_size
  with pytest.raises(Exception):
    multi_input = make_text_input(name=input_name, bs=10)
    res = inference(triton_client, model_name, [multi_input], output_names=output_names)


class TestModelTypes:

  # --------- Image Input --------- #
  def test_visual_detector(self, triton_client):
    model_name = "visual-detector"
    input_name = "image"
    output_names = ["predicted_bboxes", "predicted_labels", "predicted_scores"]
    execute_test_image_as_input(triton_client, model_name, input_name, output_names)

  def test_visual_classifier(self, triton_client):
    model_name = "visual-classifier"
    input_name = "image"
    output_names = ["softmax_predictions"]
    execute_test_image_as_input(triton_client, model_name, input_name, output_names)

  def test_visual_embedder(self, triton_client):
    model_name = "visual-embedder"
    input_name = "image"
    output_names = ["embeddings"]
    execute_test_image_as_input(triton_client, model_name, input_name, output_names)

  def test_visual_segmenter(self, triton_client):
    model_name = "visual-segmenter"
    input_name = "image"
    output_names = ["predicted_mask"]
    execute_test_image_as_input(triton_client, model_name, input_name, output_names)

  # --------- Text Input --------- #
  def test_text_to_image(self, triton_client):
    model_name = "text-to-image"
    input_name = "text"
    output_names = ["image"]
    execute_test_text_as_input(triton_client, model_name, input_name, output_names)

  def test_text_classifier(self, triton_client):
    model_name = "text-classifier"
    input_name = "text"
    output_names = ["softmax_predictions"]
    execute_test_text_as_input(triton_client, model_name, input_name, output_names)

  def test_text_embedder(self, triton_client):
    model_name = "text-embedder"
    input_name = "text"
    output_names = ["embeddings"]
    execute_test_text_as_input(triton_client, model_name, input_name, output_names)

  def test_text_to_text(self, triton_client):
    model_name = "text-to-text"
    input_name = "text"
    output_names = ["text"]
    execute_test_text_as_input(triton_client, model_name, input_name, output_names)

  # --------- Multimodal Inputs --------- #
  def test_multimodal_embedder(self, triton_client):
    model_name = "multimodal-embedder"
    output_names = ["embeddings"]
    execute_test_image_as_input(triton_client, model_name, "image", output_names)
    execute_test_text_as_input(triton_client, model_name, "text", output_names)
