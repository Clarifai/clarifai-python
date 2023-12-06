import os

import pytest
from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.errors import UserError

DOG_IMAGE_URL = "https://samples.clarifai.com/dog2.jpeg"
NON_EXISTING_IMAGE_URL = "http://example.com/non-existing.jpg"
RED_TRUCK_IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"
BEER_VIDEO_URL = "https://samples.clarifai.com/beer.mp4"

MAIN_APP_ID = "main"
MAIN_APP_USER_ID = "clarifai"
GENERAL_MODEL_ID = "aaa03c23b3724a16a56b629203edc62c"
CLIP_EMBED_MODEL_ID = "multimodal-clip-embed"

RAW_TEXT = "Hi my name is Jim."
RAW_TEXT_BYTES = b"Hi my name is Jim."

CLARIFAI_PAT = os.environ["CLARIFAI_PAT"]


@pytest.fixture
def model():
  return Model(
      user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, model_id=GENERAL_MODEL_ID, pat=CLARIFAI_PAT)


def validate_concepts_length(response):
  assert len(response.outputs[0].data.concepts) > 0


@pytest.mark.requires_secrets
class TestModelPredict:

  def test_predict_image_url(self, model):
    response = model.predict_by_url(DOG_IMAGE_URL, 'image')
    validate_concepts_length(response)

  def test_predict_filepath(self, model):
    response = model.predict_by_filepath(RED_TRUCK_IMAGE_FILE_PATH, 'image')
    validate_concepts_length(response)

  def test_predict_image_bytes(self, model):
    with open(RED_TRUCK_IMAGE_FILE_PATH, "rb") as f:
      image_bytes = f.read()

    response = model.predict_by_bytes(image_bytes, 'image')
    validate_concepts_length(response)

  def test_predict_image_url_with_selected_concepts(self):
    selected_concepts = [
        resources_pb2.Concept(name="dog"),
        resources_pb2.Concept(name="cat"),
    ]
    model_with_selected_concepts = Model(
        user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, model_id=GENERAL_MODEL_ID)

    response = model_with_selected_concepts.predict_by_url(
        DOG_IMAGE_URL, 'image', output_config=dict(select_concepts=selected_concepts))
    concepts = response.outputs[0].data.concepts

    assert len(concepts) == 2
    dog_concept = next(c for c in concepts if c.name == "dog")
    cat_concept = next(c for c in concepts if c.name == "cat")
    assert dog_concept.value > cat_concept.value

  def test_predict_image_url_with_min_value(self):
    model_with_min_value = Model(
        user_id=MAIN_APP_USER_ID,
        app_id=MAIN_APP_ID,
        model_id=GENERAL_MODEL_ID,
    )

    response = model_with_min_value.predict_by_url(
        DOG_IMAGE_URL, 'image', output_config=dict(min_value=0.98))
    assert len(response.outputs[0].data.concepts) > 0
    for c in response.outputs[0].data.concepts:
      assert c.value >= 0.98

  def test_predict_image_url_with_max_concepts(self):
    model_with_max_concepts = Model(
        user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, model_id=GENERAL_MODEL_ID)

    response = model_with_max_concepts.predict_by_url(
        DOG_IMAGE_URL, 'image', output_config=dict(max_concepts=3))
    assert len(response.outputs[0].data.concepts) == 3

  def test_failed_predicts(self, model):
    # Invalid FilePath
    false_filepath = "false_filepath"
    with pytest.raises(UserError):
      model.predict_by_filepath(false_filepath, 'image')

    # Invalid URL
    with pytest.raises(Exception):
      model.predict_by_url(NON_EXISTING_IMAGE_URL, 'image')

    # Invalid Input Type
    with pytest.raises(UserError):
      model.predict_by_url(DOG_IMAGE_URL, 'invalid_input_type')

  def test_predict_video_url_with_custom_sample_ms(self):
    model_with_custom_sample_ms = Model(
        user_id=MAIN_APP_USER_ID,
        app_id=MAIN_APP_ID,
        model_id=GENERAL_MODEL_ID,
    )
    video_proto = Inputs.get_input_from_url("", video_url=BEER_VIDEO_URL)
    response = model_with_custom_sample_ms.predict(
        [video_proto], output_config=dict(sample_ms=2000))
    # The expected time per frame is the middle between the start and the end of the frame
    # (in milliseconds).
    expected_time = 1000

    assert len(response.outputs[0].data.frames) > 0
    for frame in response.outputs[0].data.frames:
      assert frame.frame_info.time == expected_time
      expected_time += 2000

  def test_text_embed_predict_with_raw_text(self):
    clip_dim = 512
    clip_embed_model = Model(
        user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, model_id=CLIP_EMBED_MODEL_ID)

    input_text_proto = Inputs.get_input_from_bytes(
        "", text_bytes=RAW_TEXT.encode(encoding='UTF-8'))
    response = clip_embed_model.predict([input_text_proto])
    assert response.outputs[0].data.embeddings[0].num_dimensions == clip_dim

    response = clip_embed_model.predict([input_text_proto])
    assert response.outputs[0].data.embeddings[0].num_dimensions == clip_dim

  def test_model_load_info(self):
    clip_embed_model = Model(
        user_id=MAIN_APP_USER_ID, app_id=MAIN_APP_ID, model_id=CLIP_EMBED_MODEL_ID)
    assert len(clip_embed_model.kwargs) == 4
    clip_embed_model.load_info()
    assert len(clip_embed_model.kwargs) > 10
