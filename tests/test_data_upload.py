import logging
import os

from clarifai.client.user import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
CREATE_APP_ID = "ci_input_app"
CREATE_DATASET_ID = "ci_input_test_dataset"

#assets
IMAGE_URL = "https://samples.clarifai.com/metro-north.jpg"
VIDEO_URL = "https://samples.clarifai.com/beer.mp4"
TEXT_URL = "https://samples.clarifai.com/featured-models/Llama2_Conversational-agent.txt"
AUDIO_URL = "https://samples.clarifai.com/english_audio_sample.mp3"
IMAGE_FILE_PATH = os.path.dirname(__file__) + "/assets/red-truck.png"
VIDEO_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.mp4"
TEXT_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.txt"
AUDIO_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.mp3"
CSV_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.csv"
FOLDER_PATH = os.path.dirname(__file__) + "/assets/test"


def create_app():
  client = User(user_id=CREATE_APP_USER_ID)
  return client.create_app(app_id=CREATE_APP_ID, base_workflow="Empty")


class Testdataupload:
  """Tests for data uploads.
  Uploads are tested for each of the following resources:
  - image
  - video
  - text
  - audio

  Tests for the following upload methods:
  - url
  - filepath
  - rawtext
  - CSV
  - Folder
  """

  def setup_class(self):
    self.app = create_app()
    self.input_object = self.app.inputs()
    self.dataset = self.app.create_dataset(dataset_id=CREATE_DATASET_ID)

  def test_upload_image_url(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_url(input_id='input_1', image_url=IMAGE_URL)
      assert "SUCCESS" in caplog.text

  def test_upload_video_url(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_url(input_id='input_2', video_url=VIDEO_URL)
      assert "SUCCESS" in caplog.text

  def test_upload_text_url(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_url(input_id='input_3', text_url=TEXT_URL)
      assert "SUCCESS" in caplog.text

  def test_upload_audio_url(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_url(input_id='input_4', audio_url=AUDIO_URL)
      assert "SUCCESS" in caplog.text

  def test_upload_image_filepath(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_file(input_id='input_5', image_file=IMAGE_FILE_PATH)
      assert "SUCCESS" in caplog.text

  def test_upload_video_filepath(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_file(input_id='input_6', video_file=VIDEO_FILE_PATH)
      assert "SUCCESS" in caplog.text

  def test_upload_audio_filepath(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_file(input_id='input_7', audio_file=AUDIO_FILE_PATH)
      assert "SUCCESS" in caplog.text

  def test_upload_text_filepath(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_from_file(input_id='input_8', text_file=TEXT_FILE_PATH)
      assert "SUCCESS" in caplog.text

  def test_upload_rawtext(self, caplog):
    with caplog.at_level(logging.INFO):
      self.input_object.upload_text(input_id='input_9', raw_text='This is a test text')
      assert "SUCCESS" in caplog.text

  def test_aggregate_inputs(self, caplog):
    uploaded_inputs = self.input_object.list_inputs()
    with caplog.at_level(logging.INFO):
      self.input_object.delete_inputs(uploaded_inputs)
      assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
    assert len(uploaded_inputs) == 9  # 9 inputs uploaded in the above tests

  def test_upload_csv(self, caplog):
    self.dataset.upload_from_csv(
        csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True)
    uploaded_inputs = self.input_object.list_inputs()
    with caplog.at_level(logging.INFO):
      self.input_object.delete_inputs(uploaded_inputs)
      assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
    assert uploaded_inputs[0].data.concepts[0].name == 'neg'  # label of the first input in the CSV file
    assert len(uploaded_inputs) == 5  # 5 inputs are uploaded from the CSV file

  def test_upload_folder(self, caplog):
    self.dataset.upload_from_folder(folder_path=FOLDER_PATH, input_type='image', labels=True)
    uploaded_inputs = self.input_object.list_inputs()
    with caplog.at_level(logging.INFO):
      self.input_object.delete_inputs(uploaded_inputs)
      assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
    assert uploaded_inputs[0].data.concepts[0].name == 'test'  # label of the first input in the folder
    assert len(uploaded_inputs) == 3  # 3 inputs are uploaded from the folder

  def teardown_class(self):
    self.app.delete_dataset(dataset_id=CREATE_DATASET_ID)
    User(user_id=CREATE_APP_USER_ID).delete_app(app_id=CREATE_APP_ID)
