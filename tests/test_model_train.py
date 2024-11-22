import logging
import os
import time
import uuid

import pytest
import yaml

from clarifai.client.user import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
NOW = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"ci_input_app_{NOW}"
CREATE_DATASET_ID = "ci_input_test_dataset"
CREATE_MODEL_ID = "ci_input_test_model"
CREATE_MODEL_ID_1 = "ci_input_test_model_1"
CSV_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.csv"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def client():
  return User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)


@pytest.mark.requires_secrets
class Testmodeltrain:
  """Tests for model training.
  """

  @classmethod
  def setup_class(self):
    self.app = client().create_app(app_id=CREATE_APP_ID, base_workflow="Empty")
    self.input_object = self.app.inputs()
    self.dataset = self.app.create_dataset(dataset_id=CREATE_DATASET_ID)
    self.visual_classifier_model = self.app.create_model(
        model_id=CREATE_MODEL_ID, model_type_id='visual-classifier')
    self.text_classifier_model = self.app.create_model(
        model_id=CREATE_MODEL_ID_1, model_type_id='text-classifier')

  def test_model_templates(self):
    model_types = self.app.list_trainable_model_types()
    templates = self.visual_classifier_model.list_training_templates()
    assert self.visual_classifier_model.model_type_id == 'visual-classifier'  #create model test
    assert len(model_types) == 8  #list trainable model types test
    assert len(templates) >= 11  #list training templates test

  def test_model_params(self):
    model_params = self.visual_classifier_model.get_params(
        template='MMClassification_AdvancedConfig', save_to='tests/assets/model_params.yaml')
    with open('tests/assets/model_params.yaml', 'r') as file:
      expected_data = yaml.safe_load(file)
    with open(expected_data['train_params']['custom_config'], 'r') as python_file:
      custom_config = python_file.read()

    assert expected_data['train_params'].keys() == model_params[
        'train_params'].keys()  #test yaml model params
    assert len(model_params['concepts']) == 0  #test model params concepts
    assert 'custom_config' in expected_data['train_params'].keys()  #test custom config in yaml
    assert expected_data['train_params']['template'] == 'MMClassification_AdvancedConfig'  #test template param in yaml
    assert 'image_size' in expected_data['train_params'].keys()  #test image size param in yaml
    assert 'dataset_id' in expected_data.keys()  #test dataset id param in yaml
    assert '_base_' in custom_config  #test custom config script
    assert 'max_epochs' in custom_config  #test custom config script
    # cleanup
    os.remove('tests/assets/model_params.yaml')
    os.remove(expected_data['train_params']['custom_config'])

  def test_model_train(self, caplog):
    self.dataset.upload_from_csv(
        csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True)
    dataset_version = self.dataset.create_version()
    concepts = [concept.id for concept in self.app.list_concepts()]
    self.text_classifier_model.get_params(
        template='HF_GPTNeo_125m_lora', save_to='tests/assets/model_params.yaml')
    param_info = self.text_classifier_model.get_param_info(param='tokenizer_config')
    assert param_info['param'] == 'tokenizer_config'  #test get param info
    assert len(concepts) == 2  #test data upload for training
    self.text_classifier_model.update_params(
        dataset_id=CREATE_DATASET_ID,
        concepts=concepts,
        dataset_version_id=dataset_version.version.id)
    with caplog.at_level(logging.INFO):
      model_version_id = self.text_classifier_model.train()
      assert "Model Training Started" in caplog.text  #test model training

    training_status = self.text_classifier_model.training_status(version_id=model_version_id)
    assert model_version_id  #test model version id from model.train()
    assert training_status.code  #test training_status
    # NOTE (EAGLE-4139) - Immediately deleting the app causes the training to fail because the first
    # step of training requires an app, this triggers infra alerts for non graceful exit. We need to delay the deletion
    # until we get past this step.

    time.sleep(5)

    with caplog.at_level(logging.INFO):
      self.text_classifier_model.delete_version(version_id=model_version_id)
      assert "Model Version Deleted" in caplog.text  #test model version deletion
    # cleanup
    os.remove('tests/assets/model_params.yaml')

  @classmethod
  def teardown_class(self):
    self.app.delete_dataset(dataset_id=CREATE_DATASET_ID)
    self.app.delete_model(model_id=CREATE_MODEL_ID)
    self.app.delete_model(model_id=CREATE_MODEL_ID_1)
    client().delete_app(app_id=CREATE_APP_ID)
