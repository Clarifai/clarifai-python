import logging
import os
import uuid

import pytest

from clarifai.client.user import User

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
NOW = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"ci_input_app_{NOW}"
CREATE_DATASET_ID = "ci_input_test_dataset"
CREATE_MODEL_ID = "ci_input_test_model_1"
CSV_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.csv"


def create_app():
  client = User(user_id=CREATE_APP_USER_ID)
  return client.create_app(app_id=CREATE_APP_ID, base_workflow="Empty")


@pytest.mark.requires_secrets
class TestEval:
  """Tests for model evaluating.
  """

  @classmethod
  def setup_class(self):
    self.app = create_app()
    self.input_object = self.app.inputs()
    self.dataset = self.app.create_dataset(dataset_id=CREATE_DATASET_ID)
    self.model = self.app.create_model(model_id=CREATE_MODEL_ID, model_type_id='text-classifier')

  def test_evaluate(self, caplog):
    # Prepare dataset
    self.dataset.upload_from_csv(
        csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True)
    dataset_version = self.dataset.create_version()
    concepts = [concept.id for concept in self.app.list_concepts()]
    # Prepare for training
    self.model.get_params(
        template='HF_GPTNeo_125m_lora', save_to='tests/assets/model_params_for_eval.yaml')
    param_info = self.model.get_param_info(param='tokenizer_config')
    assert param_info['param'] == 'tokenizer_config'  #test get param info
    assert len(concepts) == 2  #test data upload for training
    self.model.update_params(
        dataset_id=CREATE_DATASET_ID,
        concepts=concepts,
        dataset_version_id=dataset_version.version.id)
    with caplog.at_level(logging.INFO):
      model_version_id = self.model.train()
      assert "Model Training Started" in caplog.text  #test model training

    # Test eval
    ## Test list eval
    with pytest.raises(Exception) as e:
      self.model.list_evaluations()
    assert "model version is empty" in str(e.value).lower()

    self.model.model_info.model_version.id = model_version_id
    with pytest.raises(Exception) as e:
      self.model.list_evaluations()
    assert "model not yet trained" in str(e.value).lower()
    ## Test evaluate
    with pytest.raises(Exception) as e:
      self.model.evaluate(dataset_id=CREATE_DATASET_ID)
    assert "model not yet trained" in str(e.value).lower()

    ## Test get eval
    with pytest.raises(Exception) as e:
      self.model.get_latest_eval(
          label_counts=True,
          test_set=True,
          binary_metrics=True,
          confusion_matrix=True,
          metrics_by_area=True,
          metrics_by_class=True)
    assert "model not yet trained" in str(e.value).lower()

    # cleanup
    with caplog.at_level(logging.INFO):
      self.model.delete_version(version_id=model_version_id)
      assert "Model Version Deleted" in caplog.text  #test model version deletion
    os.remove('tests/assets/model_params_for_eval.yaml')

  @classmethod
  def teardown_class(self):
    self.app.delete_dataset(dataset_id=CREATE_DATASET_ID)
    self.app.delete_model(model_id=CREATE_MODEL_ID)
    User(user_id=CREATE_APP_USER_ID).delete_app(app_id=CREATE_APP_ID)
