import logging
import os
import time
import uuid

import pytest
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.user import User
from clarifai.utils.evaluation import EvalResultCompare

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
NOW = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"ci_input_app_{NOW}"
CREATE_DATASET_ID = "ci_input_test_dataset"
CREATE_DATASET_NEW_ID = "ci_input_test_dataset_new"
CREATE_MODEL_ID = "ci_input_test_model_1"
CSV_FILE_PATH = os.path.dirname(__file__) + "/assets/sample.csv"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def client():
    return User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)


@pytest.mark.requires_secrets
class TestEval:
    """Tests for model evaluating."""

    @classmethod
    def setup_class(self):
        self.app = client().create_app(app_id=CREATE_APP_ID, base_workflow="Universal")
        self.input_object = self.app.inputs()
        self.dataset = self.app.create_dataset(dataset_id=CREATE_DATASET_ID)
        self.dataset_new = self.app.create_dataset(dataset_id=CREATE_DATASET_NEW_ID)
        self.model = self.app.create_model(
            model_id=CREATE_MODEL_ID, model_type_id='embedding-classifier'
        )

    def test_evaluate(self, caplog):
        # Prepare dataset
        self.dataset.upload_from_csv(
            csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True
        )
        dataset_version = self.dataset.create_version()
        self.dataset_new.upload_from_csv(
            csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True
        )
        self.dataset_new.create_version()
        concepts = [concept.id for concept in self.app.list_concepts()]
        # Prepare for training
        self.model.get_params(save_to='tests/assets/model_params_for_eval.yaml')
        assert len(concepts) == 2  # test data upload for training
        self.model.update_params(
            dataset_id=CREATE_DATASET_ID,
            concepts=concepts,
            dataset_version_id=dataset_version.version.id,
        )
        with caplog.at_level(logging.INFO):
            model_version_id = self.model.train()
            assert "Model Training Started" in caplog.text  # test model training

        while True:
            status = self.model.training_status(version_id=model_version_id, training_logs=True)
            if status.code == 21106:  # MODEL_TRAINING_FAILED
                break
            elif status.code == 21100:  # MODEL_TRAINED
                break
            else:
                time.sleep(2)

        # Test eval
        ## Test list eval
        all_evals = self.model.list_evaluations()
        assert len(all_evals) == 0

        ## Test evaluate
        self.model.evaluate(dataset=self.dataset, eval_id="one", eval_info={"use_kfold": False})
        all_evals = self.model.list_evaluations()
        assert len(all_evals) == 1

        ## Test get_eval_by_id
        time.time()
        while True:
            response1 = self.model.get_eval_by_id("one")
            if response1.status.code == status_code_pb2.MODEL_EVALUATED:
                break
            else:
                time.sleep(2)

        ## Test get_eval_by_dataset
        dataset_eval = self.model.get_eval_by_dataset(dataset=self.dataset)
        assert (
            dataset_eval[0].id == "one"
            and dataset_eval[0].status.code == status_code_pb2.MODEL_EVALUATED
        )

        ## Test get_raw_eval
        with pytest.raises(Exception) as e:
            self.model.get_raw_eval(dataset=self.dataset)
        assert (
            "method only supports model types ['visual-classifier', 'text-classifier', 'visual-detector']"
            in str(e.value).lower()
        )

        ## Test get_latest_eval
        current_eval = self.model.get_latest_eval(
            label_counts=True,
            test_set=True,
            binary_metrics=True,
            confusion_matrix=True,
            metrics_by_area=True,
            metrics_by_class=True,
        )
        assert (
            current_eval.id == "one"
            and current_eval.status.code == status_code_pb2.MODEL_EVALUATED
        )

        ## Test EvalResultCompare
        eval_result = EvalResultCompare(
            models=[self.model],
            datasets=[self.dataset, self.dataset_new],
            attempt_evaluate=True,
            eval_info={"use_kfold": False},
        )
        eval_result.all('tests/eval/', overwrite=True)
        assert os.path.exists('tests/eval/conf-0.5') is True

        # cleanup
        with caplog.at_level(logging.INFO):
            self.model.delete_version(version_id=model_version_id)
            assert "Model Version Deleted" in caplog.text  # test model version deletion
        os.remove('tests/assets/model_params_for_eval.yaml')

    @classmethod
    def teardown_class(self):
        self.app.delete_model(model_id=CREATE_MODEL_ID)
        self.app.delete_dataset(dataset_id=CREATE_DATASET_ID)
        client().delete_app(app_id=CREATE_APP_ID)
