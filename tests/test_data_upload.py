import logging
import os
import time
import uuid

import pytest
from google.protobuf.struct_pb2 import Struct

from clarifai.client.user import User
from clarifai.datasets.upload.loaders.coco_detection import COCODetectionDataLoader
from clarifai.datasets.upload.loaders.imagenet_classification import ImageNetDataLoader
from clarifai.datasets.upload.utils import load_module_dataloader

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
NOW = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"ci_input_app_{NOW}"
CREATE_DATASET_ID = "ci_input_test_dataset"
CREATE_DATASET_NEW_ID = "ci_input_test_dataset_new"

# assets
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
MODULE_DIR = os.path.dirname(__file__) + "/assets/voc"
COCO_DET_DIR = os.path.dirname(__file__) + "/assets/coco_detection"
IMAGENET_DIR = os.path.dirname(__file__) + "/assets/imagenet_classification"
TEXTS_FOLDER_PATH = os.path.dirname(__file__) + "/assets/sample_texts"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def client():
    return User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)


@pytest.mark.requires_secrets
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

    @classmethod
    def setup_class(self):
        self.app = client().create_app(app_id=CREATE_APP_ID, base_workflow="Empty")
        self.input_object = self.app.inputs()
        self.dataset = self.app.create_dataset(dataset_id=CREATE_DATASET_ID)
        self.dataset_new = self.app.create_dataset(dataset_id=CREATE_DATASET_NEW_ID)

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

    def test_upload_image_bytes(self, caplog):
        with open(IMAGE_FILE_PATH, 'rb') as image_file:
            image_bytes = image_file.read()
        with caplog.at_level(logging.INFO):
            self.input_object.upload_from_bytes(input_id='input_10', image_bytes=image_bytes)
            assert "SUCCESS" in caplog.text

    def test_upload_video_bytes(self, caplog):
        with open(VIDEO_FILE_PATH, 'rb') as video_file:
            video_bytes = video_file.read()
        with caplog.at_level(logging.INFO):
            self.input_object.upload_from_bytes(input_id='input_11', video_bytes=video_bytes)
            assert "SUCCESS" in caplog.text

    def test_upload_audio_bytes(self, caplog):
        with open(AUDIO_FILE_PATH, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        with caplog.at_level(logging.INFO):
            self.input_object.upload_from_bytes(input_id='input_12', audio_bytes=audio_bytes)
            assert "SUCCESS" in caplog.text

    def test_upload_text_bytes(self, caplog):
        with open(TEXT_FILE_PATH, 'rb') as text_file:
            text_bytes = text_file.read()
        with caplog.at_level(logging.INFO):
            self.input_object.upload_from_bytes(input_id='input_13', text_bytes=text_bytes)
            assert "SUCCESS" in caplog.text

    def test_get_multimodal_input(self):
        input_object = self.input_object.get_multimodal_input(
            input_id='input_14', raw_text='This is a multimodal test text', image_url=IMAGE_URL
        )
        assert (
            input_object.id == 'input_14'
            and input_object.data.text.raw == 'This is a multimodal test text'
        )

    def test_get_text_inputs_from_folder(self):
        text_inputs = self.input_object.get_text_inputs_from_folder(TEXTS_FOLDER_PATH)
        assert len(text_inputs) == 3

    def test_get_input_from_app(self):
        input_object = self.input_object.get_input(input_id='input_1')
        assert input_object.id == 'input_1'

    def test_get_mask_proto(self):
        polygon_points = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        annotation = self.input_object.get_mask_proto(
            input_id="input_1",
            label="input_1_polygon_label",
            polygons=polygon_points,
            label_id="input_1_polygon_label",
            annot_id="input_1_polygon_annot",
        )
        assert annotation.id == "input_1_polygon_annot" and annotation.input_id == "input_1"

    def test_list_inputs(self):
        paginated_inputs = list(self.input_object.list_inputs(page_no=1, per_page=5))
        image_filterd_inputs = list(self.input_object.list_inputs(input_type='image'))
        downloaded_inputs = self.input_object.download_inputs(image_filterd_inputs)
        assert len(downloaded_inputs) == 3  # download inputs check
        assert len(paginated_inputs) == 5
        assert len(image_filterd_inputs) == 3  # 2 images uploaded in the above tests

    def test_patch_inputs(self):
        metadata = Struct()
        metadata.update({'test': 'SUCCESS'})
        new_input = self.input_object._get_proto(input_id='input_1', metadata=metadata)
        self.input_object.patch_inputs([new_input], action='merge')
        for input_item in list(self.input_object.list_inputs()):
            if input_item.id == 'input_1':
                assert input_item.data.metadata["test"] == "SUCCESS"
                break

    def test_patch_annotations(self, caplog):
        bbox_points = [0.2, 0.2, 0.8, 0.8]
        annotation = self.input_object.get_bbox_proto(
            input_id="input_1",
            label="input_1_label",
            bbox=bbox_points,
            label_id="input_1_label",
            annot_id="input_1_annot",
        )
        with caplog.at_level(logging.INFO):
            self.input_object.upload_annotations([annotation])
            assert "SUCCESS" in caplog.text  # upload annotations check

        bbox_points = [0.4, 0.4, 0.6, 0.6]
        annotation = self.input_object.get_bbox_proto(
            input_id="input_1",
            label="input_1_label",
            bbox=bbox_points,
            label_id="input_1_label",
            annot_id="input_1_annot",
        )
        self.input_object.patch_annotations([annotation], action='merge')
        test_annotation = list(self.input_object.list_annotations())[0]
        assert test_annotation.id == "input_1_annot"
        annot_bbox = test_annotation.data.regions[0].region_info.bounding_box
        assert (
            round(annot_bbox.left_col, 1) == 0.4
            and round(annot_bbox.top_row, 1) == 0.4
            and round(annot_bbox.right_col, 1) == 0.6
            and round(annot_bbox.bottom_row, 1) == 0.6
        )

    def test_patch_concepts(self):
        self.input_object.patch_concepts(
            concept_ids=["input_1_label"], labels=["SUCCESS"], values=[], action='overwrite'
        )
        concepts = list(self.app.list_concepts())
        for concept in concepts:
            if concept.id == "input_1_label":
                assert concepts[0].name == "SUCCESS"
                break

    def test_aggregate_inputs(self, caplog):
        uploaded_inputs = list(self.input_object.list_inputs())
        with caplog.at_level(logging.INFO):
            self.input_object.delete_inputs(uploaded_inputs)
            assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
        assert len(uploaded_inputs) == 13  # 9 inputs uploaded in the above tests

    def test_upload_csv(self, caplog):
        self.dataset.upload_from_csv(
            csv_path=CSV_FILE_PATH, input_type='text', csv_type='raw', labels=True
        )
        uploaded_inputs = list(self.input_object.list_inputs(dataset_id=CREATE_DATASET_ID))
        concepts = list(self.app.list_concepts())
        with caplog.at_level(logging.INFO):
            self.input_object.delete_inputs(uploaded_inputs)
            assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
        assert len(uploaded_inputs) == 5  # 5 inputs are uploaded from the CSV file
        assert len(concepts) == 3  # Test for list concepts

    def test_upload_folder(self, caplog):
        self.dataset.upload_from_folder(folder_path=FOLDER_PATH, input_type='image', labels=True)
        uploaded_inputs = list(self.input_object.list_inputs())
        with caplog.at_level(logging.INFO):
            self.input_object.delete_inputs(uploaded_inputs)
            assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
        assert (
            uploaded_inputs[0].data.concepts[0].name == 'test'
        )  # label of the first input in the folder
        assert len(uploaded_inputs) == 3  # 3 inputs are uploaded from the folder

    def test_upload_dataset(self, caplog):
        dataloader = load_module_dataloader(module_dir=MODULE_DIR, split="train")
        self.dataset.upload_dataset(dataloader, get_upload_status=True, log_warnings=True)
        uploaded_inputs = list(self.input_object.list_inputs())
        input_ids = [input_item.id for input_item in uploaded_inputs]
        annotations = list(self.input_object.list_annotations(batch_input=uploaded_inputs))
        with caplog.at_level(logging.INFO):
            self.input_object.delete_annotations(input_ids=input_ids)
            assert "Annotations Deleted" in caplog.text  # Testing delete annotations action
            self.input_object.delete_inputs(uploaded_inputs)
            assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
        assert len(uploaded_inputs) == 10  # 3 inputs are uploaded from the folder
        assert len(annotations) == 28  # Test for list annotations

    def test_upload_coco_dataset(self, caplog):
        dataloader = COCODetectionDataLoader(
            images_dir=os.path.join(COCO_DET_DIR, "images"),
            label_filepath=os.path.join(COCO_DET_DIR, "instances_default.json"),
        )
        self.dataset.upload_dataset(dataloader)
        uploaded_inputs = list(self.input_object.list_inputs())
        annotations = list(self.input_object.list_annotations(batch_input=uploaded_inputs))
        with caplog.at_level(logging.INFO):
            self.input_object.delete_inputs(uploaded_inputs)
            assert "Inputs Deleted" in caplog.text  # Testing delete inputs action
        assert len(uploaded_inputs) == 3  # 3 inputs are uploaded from the folder
        assert len(annotations) == 7  # Test for list annotations

    def test_upload_imagenet_dataset(self):
        dataloader = ImageNetDataLoader(data_dir=IMAGENET_DIR)
        self.dataset.upload_dataset(dataloader)
        uploaded_inputs = list(self.input_object.list_inputs())
        annotations = list(self.input_object.list_annotations(batch_input=uploaded_inputs))
        assert len(uploaded_inputs) == 10  # 10 inputs are uploaded from the folder
        assert len(annotations) == 10  # Test for list annotations

    def test_merge_datasets(self, caplog):
        with caplog.at_level(logging.INFO):
            self.dataset_new.merge_dataset(merge_dataset_id=CREATE_DATASET_ID)
            assert "SUCCESS" in caplog.text

    def test_export_dataset(self):
        dataset_demo_version = self.dataset.create_version()
        time.sleep(5)
        dataset_demo_version.export(save_path='tests/output_demo.zip')
        assert os.path.exists('tests/output_demo.zip') is True
        os.remove('tests/output_demo.zip')

    @classmethod
    def teardown_class(self):
        self.app.delete_dataset(dataset_id=CREATE_DATASET_ID)
        client().delete_app(app_id=CREATE_APP_ID)
