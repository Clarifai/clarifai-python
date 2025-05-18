import os
import typing
import uuid

import pytest
from google.protobuf import struct_pb2

from clarifai.client.search import Search
from clarifai.client.user import User
from clarifai.errors import UserError

CREATE_APP_USER_ID = os.environ["CLARIFAI_USER_ID"]
uniq = uuid.uuid4().hex[:10]
CREATE_APP_ID = f"ci_search_app_{uniq}"
CREATE_DATASET_ID = "ci_search_dataset"
DOG_IMG_URL = "https://samples.clarifai.com/dog.tiff"
DATASET_IMAGES_DIR = os.path.dirname(__file__) + "/assets/voc/images"

CLARIFAI_API_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")


def get_filters_for_test() -> [(typing.List[typing.Dict], int)]:
    return [
        ([{"geo_point": {"longitude": -29.0, "latitude": 40.0, "geo_limit": 100}}], 1),
        ([{"concepts": [{"name": "dog", "value": 1}]}], 1),
        (
            [
                {  # OR
                    "concepts": [{"name": "deer", "value": 1}, {"name": "dog", "value": 1}]
                }
            ],
            1,
        ),
        (
            [
                {  # AND
                    "concepts": [{"name": "dog", "value": 1}]
                },
                {"concepts": [{"name": "deer", "value": 1}]},
            ],
            0,
        ),
        ([{"metadata": {"Breed": "Saint Bernard"}}], 1),
        # Input Search
        (
            [
                {  # AND
                    "input_types": ["image"],
                },
                {
                    "input_status_code": 30000  # Download Success
                },
            ],
            1,
        ),
        (
            [
                {
                    "input_types": ["text", "audio", "video"],
                }
            ],
            0,
        ),
        (
            [
                {  # OR
                    "input_types": ["text", "audio", "video"],
                    "input_status_code": 30000,  # Download Success
                },
            ],
            1,
        ),
        (
            [
                {"input_dataset_ids": ["random_dataset"]},
            ],
            0,
        ),
    ]


@pytest.mark.requires_secrets
class TestAnnotationSearch:
    @classmethod
    def setup_class(cls):
        cls.client = User(user_id=CREATE_APP_USER_ID, base_url=CLARIFAI_API_BASE)
        cls.search = Search(
            user_id=CREATE_APP_USER_ID,
            app_id=CREATE_APP_ID,
            base_url=CLARIFAI_API_BASE,
            top_k=1,
            metric="euclidean",
        )
        cls.search_with_pagination = Search(
            user_id=CREATE_APP_USER_ID,
            app_id=CREATE_APP_ID,
            base_url=CLARIFAI_API_BASE,
            metric="euclidean",
            pagination=True,
        )
        cls.search_deduplicate = Search(
            user_id=CREATE_APP_USER_ID,
            app_id=CREATE_APP_ID,
            base_url=CLARIFAI_API_BASE,
            top_k=2,
            metric="euclidean",
        )
        cls.upload_data()

    @classmethod
    def upload_data(self):
        app_obj = self.client.create_app(CREATE_APP_ID, base_workflow="General")
        dataset_obj = app_obj.create_dataset(CREATE_DATASET_ID)
        inp_obj = app_obj.inputs()
        metadata = struct_pb2.Struct()
        metadata.update({"Breed": "Saint Bernard"})
        input_proto = inp_obj.get_input_from_url(
            dataset_id=CREATE_DATASET_ID,
            input_id="dog-tiff",
            image_url=DOG_IMG_URL,
            labels=["dog"],
            geo_info=[-30.0, 40.0],  # longitude, latitude
            metadata=metadata,
        )
        inp_obj.upload_inputs([input_proto])
        dataset_obj.upload_from_folder(DATASET_IMAGES_DIR, input_type="image", labels=False)

    @pytest.mark.parametrize("filter_dict_list,expected_hits", get_filters_for_test())
    def test_filter_search(self, filter_dict_list: typing.List[typing.Dict], expected_hits: int):
        query = self.search.query(filters=filter_dict_list)
        assert len(list(query)) == expected_hits

    def test_rank_search(self):
        query = self.search.query(ranks=[{"image_url": "https://samples.clarifai.com/dog.tiff"}])
        for q in query:
            assert len(q.hits) == 1
            assert q.hits[0].input.id == "dog-tiff"

    def test_rank_filter_search(self):
        query = self.search.query(
            ranks=[{"image_url": "https://samples.clarifai.com/dog.tiff"}],
            filters=[{"input_types": ["image"]}],
        )
        for q in query:
            assert len(q.hits) == 1
            assert q.hits[0].input.id == "dog-tiff"

    def test_per_page(self):
        query = self.search_with_pagination.query(
            filters=[{"input_types": ["image"]}], per_page=3, page_no=1
        )
        for q in query:
            assert len(q.hits) == 3

    def test_pagination(self):
        query = self.search_with_pagination.query(filters=[{"input_types": ["image"]}])
        for q in query:
            assert len(q.hits) == 11

    def test_schema_error(self):
        with pytest.raises(UserError):
            _ = self.search.query(
                filters=[
                    {
                        "geo_point": {
                            "longitude": -29.0,
                            "latitude": 40.0,
                            "geo_limit": 10,
                            "extra": 1,
                        }
                    }
                ]
            )

        # Incorrect Concept Keys
        with pytest.raises(UserError):
            _ = self.search.query(
                filters=[
                    {"concepts": [{"value": 1, "concept_id": "deer"}, {"name": "dog", "value": 1}]}
                ]
            )

        # Incorrect Concept Values
        with pytest.raises(UserError):
            _ = self.search.query(
                filters=[{"concepts": [{"name": "deer", "value": 2}, {"name": "dog", "value": 1}]}]
            )

        # Incorrect input type search
        with pytest.raises(UserError):
            _ = self.search.query(filters=[{"input_types": ["imaage"]}])

        # Incorrect input search filter key
        with pytest.raises(UserError):
            _ = self.search.query(filters=[{"input_id": "test"}])

    def test_rank_search_deduplicate(self):
        query = self.search_deduplicate.query(
            ranks=[{"image_url": "https://samples.clarifai.com/dog.tiff"}]
        )
        for q in query:
            assert len(q.hits) == 2
            assert q.hits[0].input.id != q.hits[1].input.id

    def teardown_class(cls):
        cls.client.delete_app(app_id=CREATE_APP_ID)
