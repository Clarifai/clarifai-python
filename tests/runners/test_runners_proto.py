import importlib
import os
import threading
import time
import uuid

import pytest
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.client import BaseClient, Model, User
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.utils.logging import logger

MY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "dummy_runner_proto_model", "1", "model.py"
)
# logger.disabled = True

TEXT_FILE_PATH = os.path.dirname(os.path.dirname(__file__)) + "/assets/sample.txt"
TEXT_URL = "https://samples.clarifai.com/negative_sentence_12.txt"


def _get_model_instance(model_path, model_name="MyModel"):
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, model_name)
    return cls()


def init_components(
    auth: ClarifaiAuthHelper,
    client: BaseClient,
    app_id,
    model_id,
    nodepool_id,
    compute_cluster_id,
):
    user = User(user_id=auth.user_id, base_url=auth.base, pat=auth.pat, token=auth._token)
    # try:
    app = user.create_app(app_id=app_id)
    # except Exception:
    #   app = App.from_auth_helper(auth=auth, app_id=app_id)
    # try:
    model = app.create_model(model_id=model_id, model_type_id="multimodal-to-text")
    # except Exception as _:
    #   model = Model.from_auth_helper(auth=auth, model_id=model_id)

    new_model = model.create_version(
        pretrained_model_config=resources_pb2.PretrainedModelConfig(
            local_dev=True,
        )
    )

    new_model_version = new_model.model_version.id
    compute_cluster = resources_pb2.ComputeCluster(
        id=compute_cluster_id,
        description="test runners repo",
        cloud_provider=resources_pb2.CloudProvider(id="local", name="Colo 1"),
        region="us-east-1",
        user_id=auth.user_id,
        cluster_type="local-dev",
        managed_by="user",
        key=resources_pb2.Key(id=os.environ["CLARIFAI_PAT"]),
    )
    compute_cluster_request = service_pb2.PostComputeClustersRequest(
        user_app_id=client.user_app_id,
        compute_clusters=[compute_cluster],
    )
    res = client.STUB.PostComputeClusters(compute_cluster_request)
    if res.status.code != status_code_pb2.SUCCESS:
        logger.error(json_format.MessageToDict(res, preserving_proto_field_name=True))
        raise Exception(res.status)

    nodepool = resources_pb2.Nodepool(
        id=nodepool_id,
        description="test runners repo",
        compute_cluster=compute_cluster,
        node_capacity_type=resources_pb2.NodeCapacityType(capacity_types=[1]),
        instance_types=[
            resources_pb2.InstanceType(
                id='instance-1',
                compute_info=resources_pb2.ComputeInfo(
                    cpu_limit="1",
                    cpu_memory="8Gi",
                    num_accelerators=0,
                ),
            )
        ],
        max_instances=1,
    )
    nodepools_request = service_pb2.PostNodepoolsRequest(
        user_app_id=client.user_app_id, compute_cluster_id=compute_cluster_id, nodepools=[nodepool]
    )
    res = client.STUB.PostNodepools(nodepools_request)
    if res.status.code != status_code_pb2.SUCCESS:
        logger.error(json_format.MessageToDict(res, preserving_proto_field_name=True))
        raise Exception(res.status)

    runner = resources_pb2.Runner(
        description="test runners repo",
        worker=resources_pb2.Worker(
            model=resources_pb2.Model(
                id=model_id,
                user_id=auth.user_id,
                app_id=app_id,
                model_version=resources_pb2.ModelVersion(id=new_model_version),
            )
        ),
        num_replicas=1,
        nodepool=nodepool,
    )
    runners_request = service_pb2.PostRunnersRequest(
        user_app_id=client.user_app_id,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        runners=[runner],
    )
    res = client.STUB.PostRunners(runners_request)
    if res.status.code != status_code_pb2.SUCCESS:
        logger.error(json_format.MessageToDict(res, preserving_proto_field_name=True))
        raise Exception(res.status)

    return new_model_version, res.runners[0].id


@pytest.mark.requires_secrets
class TestRunnerServer:
    MODEL_PATH = MY_MODEL_PATH

    @classmethod
    def setup_class(cls):
        NOW = uuid.uuid4().hex[:10]
        cls.MODEL_ID = f"test-runner-model-{NOW}"
        cls.NODEPOOL_ID = f"test-nodepool-{NOW}"
        cls.COMPUTE_CLUSTER_ID = f"test-compute_cluster-{NOW}"
        cls.APP_ID = f"ci-test-runner-app-{NOW}"
        cls.CLIENT = BaseClient.from_env()
        cls.AUTH = cls.CLIENT.auth_helper
        cls.AUTH.app_id = cls.APP_ID
        print("Starting runner server")
        cls.logger = logger
        cls.logger.info("Starting runner server")

        cls.MODEL_VERSION_ID, cls.RUNNER_ID = init_components(
            cls.AUTH,
            cls.CLIENT,
            cls.APP_ID,
            cls.MODEL_ID,
            cls.NODEPOOL_ID,
            cls.COMPUTE_CLUSTER_ID,
        )
        cls.model = Model(
            user_id=cls.AUTH.user_id,
            app_id=cls.AUTH.app_id,
            model_id=cls.MODEL_ID,
            model_version={'id': cls.MODEL_VERSION_ID},
            base_url=cls.AUTH.base,
            pat=cls.AUTH.pat,
            compute_cluster_id=cls.COMPUTE_CLUSTER_ID,
            nodepool_id=cls.NODEPOOL_ID,
            deployment_user_id=cls.AUTH.user_id,
        )

        cls.runner_model = _get_model_instance(cls.MODEL_PATH)

        cls.runner = ModelRunner(
            model=cls.runner_model,
            runner_id=cls.RUNNER_ID,
            nodepool_id=cls.NODEPOOL_ID,
            compute_cluster_id=cls.COMPUTE_CLUSTER_ID,
            num_parallel_polls=1,
            base_url=cls.AUTH.base,
            user_id=cls.AUTH.user_id,
            health_check_port=None,
        )
        cls.thread = threading.Thread(target=cls.runner.start)
        cls.thread.daemon = True  # close when python closes
        cls.thread.start()

        # Wait for runner to initialize before running tests
        # This is especially important on CI environments like Ubuntu where initialization
        # can be slower and tests may start before the runner is ready
        time.sleep(2)

    @classmethod
    def teardown_class(cls):
        auth = cls.AUTH
        client = cls.CLIENT
        compute_cluster_id = cls.COMPUTE_CLUSTER_ID
        nodepool_id = cls.NODEPOOL_ID
        runner_id = cls.RUNNER_ID
        print("Stopping runner server")
        cls.logger.info("Stopping runner server")
        cls.thread.join(timeout=0.01)

        cls.runner.stop()

        user = User(user_id=auth.user_id, base_url=auth.base, pat=auth.pat, token=auth._token)
        user.delete_app(cls.APP_ID)
        runner_delete_request = service_pb2.DeleteRunnersRequest(
            user_app_id=client.user_app_id,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            ids=[runner_id],
        )
        client.STUB.DeleteRunners(runner_delete_request)

        nodepool_delete_request = service_pb2.DeleteNodepoolsRequest(
            user_app_id=client.user_app_id,
            compute_cluster_id=compute_cluster_id,
            ids=[nodepool_id],
        )
        client.STUB.DeleteNodepools(nodepool_delete_request)

        compute_cluster_delete_request = service_pb2.DeleteComputeClustersRequest(
            user_app_id=client.user_app_id,
            ids=[compute_cluster_id],
        )
        client.STUB.DeleteComputeClusters(compute_cluster_delete_request)

    def _validate_response(self, res, expected):
        if res.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to predict: {res}")
        if len(res.outputs) == 0:
            raise Exception(f"Failed to get outputs: {res}")
        out = res.outputs[0].data.text.raw
        out = out.replace("\r\n", "\n")
        assert expected == out

    def _format_request(self, text):
        runner_selector = resources_pb2.RunnerSelector(
            nodepool=resources_pb2.Nodepool(
                id=self.NODEPOOL_ID,
                compute_cluster=resources_pb2.ComputeCluster(
                    id=self.COMPUTE_CLUSTER_ID, user_id=self.AUTH.user_id
                ),
            )
        )

        return service_pb2.PostModelOutputsRequest(
            model_id=self.MODEL_ID,
            version_id=self.MODEL_VERSION_ID,
            user_app_id=self.AUTH.get_user_app_id_proto(),
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(raw=text),
                    )
                ),
            ],
            runner_selector=runner_selector,
        )

    def _format_client_request(self, text):
        runner_selector = resources_pb2.RunnerSelector(
            nodepool=resources_pb2.Nodepool(
                id=self.NODEPOOL_ID,
                compute_cluster=resources_pb2.ComputeCluster(
                    id=self.COMPUTE_CLUSTER_ID, user_id=self.AUTH.user_id
                ),
            )
        )
        inputs = [
            resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text))),
        ]
        return inputs, runner_selector

    def test_unary(self):
        # self.logger.info("Testing unary")
        text = "Test"
        expected = f"{text}Hello World"
        req = self._format_request(text)
        stub = self.CLIENT.STUB
        res = stub.PostModelOutputs(req)
        self._validate_response(res, expected)

    def test_client_predict(self):
        text = "Test"
        expected = f"{text}Hello World"

        # Test predict
        inputs, runner_selector = self._format_client_request(text)
        res = self.model.predict(
            inputs=inputs,
        )
        self._validate_response(res, expected)

    def test_client_predict_by_bytes(self):
        text = "Test"
        expected = f"{text}Hello World"
        res = self.model.predict_by_bytes(
            text.encode("utf-8"),
            "text",
        )
        self._validate_response(res, expected)

    def test_client_predict_by_url(self):
        res = self.model.predict_by_url(
            TEXT_URL,
            "text",
        )
        expected = "He doesn't have to commute to work.Hello World"
        self._validate_response(res, expected)

    def test_client_predict_by_filepath(self):
        res = self.model.predict_by_filepath(
            TEXT_FILE_PATH,
            "text",
        )

        with open(TEXT_FILE_PATH, "r") as f:
            expected = f"{f.read()}Hello World"

        self._validate_response(res, expected)
