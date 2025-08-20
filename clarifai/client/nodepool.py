import os
from typing import Any, Dict, Generator, List

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.deployment import Deployment
from clarifai.client.lister import Lister
from clarifai.client.runner import Runner
from clarifai.errors import UserError
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import logger


class Nodepool(Lister, BaseClient):
    """Nodepool is a class that provides access to Clarifai API endpoints related to Nodepool information."""

    def __init__(
        self,
        nodepool_id: str = None,
        user_id: str = None,
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes a Nodepool object.

        Args:
            nodepool_id (str): The Nodepool ID for the Nodepool to interact with.
            user_id (str): The user ID of the user.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the nodepool.
        """
        self.kwargs = {**kwargs, 'id': nodepool_id}
        self.nodepool_info = resources_pb2.Nodepool(**self.kwargs)
        self.logger = logger
        BaseClient.__init__(
            self,
            user_id=user_id,
            base=base_url,
            pat=pat,
            token=token,
            root_certificates_path=root_certificates_path,
        )
        Lister.__init__(self)

    def list_deployments(
        self, filter_by: Dict[str, Any] = {}, page_no: int = None, per_page: int = None
    ) -> Generator[Deployment, None, None]:
        """Lists all the available deployments of compute cluster.

        Args:
            filter_by (Dict[str, Any]): The filter to apply to the list of deployments.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Deployment: Deployment objects for the nodepools in the compute cluster.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> all_deployments = list(nodepool.list_deployments())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, nodepool_id=self.id, **filter_by)
        all_deployments_info = self.list_pages_generator(
            self.STUB.ListDeployments,
            service_pb2.ListDeploymentsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for deployment_info in all_deployments_info:
            yield Deployment.from_auth_helper(auth=self.auth_helper, **deployment_info)

    def _process_deployment_config(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        assert "deployment" in deployment_config, "deployment info not found in the config file"
        deployment = deployment_config['deployment']
        assert ("worker" in deployment) and (
            ("model" in deployment["worker"]) or ("workflow" in deployment["worker"])
        ), "worker info not found in the config file"
        assert "scheduling_choice" in deployment, "scheduling_choice not found in the config file"
        assert "nodepools" in deployment, "nodepools not found in the config file"
        deployment['user_id'] = (
            deployment['user_id'] if 'user_id' in deployment else self.user_app_id.user_id
        )
        if "autoscale_config" in deployment:
            deployment['autoscale_config'] = resources_pb2.AutoscaleConfig(
                **deployment['autoscale_config']
            )
        deployment['nodepools'] = [
            resources_pb2.Nodepool(
                id=nodepool['id'],
                compute_cluster=resources_pb2.ComputeCluster(
                    id=nodepool['compute_cluster']['id'],
                    user_id=nodepool['compute_cluster']['user_id']
                    if 'user_id' in nodepool['compute_cluster']
                    else self.user_app_id.user_id,
                ),
            )
            for nodepool in deployment['nodepools']
        ]
        if 'user' in deployment['worker']:
            deployment['worker']['user'] = resources_pb2.User(**deployment['worker']['user'])
        elif 'model' in deployment['worker']:
            deployment['worker']['model'] = resources_pb2.Model(**deployment['worker']['model'])
        elif 'workflow' in deployment['worker']:
            deployment['worker']['workflow'] = resources_pb2.Workflow(
                **deployment['worker']['workflow']
            )
        deployment['worker'] = resources_pb2.Worker(**deployment['worker'])
        if "visibility" in deployment:
            deployment["visibility"] = resources_pb2.Visibility(**deployment["visibility"])
        return deployment

    @staticmethod
    def get_runner_selector(
        user_id: str, compute_cluster_id: str, nodepool_id: str
    ) -> resources_pb2.RunnerSelector:
        """Returns a RunnerSelector object for the specified compute cluster and nodepool.

        Args:
            user_id (str): The user ID of the user.
            compute_cluster_id (str): The compute cluster ID for the compute cluster.
            nodepool_id (str): The nodepool ID for the nodepool.

        Returns:
            resources_pb2.RunnerSelector: A RunnerSelector object for the specified compute cluster and nodepool.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> runner_selector = Nodepool.get_runner_selector(user_id="user_id", compute_cluster_id="compute_cluster_id", nodepool_id="nodepool_id")
        """
        compute_cluster = resources_pb2.ComputeCluster(id=compute_cluster_id, user_id=user_id)
        nodepool = resources_pb2.Nodepool(id=nodepool_id, compute_cluster=compute_cluster)
        return resources_pb2.RunnerSelector(nodepool=nodepool)

    def create_deployment(
        self,
        config_filepath: str = None,
        deployment_id: str = None,
        deployment_config: Dict[str, Any] = None,
    ) -> Deployment:
        """Creates a deployment for the nodepool.

        Args:
            config_filepath (str): The path to the deployment config file.
            deployment_id (str): New deployment ID for the deployment to create.

        Returns:
            Deployment: A Deployment object for the specified deployment ID.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> deployment = nodepool.create_deployment(config_filepath="config.yml")
        """

        if config_filepath is not None:
            assert deployment_config is None, (
                "deployment_config has to be None if config_filepath is provided"
            )

            if not os.path.exists(config_filepath):
                raise UserError(f"Deployment config file not found at {config_filepath}")
            with open(config_filepath, "r") as file:
                deployment_config = yaml.safe_load(file)
        elif deployment_config is not None:
            assert isinstance(deployment_config, dict), (
                "deployment_config should be a dictionary if provided."
            )
        else:
            raise AssertionError("Either config_filepath or deployment_config must be provided.")

        deployment_config = self._process_deployment_config(deployment_config)

        if 'id' in deployment_config:
            if deployment_id is None:
                deployment_id = deployment_config['id']
            deployment_config.pop('id')

        request = service_pb2.PostDeploymentsRequest(
            user_app_id=self.user_app_id,
            deployments=[resources_pb2.Deployment(id=deployment_id, **deployment_config)],
        )
        response = self._grpc_request(self.STUB.PostDeployments, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info(
            f"Deployment with ID '{response.deployments[0].id}' is created:\n{response.status}"
        )

        dict_response = MessageToDict(
            response.deployments[0], preserving_proto_field_name=True, use_integers_for_enums=True
        )
        kwargs = self.process_response_keys(dict_response, "deployment")
        return Deployment.from_auth_helper(auth=self.auth_helper, **kwargs)

    def deployment(self, deployment_id: str) -> Deployment:
        """Returns a Deployment object for the existing deployment ID.

        Args:
            deployment_id (str): The deployment ID for the deployment to interact with.

        Returns:
            Deployment: A Deployment object for the existing deployment ID.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> deployment = nodepool.deployment(deployment_id="deployment_id")
        """
        request = service_pb2.GetDeploymentRequest(
            user_app_id=self.user_app_id, deployment_id=deployment_id
        )
        response = self._grpc_request(self.STUB.GetDeployment, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(
            response, preserving_proto_field_name=True, use_integers_for_enums=True
        )
        kwargs = self.process_response_keys(dict_response["deployment"], "deployment")
        return Deployment.from_auth_helper(auth=self.auth_helper, **kwargs)

    def delete_deployments(self, deployment_ids: List[str]) -> None:
        """Deletes list of deployments for the nodepool.

        Args:
            deployment_ids (List[str]): The list of deployment IDs to delete.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> nodepool.delete_deployments(deployment_ids=["deployment_id1", "deployment_id2"])
        """
        assert isinstance(deployment_ids, list), "deployment_ids param should be a list"

        request = service_pb2.DeleteDeploymentsRequest(
            user_app_id=self.user_app_id, ids=deployment_ids
        )
        response = self._grpc_request(self.STUB.DeleteDeployments, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nDeployments Deleted\n%s", response.status)

    def runner(self, runner_id: str) -> Runner:
        """Returns a Runner object for the existing runner ID.

        Args:
            runner_id (str): The runner ID for the runner to interact with.

        Returns:
            Runner: A Runner object for the existing runner ID.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> runner = nodepool.runner(runner_id="runner_id")
        """
        request = service_pb2.GetRunnerRequest(user_app_id=self.user_app_id, runner_id=runner_id)
        response = self._grpc_request(self.STUB.GetRunner, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(
            response, preserving_proto_field_name=True, use_integers_for_enums=True
        )
        kwargs = self.process_response_keys(dict_response["runner"], "runner")
        return Runner.from_auth_helper(auth=self.auth_helper, **kwargs)

    def create_runner(
        self, config_filepath: str = None, runner_config: Dict[str, Any] = None
    ) -> Runner:
        """Creates a runner for the nodepool. Only needed for local runners.

        Args:
            config_filepath (str): The path to the runner config file.
            nodepool_config (Dict[str, Any]) = nodepool_config or {}

        Returns:
            resources_pb2.Runner: A Runner object for the specified deployment ID.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> runner = nodepool.create_runner(deployment_id="deployment_id")
        """

        if config_filepath is not None:
            assert runner_config is None, (
                "runner_config has to be None if config_filepath is provided"
            )

            if not os.path.exists(config_filepath):
                raise UserError(f"Runner config file not found at {config_filepath}")
            with open(config_filepath, "r") as file:
                runner_config = yaml.safe_load(file)
        elif runner_config is not None:
            assert isinstance(runner_config, dict), (
                "runner_config should be a dictionary if provided."
            )
        else:
            raise AssertionError("Either config_filepath or runner_config must be provided.")

        runner_config = self._process_runner_config(runner_config)

        request = service_pb2.PostRunnersRequest(
            user_app_id=self.user_app_id,
            nodepool_id=self.id,
            compute_cluster_id=self.compute_cluster.id,
            runners=[resources_pb2.Runner(**runner_config)],
        )
        response = self._grpc_request(self.STUB.PostRunners, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info(
            f"Runner with ID '{response.runners[0].id}' is created:\n{response.status}"
        )

        dict_response = MessageToDict(response.runners[0], preserving_proto_field_name=True)
        kwargs = self.process_response_keys(dict_response, 'runner')
        return Runner.from_auth_helper(auth=self.auth_helper, **kwargs)

    def delete_runners(self, runner_ids: List[str]) -> None:
        """Deletes list of runners for the nodepool.

        Args:
            runner_ids (List[str]): The list of runner IDs to delete.

        Example:
            >>> from clarifai.client.nodepool import Nodepool
            >>> nodepool = Nodepool(nodepool_id="nodepool_id", user_id="user_id")
            >>> nodepool.delete_runners(runner_ids=["runner_id1", "runner_id2"])
        """
        assert isinstance(runner_ids, list), "runner_ids param should be a list"

        request = service_pb2.DeleteRunnersRequest(
            user_app_id=self.user_app_id,
            ids=runner_ids,
            compute_cluster_id=self.compute_cluster.id,
            nodepool_id=self.id,
        )
        response = self._grpc_request(self.STUB.DeleteRunners, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nRunners Deleted\n%s", response.status)

    def _process_runner_config(self, runner_config: str) -> Dict[str, Any]:
        assert "runner" in runner_config, "runner info not found in the config file"
        runner = runner_config['runner']
        assert "worker" in runner, "worker not found in the config file"
        assert "num_replicas" in runner, "num_replicas not found in the config file"
        return runner

    def __getattr__(self, name):
        return getattr(self.nodepool_info, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.nodepool_info, param)}"
            for param in init_params
            if hasattr(self.nodepool_info, param)
        ]
        return f"Nodepool Details: \n{', '.join(attribute_strings)}\n"
