from typing import Dict

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import logger
from clarifai.utils.protobuf import dict_to_protobuf


class Deployment(Lister, BaseClient):
    """Deployment is a class that provides access to Clarifai API endpoints related to Deployment information."""

    def __init__(
        self,
        deployment_id: str = None,
        user_id: str = None,
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes a Deployment object.

        Args:
            deployment_id (str): The Deployment ID for the Deployment to interact with.
            user_id (str): The user ID of the user.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the deployment.
        """
        self.kwargs = {**kwargs, 'id': deployment_id, 'user_id': user_id}

        # Filter kwargs to only include fields that exist in the Deployment proto
        proto_fields = {
            f.name for f in resources_pb2.Deployment.DESCRIPTOR.fields if f.name in self.kwargs
        }
        proto_kwargs = {k: self.kwargs[k] for k in proto_fields}

        self.deployment_info = resources_pb2.Deployment()
        dict_to_protobuf(self.deployment_info, proto_kwargs)
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

    def refresh(self):
        """Refresh the deployment info from the API."""
        request = service_pb2.GetDeploymentRequest(
            user_app_id=self.user_app_id, deployment_id=self.id
        )
        response = self._grpc_request(self.STUB.GetDeployment, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to get deployment: {response.status.details}")

        self.deployment_info.CopyFrom(response.deployment)
        return self

    @staticmethod
    def get_runner_selector(user_id: str, deployment_id: str) -> resources_pb2.RunnerSelector:
        """Returns a RunnerSelector object for the given deployment_id.

        Args:
            deployment_id (str): The deployment ID for the deployment.

        Returns:
            resources_pb2.RunnerSelector: A RunnerSelector object for the given deployment_id.
        """
        return resources_pb2.RunnerSelector(
            deployment=resources_pb2.Deployment(id=deployment_id, user_id=user_id)
        )

    def __getattr__(self, name):
        return getattr(self.deployment_info, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.deployment_info, param)}"
            for param in init_params
            if hasattr(self.deployment_info, param)
        ]
        return f"Deployment Details: \n{', '.join(attribute_strings)}\n"

    def logs(
        self, stream: bool = False, log_type: str = "runner", page: int = 1, per_page: int = 100
    ):
        """Get logs for the deployment.

        Args:
            stream (bool): Whether to stream the logs or list them.
            log_type (str): The type of logs to retrieve. Defaults to "runner".
                Valid types are "runner" and "runner.events".
            page (int): The page number to list (only for list).
            per_page (int): The number of items per page (only for list).

        Yields:
            LogEntry: Log entry objects.

        Example:
            >>> from clarifai.client.deployment import Deployment
            >>> deployment = Deployment(deployment_id="deployment_id", user_id="user_id")
            >>> for entry in deployment.logs(stream=True):
            ...     print(entry.message)
        """
        if log_type not in ["runner", "runner.events"]:
            raise ValueError(
                f"Invalid log_type '{log_type}'. Valid types for deployment are 'runner' and 'runner.events'."
            )

        if not self.deployment_info.HasField("worker"):
            self.refresh()

        request_kwargs = {
            "user_app_id": self.user_app_id,
            "log_type": log_type,
        }

        # Add additional fields from deployment_info if they exist
        if self.deployment_info.nodepools:
            nodepool = self.deployment_info.nodepools[0]
            request_kwargs["nodepool_id"] = nodepool.id
            if nodepool.compute_cluster.id:
                request_kwargs["compute_cluster_id"] = nodepool.compute_cluster.id
            if nodepool.compute_cluster.user_id:
                request_kwargs["compute_cluster_user_id"] = nodepool.compute_cluster.user_id

        if self.deployment_info.HasField("worker"):
            worker = self.deployment_info.worker
            if worker.HasField("model"):
                request_kwargs["model_id"] = worker.model.id
                if worker.model.model_version.id:
                    request_kwargs["model_version_id"] = worker.model.model_version.id
            elif worker.HasField("workflow"):
                request_kwargs["workflow_id"] = worker.workflow.id
                if worker.workflow.workflow_version.id:
                    request_kwargs["workflow_version_id"] = worker.workflow.workflow_version.id

        if stream:
            request = service_pb2.StreamLogEntriesRequest(**request_kwargs)
            for response in self.STUB.StreamLogEntries(request):
                if response.status.code != status_code_pb2.SUCCESS:
                    raise Exception(f"Failed to stream logs: {response}")
                for entry in response.log_entries:
                    yield entry
        else:
            request_kwargs["page"] = page
            request_kwargs["per_page"] = per_page
            request = service_pb2.ListLogEntriesRequest(**request_kwargs)
            response = self.STUB.ListLogEntries(request)
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Failed to list logs: {response}")
            for entry in response.log_entries:
                yield entry

    def patch(self, action: str = "overwrite", **kwargs):
        """Patch the deployment.

        Args:
            action (str): The action to perform (overwrite, merge, remove). Defaults to "overwrite".
            **kwargs: The fields to patch on the deployment.
        """
        deployment = resources_pb2.Deployment(id=self.id)
        dict_to_protobuf(deployment, kwargs)

        request = service_pb2.PatchDeploymentsRequest(
            user_app_id=self.user_app_id, action=action, deployments=[deployment]
        )
        response = self._grpc_request(self.STUB.PatchDeployments, request)
        if response.status.code != status_code_pb2.SUCCESS:
            self.logger.error(f"PatchDeployments failed. Status: {response.status}")
            raise Exception(f"Failed to patch deployment: {response}")

        # Update local deployment_info if success
        dict_to_protobuf(self.deployment_info, kwargs)
        return response

    def runner_metrics(self) -> Dict[str, int]:
        """Get the accumulated runner metrics for the deployment.

        This aggregates runner metrics across all nodepools to find the total pods
        running across all of them.

        Returns:
            Dict[str, int]: A dictionary with 'pods_total' and 'pods_running'.

        Example:
            >>> from clarifai.client.deployment import Deployment
            >>> deployment = Deployment(deployment_id="deployment_id", user_id="user_id")
            >>> print(deployment.runner_metrics())
        """
        if not self.deployment_info.worker.HasField(
            "model"
        ) and not self.deployment_info.worker.HasField("workflow"):
            self.refresh()

        from clarifai.client.user import User

        user = User(user_id=self.user_app_id.user_id, pat=self.pat, base_url=self.base)

        model_version_ids = None
        workflow_version_ids = None
        if self.worker.HasField("model"):
            model_version_ids = [self.worker.model.model_version.id]
        elif self.worker.HasField("workflow"):
            workflow_version_ids = [self.worker.workflow.workflow_version.id]

        pods_total = 0
        pods_running = 0

        for np_proto in self.deployment_info.nodepools:
            filter_by = {
                "nodepool_id": np_proto.id,
                "compute_cluster_id": np_proto.compute_cluster.id,
            }
            if model_version_ids:
                filter_by["model_version_ids"] = model_version_ids
            if workflow_version_ids:
                filter_by["workflow_version_ids"] = workflow_version_ids

            runners = user.list_runners(filter_by=filter_by)

            for runner in runners:
                metrics = runner.get("runner_metrics")
                if metrics:
                    pods_total += metrics.get("pods_total", 0)
                    pods_running += metrics.get("pods_running", 0)

        return {"pods_total": pods_total, "pods_running": pods_running}

    def update(self, min_replicas: int = None, max_replicas: int = None):
        """Update deployment replicas.

        Args:
            min_replicas (int): The minimum number of replicas.
            max_replicas (int): The maximum number of replicas.

        Example:
            >>> from clarifai.client.deployment import Deployment
            >>> deployment = Deployment(deployment_id="deployment_id", user_id="user_id")
            >>> deployment.update(min_replicas=1, max_replicas=2)
        """
        patch_kwargs = {}
        if min_replicas is not None or max_replicas is not None:
            autoscale_config = {}
            if min_replicas is not None:
                autoscale_config["min_replicas"] = min_replicas
            if max_replicas is not None:
                autoscale_config["max_replicas"] = max_replicas
            patch_kwargs["autoscale_config"] = autoscale_config

        if not patch_kwargs:
            return

        return self.patch(action="overwrite", **patch_kwargs)
