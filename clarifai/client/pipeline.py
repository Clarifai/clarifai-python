import time
import uuid
from typing import Dict, List, Optional

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import logger


class Pipeline(Lister, BaseClient):
    """Pipeline is a class that provides access to Clarifai API endpoints related to Pipeline information."""

    def __init__(
        self,
        url: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        pipeline_version_id: Optional[str] = None,
        pipeline_version_run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        nodepool_id: Optional[str] = None,
        compute_cluster_id: Optional[str] = None,
        log_file: Optional[str] = None,
        base_url: str = DEFAULT_BASE,
        pat: Optional[str] = None,
        token: Optional[str] = None,
        root_certificates_path: Optional[str] = None,
        **kwargs,
    ):
        """Initializes a Pipeline object.

        Args:
            url (Optional[str]): The URL to initialize the pipeline object.
            pipeline_id (Optional[str]): The Pipeline ID to interact with.
            pipeline_version_id (Optional[str]): The Pipeline Version ID to interact with.
            pipeline_version_run_id (Optional[str]): The Pipeline Version Run ID. If not provided, a UUID will be generated.
            user_id (Optional[str]): The User ID that owns the pipeline.
            app_id (Optional[str]): The App ID that contains the pipeline.
            nodepool_id (Optional[str]): The Nodepool ID to run the pipeline on.
            compute_cluster_id (Optional[str]): The Compute Cluster ID to run the pipeline on.
            log_file (Optional[str]): Path to file where logs should be written. If not provided, logs are displayed on console.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (Optional[str]): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the Pipeline.
        """
        if url and pipeline_id:
            raise UserError("You can only specify one of url or pipeline_id.")
        if not url and not pipeline_id:
            raise UserError("You must specify one of url or pipeline_id.")
        if url:
            parsed_user_id, parsed_app_id, _, parsed_pipeline_id, parsed_version_id = (
                ClarifaiUrlHelper.split_clarifai_url(url)
            )
            user_id = user_id or parsed_user_id
            app_id = app_id or parsed_app_id
            pipeline_id = parsed_pipeline_id
            pipeline_version_id = pipeline_version_id or parsed_version_id

        self.pipeline_id = pipeline_id
        self.pipeline_version_id = pipeline_version_id
        self.pipeline_version_run_id = pipeline_version_run_id or str(uuid.uuid4().hex)
        self.user_id = user_id
        self.app_id = app_id
        self.nodepool_id = nodepool_id
        self.compute_cluster_id = compute_cluster_id
        self.log_file = log_file

        # Store all kwargs as attributes for API data
        for key, value in kwargs.items():
            setattr(self, key, value)

        BaseClient.__init__(
            self,
            user_id=user_id,
            app_id=app_id,
            base=base_url,
            pat=pat,
            token=token,
            root_certificates_path=root_certificates_path,
        )
        Lister.__init__(self)

        # Set up runner selector if compute cluster and nodepool are provided
        self._runner_selector = None
        if self.compute_cluster_id and self.nodepool_id:
            from clarifai.client.nodepool import Nodepool

            self._runner_selector = Nodepool.get_runner_selector(
                user_id=self.user_id,
                compute_cluster_id=self.compute_cluster_id,
                nodepool_id=self.nodepool_id,
            )

    def run(self, inputs: List = None, timeout: int = 3600, monitor_interval: int = 10) -> Dict:
        """Run the pipeline and monitor its progress.

        Args:
            inputs (List): List of inputs to run the pipeline with. If None, runs without inputs.
            timeout (int): Maximum time to wait for completion in seconds. Default 3600 (1 hour).
            monitor_interval (int): Interval between status checks in seconds. Default 10.

        Returns:
            Dict: The pipeline run result.
        """
        # Create a new pipeline version run
        pipeline_version_run = resources_pb2.PipelineVersionRun()
        pipeline_version_run.id = self.pipeline_version_run_id

        # Set nodepools if nodepool information is available
        if self.nodepool_id and self.compute_cluster_id:
            nodepool = resources_pb2.Nodepool(
                id=self.nodepool_id,
                compute_cluster=resources_pb2.ComputeCluster(
                    id=self.compute_cluster_id, user_id=self.user_id
                ),
            )
            pipeline_version_run.nodepools.extend([nodepool])

        run_request = service_pb2.PostPipelineVersionRunsRequest()
        run_request.user_app_id.CopyFrom(self.user_app_id)
        run_request.pipeline_id = self.pipeline_id
        run_request.pipeline_version_id = self.pipeline_version_id or ""
        run_request.pipeline_version_runs.append(pipeline_version_run)

        # Add runner selector if available
        if self._runner_selector:
            run_request.runner_selector.CopyFrom(self._runner_selector)

        logger.info(f"Starting pipeline run for pipeline {self.pipeline_id}")
        response = self.STUB.PostPipelineVersionRuns(
            run_request, metadata=self.auth_helper.metadata
        )

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            if response.status.code == status_code_pb2.StatusCode.CONN_DOES_NOT_EXIST:
                logger.error(
                    f"Pipeline {self.pipeline_id} does not exist, did you call 'clarifai pipeline upload' first? "
                )
                return json_format.MessageToDict(response, preserving_proto_field_name=True)
            else:
                raise UserError(
                    f"Failed to start pipeline run: {response.status.description}. Details: {response.status.details}. Code: {status_code_pb2.StatusCode.Name(response.status.code)}."
                )

        if not response.pipeline_version_runs:
            raise UserError("No pipeline version run was created")

        pipeline_version_run = response.pipeline_version_runs[0]
        run_id = pipeline_version_run.id or self.pipeline_version_run_id

        logger.info(f"Pipeline version run created with ID: {run_id}")

        # Monitor the run
        return self._monitor_pipeline_run(run_id, timeout, monitor_interval)

    def monitor_only(self, timeout: int = 3600, monitor_interval: int = 10) -> Dict:
        """Monitor an existing pipeline run without starting a new one.

        Args:
            timeout (int): Maximum time to wait for completion in seconds. Default 3600 (1 hour).
            monitor_interval (int): Interval between status checks in seconds. Default 10.

        Returns:
            Dict: The pipeline run result.
        """
        if not self.pipeline_version_run_id:
            raise UserError("pipeline_version_run_id is required for monitoring existing runs")

        logger.info(f"Monitoring existing pipeline run with ID: {self.pipeline_version_run_id}")

        # Monitor the existing run
        return self._monitor_pipeline_run(self.pipeline_version_run_id, timeout, monitor_interval)

    def _monitor_pipeline_run(self, run_id: str, timeout: int, monitor_interval: int) -> Dict:
        """Monitor a pipeline version run until completion.

        Args:
            run_id (str): The pipeline version run ID to monitor.
            timeout (int): Maximum time to wait for completion in seconds.
            monitor_interval (int): Interval between status checks in seconds.

        Returns:
            Dict: The pipeline run result.
        """
        start_time = time.time()
        seen_logs = set()
        current_page = 1  # Track current page for log pagination.

        while time.time() - start_time < timeout:
            # Get run status
            get_run_request = service_pb2.GetPipelineVersionRunRequest()
            get_run_request.user_app_id.CopyFrom(self.user_app_id)
            get_run_request.pipeline_id = self.pipeline_id
            get_run_request.pipeline_version_id = self.pipeline_version_id or ""
            get_run_request.pipeline_version_run_id = run_id

            try:
                run_response = self.STUB.GetPipelineVersionRun(
                    get_run_request, metadata=self.auth_helper.metadata
                )

                if run_response.status.code != status_code_pb2.StatusCode.SUCCESS:
                    logger.error(f"Error getting run status: {run_response.status.description}")
                    time.sleep(monitor_interval)
                    continue

                pipeline_run = run_response.pipeline_version_run
                pipeline_run_dict = json_format.MessageToDict(
                    pipeline_run, preserving_proto_field_name=True
                )

                # Display new log entries and update current page
                current_page = self._display_new_logs(run_id, seen_logs, current_page)

                elapsed_time = time.time() - start_time
                logger.info(f"Pipeline run monitoring... (elapsed {elapsed_time:.1f}s)")

                # Check if we have orchestration status
                if (
                    hasattr(pipeline_run, 'orchestration_status')
                    and pipeline_run.orchestration_status
                ):
                    orch_status = pipeline_run.orchestration_status
                    if hasattr(orch_status, 'status') and orch_status.status:
                        status_code = orch_status.status.code
                        status_name = status_code_pb2.StatusCode.Name(status_code)
                        logger.info(f"Pipeline run status: {status_code} ({status_name})")

                        # Display orchestration status details if available
                        if hasattr(orch_status, 'description') and orch_status.description:
                            logger.info(f"Orchestration status: {orch_status.description}")

                        # Success codes that allow continuation: JOB_RUNNING, JOB_QUEUED
                        if status_code in [
                            status_code_pb2.JOB_QUEUED,
                            status_code_pb2.JOB_RUNNING,
                        ]:  # JOB_QUEUED, JOB_RUNNING
                            logger.info(f"Pipeline run in progress: {status_code} ({status_name})")
                            # Continue monitoring
                        # Successful terminal state: JOB_COMPLETED
                        elif status_code == status_code_pb2.JOB_COMPLETED:  # JOB_COMPLETED
                            logger.info("Pipeline run completed successfully!")
                            return {"status": "success", "pipeline_version_run": pipeline_run_dict}
                        # Failure terminal states: JOB_UNEXPECTED_ERROR, JOB_FAILED
                        elif status_code in [
                            status_code_pb2.JOB_FAILED,
                            status_code_pb2.JOB_UNEXPECTED_ERROR,
                        ]:  # JOB_FAILED, JOB_UNEXPECTED_ERROR
                            logger.error(
                                f"Pipeline run failed with status: {status_code} ({status_name})"
                            )
                            return {"status": "failed", "pipeline_version_run": pipeline_run_dict}
                        # Handle legacy SUCCESS status for backward compatibility
                        elif status_code == status_code_pb2.StatusCode.SUCCESS:
                            logger.info("Pipeline run completed successfully!")
                            return {"status": "success", "pipeline_version_run": pipeline_run_dict}
                        elif status_code != status_code_pb2.StatusCode.MIXED_STATUS:
                            # Log other unexpected statuses but continue monitoring
                            logger.warning(
                                f"Unexpected pipeline run status: {status_code} ({status_name}). Continuing to monitor..."
                            )

            except Exception as e:
                logger.error(f"Error monitoring pipeline run: {e}")

            time.sleep(monitor_interval)

        logger.error(f"Pipeline run timed out after {timeout} seconds")
        return {"status": "timeout"}

    def _display_new_logs(self, run_id: str, seen_logs: set, current_page: int = 1) -> int:
        """Display new log entries for a pipeline version run.

        Args:
            run_id (str): The pipeline version run ID.
            seen_logs (set): Set of already seen log entry IDs.
            current_page (int): The current page to fetch logs from.

        Returns:
            int: The next page number to fetch from in subsequent calls.
        """
        try:
            logs_request = service_pb2.ListLogEntriesRequest()
            logs_request.user_app_id.CopyFrom(self.user_app_id)
            logs_request.pipeline_id = self.pipeline_id
            logs_request.pipeline_version_id = self.pipeline_version_id or ""
            logs_request.pipeline_version_run_id = run_id
            logs_request.log_type = "pipeline.version.run"  # Set required log type
            logs_request.page = current_page
            logs_request.per_page = 50

            logs_response = self.STUB.ListLogEntries(
                logs_request, metadata=self.auth_helper.metadata
            )

            if logs_response.status.code == status_code_pb2.StatusCode.SUCCESS:
                entries_count = 0
                for log_entry in logs_response.log_entries:
                    entries_count += 1
                    # Use log entry URL or timestamp as unique identifier
                    log_id = log_entry.url or f"{log_entry.created_at.seconds}_{log_entry.message}"
                    if log_id not in seen_logs:
                        seen_logs.add(log_id)
                        log_message = f"[LOG] {log_entry.message.strip()}"

                        # Write to file if log_file is specified, otherwise log to console
                        if self.log_file:
                            with open(self.log_file, 'a', encoding='utf-8') as f:
                                f.write(log_message + '\n')
                        else:
                            logger.info(log_message)

                # If we got a full page (50 entries), there might be more logs on the next page
                # If we got fewer than 50 entries, we've reached the end and should stay on current page
                if entries_count == 50:
                    return current_page + 1
                else:
                    return current_page

        except Exception as e:
            logger.debug(f"Error fetching logs: {e}")
            # Return current page on error to retry the same page next fetch
            return current_page

    def get_pipeline_version(self, pipeline_version_id: Optional[str] = None) -> Dict:
        """Get pipeline version details including step secrets.

        Args:
            pipeline_version_id (Optional[str]): The pipeline version ID. If None, uses self.pipeline_version_id.

        Returns:
            Dict: Pipeline version information including step_version_secrets if configured.
        """
        version_id = pipeline_version_id or self.pipeline_version_id
        if not version_id:
            raise UserError("pipeline_version_id is required")

        request = service_pb2.GetPipelineVersionRequest()
        request.user_app_id.CopyFrom(self.user_app_id)
        request.pipeline_id = self.pipeline_id
        request.pipeline_version_id = version_id

        response = self.STUB.GetPipelineVersion(request, metadata=self.auth_helper.metadata)

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            raise UserError(
                f"Failed to get pipeline version: {response.status.description}. "
                f"Details: {response.status.details}"
            )

        return json_format.MessageToDict(
            response.pipeline_version, preserving_proto_field_name=True
        )

    def create_pipeline_version(
        self,
        orchestration_spec: Dict,
        step_version_secrets: Optional[Dict[str, Dict[str, str]]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Create a new pipeline version with optional step secrets.

        Note: This creates a new version by patching the pipeline with a new version.

        Args:
            orchestration_spec (Dict): The orchestration specification for the pipeline.
            step_version_secrets (Optional[Dict[str, Dict[str, str]]]): Map of step references to their secrets.
                Format: {step_ref: {secret_name: secret_path}}
                Example: {"step-0": {"API_KEY": "users/user123/secrets/my-api-key"}}
            description (Optional[str]): Description for the pipeline version.

        Returns:
            str: The created pipeline version ID.
        """
        pipeline_version = resources_pb2.PipelineVersion()
        if description:
            pipeline_version.description = description

        # Set orchestration spec
        if "argo_orchestration_spec" in orchestration_spec:
            argo_spec_str = orchestration_spec["argo_orchestration_spec"]
            import yaml

            argo_spec = yaml.safe_load(argo_spec_str)
            api_version = argo_spec.get("apiVersion", "argoproj.io/v1alpha1")

            orchestration_spec_proto = resources_pb2.OrchestrationSpec()
            argo_orchestration_spec_proto = resources_pb2.ArgoOrchestrationSpec()
            argo_orchestration_spec_proto.api_version = api_version
            import json

            argo_orchestration_spec_proto.spec_json = json.dumps(argo_spec)

            orchestration_spec_proto.argo_orchestration_spec.CopyFrom(
                argo_orchestration_spec_proto
            )
            pipeline_version.orchestration_spec.CopyFrom(orchestration_spec_proto)

        # Add step_version_secrets if provided (updated for new proto format)
        if step_version_secrets:
            from google.protobuf.struct_pb2 import Struct

            for step_ref, secrets in step_version_secrets.items():
                if not secrets:
                    continue
                step_secrets_struct = Struct()
                step_secrets_struct.update(secrets)
                pipeline_version.config.step_version_secrets[step_ref].CopyFrom(
                    step_secrets_struct
                )

        # Make the API call using PatchPipelineVersions
        # This creates a new version for an existing pipeline
        request = service_pb2.PatchPipelineVersionsRequest()
        request.user_app_id.CopyFrom(self.user_app_id)
        request.pipeline_id = self.pipeline_id
        request.pipeline_versions.append(pipeline_version)
        request.action = "overwrite"  # Create a new version

        response = self.STUB.PatchPipelineVersions(request, metadata=self.auth_helper.metadata)

        if response.status.code != status_code_pb2.StatusCode.SUCCESS:
            raise UserError(
                f"Failed to create pipeline version: {response.status.description}. "
                f"Details: {response.status.details}"
            )

        if not response.pipeline_versions:
            raise UserError("No pipeline version was created")

        created_version = response.pipeline_versions[0]
        logger.info(f"Created pipeline version: {created_version.id}")
        return created_version.id

    def add_step_secret(
        self,
        step_ref: str,
        secret_name: str,
        secret_ref: str,
        pipeline_version_id: Optional[str] = None,
    ) -> None:
        """Add a secret to a specific pipeline step.

        Args:
            step_ref (str): The step reference (e.g., "step-0", "step-1").
            secret_name (str): The name of the secret environment variable.
            secret_ref (str): The secret reference path (e.g., "users/user123/secrets/my-api-key").
            pipeline_version_id (Optional[str]): The pipeline version ID. If None, uses self.pipeline_version_id.

        Note:
            This is a convenience method. For production use, manage secrets via the config.yaml
            orchestration spec and use the pipeline upload command.
        """
        raise NotImplementedError(
            "Adding secrets to existing pipeline versions is not supported. "
            "Please define step secrets in your config.yaml orchestration spec "
            "and use 'clarifai pipeline upload' to create a new pipeline version."
        )

    def list_step_secrets(
        self, step_ref: Optional[str] = None, pipeline_version_id: Optional[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """List secrets configured for pipeline steps.

        Args:
            step_ref (Optional[str]): If provided, only return secrets for this step.
            pipeline_version_id (Optional[str]): The pipeline version ID. If None, uses self.pipeline_version_id.

        Returns:
            Dict[str, Dict[str, str]]: Map of step references to their secrets.
                Format: {step_ref: {secret_name: secret_path}}
        """
        version_data = self.get_pipeline_version(pipeline_version_id)
        config = version_data.get("config", {})
        step_version_secrets = config.get("step_version_secrets", {})

        if step_ref:
            # Return only the specified step's secrets
            # With new proto format, secrets are directly in the step config (no nested 'secrets' field)
            return {step_ref: step_version_secrets.get(step_ref, {})}

        # Return all step secrets
        # With new proto format using Struct, secrets are directly accessible
        result = {}
        for step, step_config in step_version_secrets.items():
            result[step] = step_config if isinstance(step_config, dict) else {}
        return result
