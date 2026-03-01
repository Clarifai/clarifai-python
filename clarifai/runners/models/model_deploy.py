"""Non-interactive model deployment orchestrator.

Orchestrates upload + build + deploy in a single command with zero prompts.
All decisions come from parameters or config. Errors are raised as UserError
with actionable fix instructions. This class NEVER calls input(), click.confirm(),
click.prompt(), or any interactive function.
"""

import contextlib
import json
import logging
import os
import re
import time
import uuid

from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.compute_presets import (
    get_compute_cluster_config,
    get_deploy_compute_cluster_id,
    get_deploy_nodepool_id,
    get_nodepool_config,
    resolve_gpu,
)
from clarifai.utils.logging import logger

# Default deployment monitoring settings
DEFAULT_MONITOR_TIMEOUT = 600  # 10 minutes
DEFAULT_POLL_INTERVAL = 5  # seconds
DEFAULT_LOG_TAIL_DURATION = 15  # seconds to check for runner logs after pods are ready

# K8s events to skip in default (non-verbose) mode — transient scheduler noise
_SKIP_EVENTS = {"TaintManagerEviction", "SandboxChanged", "FailedCreatePodSandBox"}

# Map k8s Reason to human-friendly status for non-verbose mode
_EVENT_PHASE_MAP = {
    "FailedScheduling": "Scheduling",
    "NotTriggerScaleUp": "Scaling",
    "NominatedNode": "Nominated",
    "Nominated": "Nominated",
    "Scheduled": "Scheduled",
    "Pulling": "Pulling image",
    "Pulled": "Image pulled",
    "Created": "Starting",
    "Started": "Running",
    "BackOff": "Restarting",
    "Unhealthy": "Health check",
    "Killing": "Stopping",
    "Preempted": "Preempted",
    "FailedMount": "Volume",
    "FailedAttachVolume": "Volume",
    "SuccessfulAttachVolume": "Volume",
    "ScalingReplicaSet": "Scaling",
}


@contextlib.contextmanager
def _quiet_sdk_logger(suppress=True):
    """Temporarily suppress SDK logger INFO output for clean deploy output.

    When suppress=True, raises the logger level to WARNING so that only
    WARNING/ERROR messages are visible. INFO-level noise (thread IDs,
    microsecond timestamps, protobuf status dumps) is hidden.

    Args:
        suppress: If True, suppress INFO. If False, no-op (verbose mode).
    """
    if not suppress:
        yield
        return
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(old_level)


class ModelDeployer:
    """Non-interactive model deployment orchestrator.

    Two modes:
    1. Local model (upload + deploy): model_path provided
    2. Existing model (deploy only): model_url provided
    """

    def __init__(
        self,
        model_path=None,
        model_url=None,
        user_id=None,
        app_id=None,
        model_version_id=None,
        instance_type=None,
        cloud_provider=None,
        region=None,
        compute_cluster_id=None,
        nodepool_id=None,
        min_replicas=1,
        max_replicas=5,
        pat=None,
        base_url=None,
        stage="runtime",
        verbose=False,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.model_id = None
        self.user_id = user_id
        self.app_id = app_id
        self.model_version_id = model_version_id
        self.instance_type = instance_type
        self.cloud_provider = cloud_provider
        self.region = region
        self.compute_cluster_id = compute_cluster_id
        self.nodepool_id = nodepool_id
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.pat = pat
        self.base_url = base_url
        self.stage = stage
        self.verbose = verbose

        # Resolved during deploy
        self._builder = None
        self._gpu_preset = None

    def deploy(self):
        """Run the full deployment pipeline. Returns a result dict."""
        self._validate()

        if self.model_path:
            return self._deploy_local_model()
        else:
            return self._deploy_existing_model()

    def _validate(self):
        """Validate inputs, fail with clear error messages."""
        if not self.model_path and not self.model_url:
            raise UserError(
                "You must specify either MODEL_PATH (directory) or --model-url.\n"
                "  Local model:    clarifai model deploy ./my-model --instance g5.xlarge\n"
                "  Existing model: clarifai model deploy --model-url <url> --instance g5.xlarge"
            )
        if self.model_path and self.model_url:
            raise UserError("Specify only one of: MODEL_PATH or --model-url.")

        if self.model_url:
            user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(self.model_url)
            self.user_id = self.user_id or user_id
            self.app_id = self.app_id or app_id
            self.model_id = model_id

            if not self.instance_type and not self.nodepool_id:
                raise UserError(
                    "You must specify --instance or --nodepool-id when deploying an existing model.\n"
                    "  Example: clarifai model deploy --model-url <url> --instance g5.xlarge\n"
                    "  Run 'clarifai model deploy --instance-info' to see available options."
                )

    def _resolve_gpu(self):
        """Resolve GPU name to preset info if gpu is specified."""
        if self.instance_type and not self._gpu_preset:
            self._gpu_preset = resolve_gpu(
                self.instance_type,
                pat=self.pat,
                base_url=self.base_url,
                cloud_provider=self.cloud_provider,
                region=self.region,
            )
        return self._gpu_preset

    def _deploy_local_model(self):
        """Upload model from local path, then deploy."""
        from clarifai.runners.models import deploy_output as out
        from clarifai.runners.models.model_builder import ModelBuilder

        suppress = not self.verbose

        # ── Validate ──
        out.phase_header("Validate")
        with _quiet_sdk_logger(suppress):
            self._builder = ModelBuilder(
                self.model_path,
                app_not_found_action="auto_create",
                pat=self.pat,
                base_url=self.base_url,
                user_id=self.user_id,
                app_id=self.app_id,
            )

        # Resolve IDs from the builder's config
        model_config = self._builder.config.get('model', {})
        self.user_id = self.user_id or model_config.get('user_id')
        self.app_id = self.app_id or model_config.get('app_id')
        self.model_id = self._builder.model_id

        # Read compute section from config (instance, cloud, region)
        compute = self._builder.config.get('compute', {})

        # Cloud and region from config (CLI flags take priority)
        if not self.cloud_provider:
            self.cloud_provider = compute.get('cloud')
        if not self.region:
            self.region = compute.get('region')

        # If instance not specified, try to read from config
        if not self.instance_type and not self.nodepool_id:
            compute_instance = compute.get('instance') or compute.get('gpu')
            if compute_instance:
                self.instance_type = compute_instance
            else:
                # Fallback: try to infer from inference_compute_info
                from clarifai.utils.compute_presets import infer_gpu_from_config

                inferred = infer_gpu_from_config(self._builder.config)
                if inferred:
                    self.instance_type = inferred
                else:
                    raise UserError(
                        "You must specify --instance or set 'compute.instance' in config.yaml.\n"
                        "  Example: clarifai model deploy ./my-model --instance g5.xlarge\n"
                        "  Or add to config.yaml:\n"
                        "    compute:\n"
                        "      instance: g5.xlarge\n"
                        "  Run 'clarifai model deploy --instance-info' to see available options."
                    )

        # Show clean validation summary
        model_type_id = model_config.get('model_type_id', 'unknown')
        instance_label = self.instance_type or 'cpu'
        checkpoints = self._builder.config.get('checkpoints', {})
        has_checkpoints = bool(checkpoints and checkpoints.get('repo_id'))
        dockerfile_path = os.path.join(self.model_path, 'Dockerfile')
        has_dockerfile = os.path.exists(dockerfile_path)

        out.info("Model", f"{self.user_id}/{self.app_id}/models/{self.model_id}")
        out.info("Type", model_type_id)
        out.info("Instance", instance_label)
        if has_checkpoints:
            out.info("Checkpoints", checkpoints.get('repo_id', ''))
        out.info("Dockerfile", "existing" if has_dockerfile else "auto-generated")

        # Only download checkpoints locally when config says when: upload
        # (they must be bundled in the tarball). when: runtime or when: build
        # means they'll be fetched inside the container in the cloud.
        checkpoint_when = checkpoints.get('when', 'runtime') if has_checkpoints else None
        with _quiet_sdk_logger(suppress):
            if checkpoint_when and checkpoint_when != 'runtime':
                self._builder.download_checkpoints(stage=self.stage)
            if has_dockerfile:
                pass  # Use existing
            else:
                self._builder.create_dockerfile(generate_dockerfile=True)

        # If the builder has no inference_compute_info (e.g. ollama config with
        # no compute section), resolve it from the --instance flag so the upload
        # has the required compute metadata.
        if self._builder.inference_compute_info is None and self.instance_type:
            from clarifai.utils.compute_presets import get_inference_compute_for_gpu

            ici = get_inference_compute_for_gpu(
                self.instance_type, pat=self.pat, base_url=self.base_url
            )
            if ici.get('num_accelerators', 0) > 0:
                ici.setdefault('accelerator_type', ['NVIDIA-*'])
            self._builder.config['inference_compute_info'] = ici
            self._builder.inference_compute_info = self._builder._get_inference_compute_info()

        # ── Upload ──
        out.phase_header("Upload")
        git_info = self._builder._get_git_info()

        # Callback emits Version/URL after upload completes but before build starts,
        # so these info lines appear under Upload, not under the Build phase header.
        def _on_upload_complete(version_id, url):
            out.info("Version", version_id)
            out.info("URL", url)

        with _quiet_sdk_logger(suppress):
            model_version_id = self._builder.upload_model_version(
                git_info,
                show_client_script=False,
                quiet_build=not self.verbose,
                post_upload_callback=_on_upload_complete,
            )

        if not model_version_id:
            raise UserError("Model upload failed. Check logs above for details.")

        self.model_version_id = model_version_id

        # Capture client script and method signatures for display after deployment
        try:
            from clarifai.runners.utils import code_script

            method_signatures = self._builder.get_method_signatures()
            self._method_signatures = method_signatures
            self._client_script = code_script.generate_client_script(
                method_signatures,
                user_id=self.user_id,
                app_id=self.app_id,
                model_id=self.model_id,
                colorize=True,
            )
        except Exception:
            self._method_signatures = None
            self._client_script = None

        # ── Deploy ──
        out.phase_header("Deploy")
        return self._create_deployment()

    def _deploy_existing_model(self):
        """Deploy an already-uploaded model."""
        from clarifai.client import Model
        from clarifai.runners.models import deploy_output as out

        suppress = not self.verbose

        out.phase_header("Deploy")

        with _quiet_sdk_logger(suppress):
            model = Model(
                model_id=self.model_id,
                app_id=self.app_id,
                user_id=self.user_id,
                pat=self.pat,
                base_url=self.base_url,
            )

            # Get latest version if not specified
            if not self.model_version_id:
                versions = list(model.list_versions())
                if not versions:
                    raise UserError(f"No versions found for model '{self.model_id}'.")
                self.model_version_id = versions[0].model_version.id

            # Auto-update compute info if the target instance exceeds model version's spec
            if self.instance_type:
                self._auto_update_compute_if_needed(model)

        out.info("Model", f"{self.user_id}/{self.app_id}/models/{self.model_id}")
        out.info("Version", self.model_version_id)
        out.info("Instance", self.instance_type or "cpu")

        return self._create_deployment()

    def _get_model_version_compute_info(self, model):
        """Fetch the model version's current inference_compute_info from the API.

        Returns:
            ComputeInfo proto, or None if not set.
        """
        from clarifai_grpc.grpc.api import service_pb2

        try:
            resp = model._grpc_request(
                model.STUB.GetModelVersion,
                service_pb2.GetModelVersionRequest(
                    user_app_id=model.user_app_id,
                    model_id=model.id,
                    version_id=self.model_version_id,
                ),
            )
            ci = resp.model_version.inference_compute_info
            # Check if compute info is actually populated (not just an empty proto)
            if ci.ByteSize() > 0:
                return ci
            return None
        except Exception as e:
            logger.debug(f"Failed to fetch model version compute info: {e}")
            return None

    @staticmethod
    def _needs_compute_update(model_compute_info, instance_compute_info):
        """Check if the model version's compute info needs to be updated for the target instance.

        The model version's inference_compute_info acts as a ceiling — the scheduler only
        places it on instances at or below those specs. If the target instance exceeds
        the model version's spec, we need to update the model version.

        Args:
            model_compute_info: ComputeInfo proto from the model version (or None).
            instance_compute_info: dict with instance's compute info from the preset.

        Returns:
            tuple: (needs_update: bool, reasons: list[str])
        """
        from clarifai.utils.compute_presets import parse_k8s_quantity

        reasons = []

        # No compute info on model version → needs update
        if model_compute_info is None:
            return True, ["model version has no inference_compute_info"]

        instance_num_acc = instance_compute_info.get("num_accelerators", 0)
        model_num_acc = model_compute_info.num_accelerators

        # num_accelerators: if instance has more GPUs than model specifies
        if instance_num_acc > model_num_acc:
            reasons.append(
                f"num_accelerators: instance has {instance_num_acc}, "
                f"model version specifies {model_num_acc}"
            )

        # accelerator_memory: if instance has more GPU memory than model specifies
        instance_acc_mem = parse_k8s_quantity(instance_compute_info.get("accelerator_memory", ""))
        model_acc_mem = parse_k8s_quantity(model_compute_info.accelerator_memory)

        if instance_acc_mem > 0 and model_acc_mem > 0 and instance_acc_mem > model_acc_mem:
            reasons.append(
                f"accelerator_memory: instance has {instance_compute_info.get('accelerator_memory')}, "
                f"model version specifies {model_compute_info.accelerator_memory}"
            )

        return len(reasons) > 0, reasons

    def _auto_update_compute_if_needed(self, model):
        """Auto-update model version compute info if the target instance exceeds its spec.

        Fetches the model version's current inference_compute_info, compares against
        the target instance, and patches if the instance exceeds the model's spec.

        Only patches num_accelerators and accelerator_memory — NOT accelerator_type,
        since the API rejects changing accelerator_type after upload.
        """
        from clarifai_grpc.grpc.api import resources_pb2

        gpu_preset = self._resolve_gpu()
        if not gpu_preset:
            return

        instance_compute = gpu_preset.get("inference_compute_info", {})

        # Fetch model version's current compute info
        model_compute = self._get_model_version_compute_info(model)

        needs_update, reasons = self._needs_compute_update(model_compute, instance_compute)

        if not needs_update:
            logger.debug("Model version compute info is compatible with target instance.")
            return

        # Build a ComputeInfo that preserves the existing accelerator_type
        # (the API rejects changes to accelerator_type) while updating
        # num_accelerators and accelerator_memory.
        existing_acc_type = list(model_compute.accelerator_type) if model_compute else ["NVIDIA-*"]
        patch_compute = resources_pb2.ComputeInfo(
            num_accelerators=instance_compute.get("num_accelerators", 0),
            accelerator_memory=instance_compute.get("accelerator_memory", ""),
            accelerator_type=existing_acc_type,
        )

        reason_str = "; ".join(reasons)
        if self.verbose:
            logger.info(
                f"Updating model version compute info to match instance "
                f"'{self.instance_type}' ({reason_str})"
            )
        model.patch_version(
            version_id=self.model_version_id,
            inference_compute_info=patch_compute,
        )

    def _get_cloud_and_region(self):
        """Determine cloud_provider and region for infrastructure creation.

        Priority:
        1. Explicit --cloud/--region flags
        2. Cloud/region from the resolved GPU preset
        3. Default: aws / us-east-1
        """
        gpu_preset = self._resolve_gpu()

        cloud = self.cloud_provider
        region = self.region

        if gpu_preset:
            cloud = cloud or gpu_preset.get("cloud_provider")
            region = region or gpu_preset.get("region")

        cloud = cloud or "aws"
        region = region or "us-east-1"

        return cloud, region

    def _ensure_compute_infrastructure(self):
        """Auto-create compute cluster and nodepool if needed.

        Returns:
            tuple: (compute_cluster_id, nodepool_id, cluster_user_id)
        """
        if self.nodepool_id and self.compute_cluster_id:
            return self.compute_cluster_id, self.nodepool_id, self.user_id

        from clarifai.client.user import User
        from clarifai.runners.models import deploy_output as out

        suppress = not self.verbose

        with _quiet_sdk_logger(suppress):
            user = User(user_id=self.user_id, pat=self.pat, base_url=self.base_url)
        gpu_preset = self._resolve_gpu()
        cloud, region = self._get_cloud_and_region()

        # Determine compute cluster ID (cloud/region-aware)
        cc_id = self.compute_cluster_id or get_deploy_compute_cluster_id(cloud, region)

        # Try to get existing compute cluster, create if not found
        with _quiet_sdk_logger(suppress):
            try:
                user.compute_cluster(cc_id)
            except Exception:
                out.status(f"Creating compute cluster '{cc_id}'...")
                cc_config = get_compute_cluster_config(self.user_id, cloud, region)
                user.create_compute_cluster(compute_cluster_config=cc_config)

        # Determine nodepool ID
        if self.nodepool_id:
            np_id = self.nodepool_id
        else:
            instance_type_id = gpu_preset["instance_type_id"] if gpu_preset else "cpu-t3a-2xlarge"
            np_id = get_deploy_nodepool_id(instance_type_id)

            # Try to get existing nodepool, create if not found
            from clarifai.client.compute_cluster import ComputeCluster

            with _quiet_sdk_logger(suppress):
                cc = ComputeCluster(
                    compute_cluster_id=cc_id,
                    user_id=self.user_id,
                    pat=self.pat,
                    base_url=self.base_url,
                )
                try:
                    cc.nodepool(np_id)
                except Exception:
                    out.status(f"Creating nodepool '{np_id}'...")
                    np_config = get_nodepool_config(
                        instance_type_id=instance_type_id,
                        compute_cluster_id=cc_id,
                        user_id=self.user_id,
                        compute_info=gpu_preset.get("inference_compute_info")
                        if gpu_preset
                        else None,
                    )
                    cc.create_nodepool(nodepool_config=np_config)

        return cc_id, np_id, self.user_id

    def _create_deployment(self):
        """Create the deployment using existing deploy_model function."""
        from clarifai.runners.models import deploy_output as out
        from clarifai.runners.models.model_builder import deploy_model

        cc_id, np_id, cluster_user_id = self._ensure_compute_infrastructure()

        deployment_id = f"deploy-{self.model_id}-{uuid.uuid4().hex[:6]}"
        suppress = not self.verbose

        out.status(f"Deploying to nodepool '{np_id}'...")
        with _quiet_sdk_logger(suppress):
            success = deploy_model(
                model_id=self.model_id,
                app_id=self.app_id,
                user_id=self.user_id,
                deployment_id=deployment_id,
                model_version_id=self.model_version_id,
                nodepool_id=np_id,
                compute_cluster_id=cc_id,
                cluster_user_id=cluster_user_id,
                min_replicas=self.min_replicas,
                max_replicas=self.max_replicas,
                pat=self.pat,
                base_url=self.base_url,
                quiet=suppress,
            )

        if not success:
            raise UserError(
                f"Deployment failed for model '{self.model_id}'. Check logs above for details."
            )

        out.success(f"Deployment '{deployment_id}' created")

        # Skip monitoring when min_replicas is 0 — no pods will be scheduled
        if self.min_replicas > 0:
            self._monitor_deployment(deployment_id, np_id, cc_id)

        return self._format_result(deployment_id, np_id, cc_id)

    def _monitor_deployment(self, deployment_id, nodepool_id, compute_cluster_id):
        """Monitor deployment status until runner pods are ready or timeout.

        Polls runner status and fetches runner logs to show the user what's happening
        after the deployment is created (pod scheduling, image pulling, model loading).
        """
        from clarifai_grpc.grpc.api import service_pb2

        from clarifai.client.auth import create_stub
        from clarifai.client.auth.helper import ClarifaiAuthHelper
        from clarifai.runners.models import deploy_output as out

        out.phase_header("Monitor")

        # Create a lightweight client for gRPC calls
        auth = ClarifaiAuthHelper.from_env(
            user_id=self.user_id, pat=self.pat, base=self.base_url, validate=False
        )
        stub = create_stub(auth)
        user_app_id = auth.get_user_app_id_proto()

        timeout = DEFAULT_MONITOR_TIMEOUT
        poll_interval = DEFAULT_POLL_INTERVAL
        start_time = time.time()
        seen_logs = set()
        seen_messages = set()  # Dedup simplified event messages across polls
        log_page = 1
        has_inline_progress = False  # Track if we printed \r progress

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # List runners for our model version in this nodepool
            try:
                resp = stub.ListRunners(
                    service_pb2.ListRunnersRequest(
                        user_app_id=user_app_id,
                        compute_cluster_id=compute_cluster_id,
                        nodepool_id=nodepool_id,
                        model_version_ids=[self.model_version_id],
                    )
                )
                runners = list(resp.runners)
            except Exception as e:
                logger.debug(f"Error listing runners: {e}")
                runners = []

            if runners:
                runner = runners[0]
                metrics = runner.runner_metrics
                pods_running = metrics.pods_running if metrics else 0
                pods_total = metrics.pods_total if metrics else 0

                # Collect new log lines (without printing yet)
                log_page, log_lines = self._fetch_runner_logs(
                    stub,
                    user_app_id,
                    compute_cluster_id,
                    nodepool_id,
                    runner.id,
                    seen_logs,
                    log_page,
                    verbose=self.verbose,
                    seen_messages=seen_messages,
                )

                # Print log lines if any (clear inline progress first)
                if log_lines:
                    if has_inline_progress:
                        out.clear_inline()
                        has_inline_progress = False
                    for line in log_lines:
                        print(line, flush=True)

                # Check if ready
                if pods_running >= max(self.min_replicas, 1):
                    # Brief delay to let late-arriving k8s events propagate to the API,
                    # then fetch one final time so fast deploys still show events.
                    time.sleep(3)
                    _, final_lines = self._fetch_runner_logs(
                        stub,
                        user_app_id,
                        compute_cluster_id,
                        nodepool_id,
                        runner.id,
                        seen_logs,
                        log_page,
                        verbose=self.verbose,
                        seen_messages=seen_messages,
                    )
                    if final_lines:
                        if has_inline_progress:
                            out.clear_inline()
                            has_inline_progress = False
                        for line in final_lines:
                            print(line, flush=True)

                    if has_inline_progress:
                        out.clear_inline()
                        has_inline_progress = False
                    out.success(
                        f"Model is running! Pods: {pods_running}/{pods_total} ({elapsed}s)"
                    )
                    # Tail model logs briefly to show startup output
                    self._tail_runner_logs(
                        stub,
                        user_app_id,
                        compute_cluster_id,
                        nodepool_id,
                        runner.id,
                    )
                    return

                status_msg = f"Pods: {pods_running}/{pods_total} running ({elapsed}s elapsed)"
            else:
                status_msg = f"Waiting for runner to be scheduled... ({elapsed}s elapsed)"

            # Inline progress update (overwrite same line)
            out.inline_progress(status_msg)
            has_inline_progress = True

            time.sleep(poll_interval)

        # Timeout reached
        if has_inline_progress:
            out.clear_inline()
        out.warning(
            f"Deployment monitoring timed out after {timeout}s. "
            "The deployment may still be in progress."
        )
        out.status("Check status with: clarifai deployment list")

    @staticmethod
    def _fetch_runner_logs(
        stub,
        user_app_id,
        compute_cluster_id,
        nodepool_id,
        runner_id,
        seen_logs,
        current_page,
        verbose=False,
        seen_messages=None,
    ):
        """Fetch k8s event logs during monitoring.

        Only fetches "runner.events" (k8s events like scheduling, pulling, starting).
        Model stdout/stderr ("runner" logs) are reserved for the Startup Logs phase
        to avoid consuming them here and leaving that section empty.

        Args:
            seen_messages: Optional set for deduplicating simplified output lines across
                poll cycles. When not verbose, repeated messages like "Scheduling: Waiting
                for node..." are suppressed after the first occurrence.
        """
        from clarifai_grpc.grpc.api import service_pb2

        lines = []

        try:
            resp = stub.ListLogEntries(
                service_pb2.ListLogEntriesRequest(
                    log_type="runner.events",
                    user_app_id=user_app_id,
                    compute_cluster_id=compute_cluster_id,
                    nodepool_id=nodepool_id,
                    runner_id=runner_id,
                    page=current_page,
                    per_page=50,
                )
            )
            for entry in resp.log_entries:
                log_key = ("runner.events", entry.url or entry.message[:100])
                if log_key not in seen_logs:
                    seen_logs.add(log_key)
                    event_lines = _format_event_logs(entry.message.strip(), verbose=verbose)
                    for line in event_lines:
                        # Deduplicate simplified messages across polls
                        if seen_messages is not None and not verbose:
                            if line in seen_messages:
                                continue
                            seen_messages.add(line)
                        lines.append(line)
        except Exception:
            pass  # Log fetching is best-effort

        return current_page, lines

    def _format_result(self, deployment_id, nodepool_id, compute_cluster_id):
        """Format deployment result."""
        ui_base = "https://clarifai.com"
        model_url = f"{ui_base}/{self.user_id}/{self.app_id}/models/{self.model_id}"

        instance_desc = ""
        if self._gpu_preset:
            desc = self._gpu_preset['description']
            # Avoid redundant display like "g5.2xlarge (g5.2xlarge)"
            if desc and desc.lower() != self.instance_type.lower():
                instance_desc = f"{self.instance_type} ({desc})"
            else:
                instance_desc = self.instance_type
        elif self.instance_type:
            instance_desc = self.instance_type

        cloud, region = self._get_cloud_and_region()

        return {
            "model_url": model_url,
            "model_id": self.model_id,
            "model_version_id": self.model_version_id,
            "deployment_id": deployment_id,
            "nodepool_id": nodepool_id,
            "compute_cluster_id": compute_cluster_id,
            "instance_type": instance_desc,
            "cloud_provider": cloud,
            "region": region,
            "user_id": self.user_id,
            "app_id": self.app_id,
            "client_script": getattr(self, '_client_script', None),
            "method_signatures": getattr(self, '_method_signatures', None),
        }

    def _tail_runner_logs(self, stub, user_app_id, compute_cluster_id, nodepool_id, runner_id):
        """Briefly tail runner logs after pods are ready to show model startup output.

        Fetches log_type="runner" (model pod stdout/stderr) for a short period
        so the user can see model loading progress, then exits with a hint.

        Uses the same print-raw approach as stream_model_logs() for reliability,
        with optional JSON parsing for cleaner output.
        """
        from clarifai_grpc.grpc.api import service_pb2

        from clarifai.runners.models import deploy_output as out

        model_url = f"https://clarifai.com/{self.user_id}/{self.app_id}/models/{self.model_id}"

        seen_logs = set()
        log_page = 1
        has_logs = False
        tail_start = time.time()
        total_api_entries = 0  # Track raw API entry count for diagnostics

        while time.time() - tail_start < DEFAULT_LOG_TAIL_DURATION:
            new_entries = 0
            try:
                resp = stub.ListLogEntries(
                    service_pb2.ListLogEntriesRequest(
                        log_type="runner",
                        user_app_id=user_app_id,
                        compute_cluster_id=compute_cluster_id or "",
                        nodepool_id=nodepool_id or "",
                        runner_id=runner_id,
                        page=log_page,
                        per_page=50,
                    )
                )
                entries_count = 0
                for entry in resp.log_entries:
                    entries_count += 1
                    total_api_entries += 1
                    log_key = entry.url or entry.message[:100]
                    if log_key in seen_logs:
                        continue
                    seen_logs.add(log_key)
                    msg = entry.message.strip()
                    if not msg:
                        continue

                    # Try to extract clean message from JSON logs.
                    parsed = _parse_runner_log(msg, verbose=self.verbose)
                    display = parsed
                    if not display and self.verbose:
                        display = msg[:200]

                    if display:
                        if not has_logs:
                            out.phase_header("Startup Logs")
                            has_logs = True
                        out.status(display)
                        new_entries += 1
                if entries_count == 50:
                    log_page += 1
            except Exception as e:
                # Make errors visible — logger.debug is not shown at default level
                out.event(f"Log fetch error: {e}")

            # If we displayed logs and then an empty poll, we're done
            if has_logs and new_entries == 0:
                break

            time.sleep(3)

        if not has_logs:
            out.phase_header("Startup Logs")
            if total_api_entries > 0:
                out.status(
                    f"{total_api_entries} log entries found but all filtered "
                    f"(use --verbose to see). Logs may appear shortly."
                )
            else:
                out.status("No startup logs available yet.")

        out.status("")
        out.status("Stream model logs:")
        out.status(f'  clarifai model logs --model-url "{model_url}"')


def _parse_runner_log(raw_msg, verbose=False):
    """Parse a runner log line, extracting the message from JSON if possible.

    Raw input example:
        '{"msg": "Starting MCP bridge...", "@timestamp": "...", "stack_info": null, ...}'
    Output: "Starting MCP bridge..."

    In non-verbose mode, filters out:
    - DeprecationWarning lines
    - pip download progress lines
    - Empty messages

    Args:
        raw_msg: Raw log message string.
        verbose: If True, pass through all messages unfiltered.

    Returns:
        Cleaned message string, or None if the message should be suppressed.
    """
    if not raw_msg:
        return None
    # Try to parse as JSON (runner logs are often JSON-formatted)
    try:
        data = json.loads(raw_msg)
        if isinstance(data, dict) and "msg" in data:
            msg = data["msg"]
            if msg and isinstance(msg, str):
                # Decode unicode escapes if present (e.g. \ud83d\ude80 → emoji)
                try:
                    msg = msg.encode('utf-16', 'surrogatepass').decode('utf-16')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
                return msg
            return None
    except (json.JSONDecodeError, TypeError):
        pass

    # Return raw message as-is
    return raw_msg


def _simplify_k8s_message(reason, message):
    """Simplify k8s event messages for non-verbose mode.

    Strips internal node IPs, taint specifications, and pod full names.

    Args:
        reason: K8s event reason (e.g. "FailedScheduling", "Pulling").
        message: Raw k8s event message.

    Returns:
        Simplified, human-friendly message string.
    """
    _SIMPLE = {
        "FailedScheduling": "Waiting for node to become available...",
        "NotTriggerScaleUp": "Waiting for cluster to scale up...",
        "NominatedNode": "Node selected for scheduling",
        "Nominated": "Node selected for scheduling",
        "Scheduled": "Pod scheduled on node",
        "Pulling": "Pulling model image...",
        "Pulled": "Model image pulled",
        "Created": "Container created",
        "Started": "Container started",
        "BackOff": "Container restarting (back-off)",
        "Unhealthy": "Health check failed, waiting...",
        "Killing": "Stopping container...",
        "Preempted": "Pod preempted, rescheduling...",
        "FailedMount": "Volume mount failed",
        "FailedAttachVolume": "Volume attach failed",
        "SuccessfulAttachVolume": "Volume attached",
        "ScalingReplicaSet": "Scaling replicas...",
    }
    simplified = _SIMPLE.get(reason)
    if simplified:
        return simplified
    # Truncate anything beyond 80 chars
    if len(message) > 80:
        return message[:77] + "..."
    return message


def _format_event_logs(raw_message, verbose=False):
    """Parse Kubernetes-style event log entries into formatted lines.

    Raw format: "Name: pod-xyz, Type: Warning, Source: {karpenter }, Reason: FailedScheduling,
                 FirstTimestamp: ..., LastTimestamp: ..., Message: ..."
    Multiple events may be concatenated with newlines.

    Args:
        raw_message: Raw k8s event log string.
        verbose: If True, show all events with full detail. If False, simplify and filter.

    Returns:
        list of formatted strings.
    """
    if not raw_message:
        return []

    lines = []
    # Split concatenated events (each starts with "Name:")
    events = re.split(r'\n(?=Name:\s)', raw_message)

    for event_str in events:
        event_str = event_str.strip()
        if not event_str:
            continue

        # Extract key fields
        type_match = re.search(r'Type:\s*(\w+)', event_str)
        reason_match = re.search(r'Reason:\s*(\w+)', event_str)
        message_match = re.search(r'Message:\s*(.+?)(?:\s*$)', event_str, re.DOTALL)

        event_type = type_match.group(1) if type_match else ""
        reason = reason_match.group(1) if reason_match else ""
        message = message_match.group(1).strip() if message_match else event_str

        # In non-verbose mode, skip transient noise events
        if not verbose and reason in _SKIP_EVENTS:
            continue

        # In non-verbose mode, simplify messages
        if not verbose:
            message = _simplify_k8s_message(reason, message)
            phase = _EVENT_PHASE_MAP.get(reason, reason)
        else:
            phase = reason
            # Truncate very long messages in verbose mode too
            if len(message) > 200:
                message = message[:197] + "..."

        # Format with type indicator (consistent width)
        tag = "warning" if event_type == "Warning" else "event"

        if phase:
            lines.append(f"  [{tag:7s}] {phase}: {message}")
        else:
            lines.append(f"  [{tag:7s}] {message}")

    return lines


def stream_model_logs(
    model_url=None,
    model_id=None,
    user_id=None,
    app_id=None,
    model_version_id=None,
    compute_cluster_id=None,
    nodepool_id=None,
    pat=None,
    base_url=None,
    follow=True,
    duration=None,
):
    """Stream model runner logs to stdout.

    Looks up the runner for the given model, then continuously fetches and prints
    log_type="runner" entries (model pod stdout/stderr).

    Args:
        model_url: Clarifai model URL. Used to extract user_id, app_id, model_id.
        model_id: Model ID (alternative to model_url).
        user_id: User ID.
        app_id: App ID.
        model_version_id: Specific version (default: latest).
        compute_cluster_id: Filter by compute cluster.
        nodepool_id: Filter by nodepool.
        pat: PAT for auth.
        base_url: API base URL.
        follow: If True, continuously tail logs. If False, print existing and exit.
        duration: Max seconds to tail (None = until Ctrl+C).
    """
    from clarifai_grpc.grpc.api import service_pb2

    from clarifai.client.auth import create_stub
    from clarifai.client.auth.helper import ClarifaiAuthHelper

    # Parse model URL if provided
    if model_url:
        url_user, url_app, _, url_model, _ = ClarifaiUrlHelper.split_clarifai_url(model_url)
        user_id = user_id or url_user
        app_id = app_id or url_app
        model_id = model_id or url_model

    if not model_id or not user_id:
        raise UserError(
            "You must specify --model-url or --model-id with --user-id.\n"
            "  Example: clarifai model logs --model-url https://clarifai.com/user/app/models/id"
        )

    # Get latest version if not specified
    if not model_version_id:
        from clarifai.client import Model

        model = Model(
            model_id=model_id,
            app_id=app_id,
            user_id=user_id,
            pat=pat,
            base_url=base_url,
        )
        versions = list(model.list_versions())
        if not versions:
            raise UserError(f"No versions found for model '{model_id}'.")
        model_version_id = versions[0].model_version.id

    # Create gRPC client
    auth = ClarifaiAuthHelper.from_env(user_id=user_id, pat=pat, base=base_url, validate=False)
    stub = create_stub(auth)
    user_app_id = auth.get_user_app_id_proto()

    # Find the runner
    runner_id = None
    cc_id = compute_cluster_id
    np_id = nodepool_id

    try:
        resp = stub.ListRunners(
            service_pb2.ListRunnersRequest(
                user_app_id=user_app_id,
                compute_cluster_id=cc_id or "",
                nodepool_id=np_id or "",
                model_version_ids=[model_version_id],
            )
        )
        runners = list(resp.runners)
        if runners:
            runner = runners[0]
            runner_id = runner.id
            # Extract cc/np from runner if not provided
            if not cc_id and runner.nodepool:
                cc_id = (
                    runner.nodepool.compute_cluster.id if runner.nodepool.compute_cluster else ""
                )
                np_id = runner.nodepool.id
    except Exception as e:
        logger.debug(f"Error listing runners: {e}")

    if not runner_id:
        url_hint = model_url or f"https://clarifai.com/{user_id}/{app_id}/models/{model_id}"
        raise UserError(
            f"No active runner found for model '{model_id}' (version: {model_version_id}).\n"
            "  The model is not currently deployed. To deploy it, run:\n"
            f"    clarifai model deploy --model-url \"{url_hint}\" --instance <instance-type>\n"
            "  Run 'clarifai model deploy --instance-info' to see available instance types."
        )

    print(f"Streaming logs for model '{model_id}' (runner: {runner_id})...", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    seen_logs = set()
    log_page = 1
    start_time = time.time()
    poll_interval = 3  # Slightly faster polling for log streaming

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break

            try:
                resp = stub.ListLogEntries(
                    service_pb2.ListLogEntriesRequest(
                        log_type="runner",
                        user_app_id=user_app_id,
                        compute_cluster_id=cc_id or "",
                        nodepool_id=np_id or "",
                        runner_id=runner_id,
                        page=log_page,
                        per_page=50,
                    )
                )
                entries_count = 0
                for entry in resp.log_entries:
                    entries_count += 1
                    log_key = entry.url or entry.message[:100]
                    if log_key not in seen_logs:
                        seen_logs.add(log_key)
                        msg = entry.message.strip()
                        if msg:
                            print(msg, flush=True)
                if entries_count == 50:
                    log_page += 1
            except Exception as e:
                logger.debug(f"Error fetching logs: {e}")

            if not follow:
                break

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nStopped log streaming.", flush=True)
