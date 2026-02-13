"""Non-interactive model deployment orchestrator.

Orchestrates upload + build + deploy in a single command with zero prompts.
All decisions come from parameters or config. Errors are raised as UserError
with actionable fix instructions. This class NEVER calls input(), click.confirm(),
click.prompt(), or any interactive function.
"""

import os
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
        gpu=None,
        cloud_provider=None,
        region=None,
        compute_cluster_id=None,
        nodepool_id=None,
        deployment_id=None,
        min_replicas=1,
        max_replicas=5,
        pat=None,
        base_url=None,
        stage="runtime",
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.model_id = None
        self.user_id = user_id
        self.app_id = app_id
        self.model_version_id = model_version_id
        self.gpu = gpu
        self.cloud_provider = cloud_provider
        self.region = region
        self.compute_cluster_id = compute_cluster_id
        self.nodepool_id = nodepool_id
        self.deployment_id = deployment_id
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.pat = pat
        self.base_url = base_url
        self.stage = stage

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
                "  Local model:    clarifai model deploy ./my-model --gpu g5.xlarge\n"
                "  Existing model: clarifai model deploy --model-url <url> --gpu g5.xlarge"
            )
        if self.model_path and self.model_url:
            raise UserError("Specify only one of: MODEL_PATH or --model-url.")

        if self.model_url:
            user_id, app_id, _, model_id, _ = ClarifaiUrlHelper.split_clarifai_url(self.model_url)
            self.user_id = self.user_id or user_id
            self.app_id = self.app_id or app_id
            self.model_id = model_id

            if not self.gpu and not self.nodepool_id:
                raise UserError(
                    "You must specify --gpu or --nodepool-id when deploying an existing model.\n"
                    "  Example: clarifai model deploy --model-url <url> --gpu g5.xlarge\n"
                    "  Run 'clarifai model deploy --gpu-info' to see available options."
                )

    def _resolve_gpu(self):
        """Resolve GPU name to preset info if gpu is specified."""
        if self.gpu and not self._gpu_preset:
            self._gpu_preset = resolve_gpu(
                self.gpu,
                pat=self.pat,
                base_url=self.base_url,
                cloud_provider=self.cloud_provider,
                region=self.region,
            )
        return self._gpu_preset

    def _deploy_local_model(self):
        """Upload model from local path, then deploy."""
        from clarifai.runners.models.model_builder import ModelBuilder

        logger.info("Validating model config...")
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

        # If gpu not specified, try to infer from config
        if not self.gpu:
            from clarifai.utils.compute_presets import infer_gpu_from_config

            inferred = infer_gpu_from_config(self._builder.config)
            if inferred:
                self.gpu = inferred
                logger.info(f"Inferred GPU type from config: {self.gpu}")

        # Download checkpoints
        self._builder.download_checkpoints(stage=self.stage)

        # Create dockerfile (skip if one already exists in the model folder)
        dockerfile_path = os.path.join(self.model_path, 'Dockerfile')
        if os.path.exists(dockerfile_path):
            logger.info("Using existing Dockerfile.")
        else:
            self._builder.create_dockerfile(generate_dockerfile=True)

        # Upload
        logger.info("Uploading model...")
        git_info = self._builder._get_git_info()
        model_version_id = self._builder.upload_model_version(git_info)
        if not model_version_id:
            raise UserError("Model upload failed. Check logs above for details.")

        self.model_version_id = model_version_id
        logger.info(f"Model uploaded. Version: {self.model_version_id}")

        # Deploy
        return self._create_deployment()

    def _deploy_existing_model(self):
        """Deploy an already-uploaded model."""
        from clarifai.client import Model

        # Get latest version if not specified
        if not self.model_version_id:
            logger.info("Fetching latest model version...")
            model = Model(
                model_id=self.model_id,
                app_id=self.app_id,
                user_id=self.user_id,
                pat=self.pat,
                base_url=self.base_url,
            )
            versions = list(model.list_versions())
            if not versions:
                raise UserError(f"No versions found for model '{self.model_id}'.")
            self.model_version_id = versions[0].model_version.id
            logger.info(f"Using latest version: {self.model_version_id}")

        return self._create_deployment()

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

        user = User(user_id=self.user_id, pat=self.pat, base_url=self.base_url)
        gpu_preset = self._resolve_gpu()
        cloud, region = self._get_cloud_and_region()

        # Determine compute cluster ID (cloud/region-aware)
        cc_id = self.compute_cluster_id or get_deploy_compute_cluster_id(cloud, region)

        # Try to get existing compute cluster, create if not found
        try:
            user.compute_cluster(cc_id)
            logger.debug(f"Using existing compute cluster: {cc_id}")
        except Exception:
            logger.info(f"Creating compute cluster: {cc_id}")
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

            cc = ComputeCluster(
                compute_cluster_id=cc_id,
                user_id=self.user_id,
                pat=self.pat,
                base_url=self.base_url,
            )
            try:
                cc.nodepool(np_id)
                logger.debug(f"Using existing nodepool: {np_id}")
            except Exception:
                logger.info(f"Creating nodepool: {np_id}")
                np_config = get_nodepool_config(
                    instance_type_id=instance_type_id,
                    compute_cluster_id=cc_id,
                    user_id=self.user_id,
                    compute_info=gpu_preset.get("inference_compute_info") if gpu_preset else None,
                )
                cc.create_nodepool(nodepool_config=np_config)

        return cc_id, np_id, self.user_id

    def _create_deployment(self):
        """Create the deployment using existing deploy_model function."""
        from clarifai.runners.models.model_builder import deploy_model

        cc_id, np_id, cluster_user_id = self._ensure_compute_infrastructure()

        deployment_id = self.deployment_id or f"deploy-{self.model_id}-{uuid.uuid4().hex[:6]}"

        logger.info(f"Deploying model '{self.model_id}' to nodepool '{np_id}'...")
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
        )

        if not success:
            raise UserError(
                f"Deployment failed for model '{self.model_id}'. Check logs above for details."
            )

        return self._format_result(deployment_id, np_id, cc_id)

    def _format_result(self, deployment_id, nodepool_id, compute_cluster_id):
        """Format deployment result."""
        ui_base = "https://clarifai.com"
        model_url = f"{ui_base}/{self.user_id}/{self.app_id}/models/{self.model_id}"

        gpu_desc = ""
        if self._gpu_preset:
            gpu_desc = f"{self.gpu} ({self._gpu_preset['description']})"
        elif self.gpu:
            gpu_desc = self.gpu

        return {
            "model_url": model_url,
            "model_id": self.model_id,
            "model_version_id": self.model_version_id,
            "deployment_id": deployment_id,
            "nodepool_id": nodepool_id,
            "compute_cluster_id": compute_cluster_id,
            "gpu": gpu_desc,
            "user_id": self.user_id,
            "app_id": self.app_id,
        }
