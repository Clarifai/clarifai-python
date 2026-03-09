"""Tests for model deploy, config normalization, and GPU presets."""

import os
import tempfile

import pytest
import requests
import yaml

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.compute_presets import (
    FALLBACK_GPU_PRESETS,
    _detect_quant_from_repo_name,
    _estimate_kv_cache_bytes,
    _estimate_vram_bytes,
    _estimate_weight_bytes,
    _get_hf_model_config,
    _get_hf_token,
    _select_instance_by_vram,
    get_accelerator_wildcard,
    get_compute_cluster_config,
    get_deploy_compute_cluster_id,
    get_deploy_nodepool_id,
    get_hf_model_info,
    get_inference_compute_for_gpu,
    get_nodepool_config,
    infer_gpu_from_config,
    list_gpu_presets,
    parse_k8s_quantity,
    recommend_instance,
    resolve_gpu,
)


class TestGPUPresets:
    """Test GPU preset resolution and lookup."""

    def test_resolve_known_gpu_fallback(self):
        """Known GPU names resolve from fallback presets."""
        for name in ["A10G", "L40S", "G6E", "CPU"]:
            preset = resolve_gpu(name)
            assert "description" in preset
            assert "instance_type_id" in preset
            assert "inference_compute_info" in preset

    def test_resolve_instance_type_id_case_insensitive(self):
        """GPU names should be case-insensitive."""
        preset_lower = resolve_gpu("a10g")
        preset_upper = resolve_gpu("A10G")
        assert preset_lower["instance_type_id"] == preset_upper["instance_type_id"]

    def test_resolve_unknown_gpu_raises(self):
        """Unknown GPU name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown instance type"):
            resolve_gpu("NONEXISTENT_GPU")

    def test_get_inference_compute_for_gpu(self):
        """Returns a dict with expected compute info keys."""
        info = get_inference_compute_for_gpu("A10G")
        assert "cpu_limit" in info
        assert "cpu_memory" in info
        assert "num_accelerators" in info
        assert "accelerator_type" in info
        assert info["num_accelerators"] == 1
        assert "NVIDIA-A10G" in info["accelerator_type"]

    def test_get_inference_compute_for_cpu(self):
        """CPU preset has no accelerators."""
        info = get_inference_compute_for_gpu("CPU")
        assert info["num_accelerators"] == 0

    def test_infer_gpu_from_config_a10g(self):
        """Infer A10G from inference_compute_info."""
        config = {
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 1,
                "accelerator_type": ["NVIDIA-A10G"],
                "accelerator_memory": "24Gi",
            }
        }
        assert infer_gpu_from_config(config) == "A10G"

    def test_infer_gpu_from_config_cpu(self):
        """Infer CPU when no accelerators."""
        config = {
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 0,
                "accelerator_type": [],
            }
        }
        assert infer_gpu_from_config(config) == "CPU"

    def test_infer_gpu_from_config_missing(self):
        """Returns None when no inference_compute_info."""
        assert infer_gpu_from_config({}) is None

    def test_list_gpu_presets_returns_string(self):
        """list_gpu_presets returns a string (API data or login message)."""
        result = list_gpu_presets()
        assert isinstance(result, str)
        # Either shows instance types from API or a login prompt
        assert "instance type" in result.lower() or "logged in" in result.lower()

    def test_resolve_gpu_nvidia_prefix_via_api(self):
        """'gpu-nvidia-a10g' should resolve to real API instance type, not fallback."""
        from unittest.mock import MagicMock, patch

        mock_g5 = MagicMock()
        mock_g5.id = "g5.xlarge"
        mock_g5.description = "NVIDIA A10G 24GB"
        mock_g5.cloud_provider.id = "aws"
        mock_g5.region = "us-east-1"
        mock_g5.compute_info.cpu_limit = "4"
        mock_g5.compute_info.cpu_memory = "16Gi"
        mock_g5.compute_info.num_accelerators = 1
        mock_g5.compute_info.accelerator_type = ["NVIDIA-A10G"]
        mock_g5.compute_info.accelerator_memory = "24Gi"

        with patch(
            "clarifai.utils.compute_presets._try_list_all_instance_types",
            return_value=[mock_g5],
        ):
            # 'gpu-nvidia-a10g' should normalize to 'A10G' and match via accelerator_type
            preset = resolve_gpu("gpu-nvidia-a10g")
            assert preset["instance_type_id"] == "g5.xlarge"
            assert preset["cloud_provider"] == "aws"

    def test_fallback_presets_complete(self):
        """All fallback presets have required keys."""
        for name, preset in FALLBACK_GPU_PRESETS.items():
            assert "description" in preset, f"Missing description for {name}"
            assert "instance_type_id" in preset, f"Missing instance_type_id for {name}"
            assert "inference_compute_info" in preset, f"Missing inference_compute_info for {name}"
            ici = preset["inference_compute_info"]
            assert "cpu_limit" in ici
            assert "num_accelerators" in ici


class TestComputeConfigs:
    """Test compute cluster and nodepool config generation."""

    def test_compute_cluster_config(self):
        """Cluster config has required fields."""
        config = get_compute_cluster_config("test-user")
        cc = config["compute_cluster"]
        assert cc["id"] == "deploy-cc-aws-us-east-1"
        assert cc["cloud_provider"]["id"] == "aws"
        assert cc["region"] == "us-east-1"
        assert cc["managed_by"] == "clarifai"

    def test_compute_cluster_config_custom_cloud(self):
        """Cluster config uses specified cloud and region."""
        config = get_compute_cluster_config(
            "test-user", cloud_provider="gcp", region="us-central1"
        )
        cc = config["compute_cluster"]
        assert cc["id"] == "deploy-cc-gcp-us-central1"
        assert cc["cloud_provider"]["id"] == "gcp"
        assert cc["region"] == "us-central1"

    def test_deploy_ids(self):
        """Compute cluster and nodepool IDs are deterministic."""
        assert get_deploy_compute_cluster_id("aws", "us-east-1") == "deploy-cc-aws-us-east-1"
        assert get_deploy_compute_cluster_id("gcp", "us-central1") == "deploy-cc-gcp-us-central1"
        assert get_deploy_nodepool_id("g5.xlarge") == "deploy-np-g5-xlarge"
        assert get_deploy_nodepool_id("gpu-nvidia-a10g") == "deploy-np-gpu-nvidia-a10g"
        assert get_deploy_nodepool_id("g5.2xlarge") == "deploy-np-g5-2xlarge"

    def test_nodepool_config(self):
        """Nodepool config has required fields."""
        config = get_nodepool_config(
            instance_type_id="gpu-nvidia-a10g",
            compute_cluster_id="test-cc",
            user_id="test-user",
        )
        np = config["nodepool"]
        assert np["id"] == "deploy-np-gpu-nvidia-a10g"
        assert np["compute_cluster"]["id"] == "test-cc"
        assert np["compute_cluster"]["user_id"] == "test-user"
        assert len(np["instance_types"]) == 1
        assert np["instance_types"][0]["id"] == "gpu-nvidia-a10g"

    def test_nodepool_config_with_compute_info(self):
        """Nodepool config includes compute_info when provided."""
        ci = {"cpu_limit": "4", "num_accelerators": 1}
        config = get_nodepool_config(
            instance_type_id="gpu-nvidia-a10g",
            compute_cluster_id="test-cc",
            user_id="test-user",
            compute_info=ci,
        )
        np = config["nodepool"]
        assert np["instance_types"][0]["compute_info"] == ci


class TestNormalizeConfig:
    """Test ModelBuilder.normalize_config()."""

    def test_inject_user_id_and_app_id(self):
        """user_id and app_id injected when missing."""
        config = {"model": {"id": "test", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config, user_id="user1", app_id="app1")
        assert result["model"]["user_id"] == "user1"
        assert result["model"]["app_id"] == "app1"

    def test_existing_user_id_not_overwritten(self):
        """Existing user_id/app_id are preserved."""
        config = {
            "model": {
                "id": "test",
                "model_type_id": "text-to-text",
                "user_id": "existing",
                "app_id": "existing-app",
            }
        }
        result = ModelBuilder.normalize_config(config, user_id="new-user", app_id="new-app")
        assert result["model"]["user_id"] == "existing"
        assert result["model"]["app_id"] == "existing-app"

    def test_default_app_id_when_missing(self):
        """app_id defaults to 'main' when not in config and not provided."""
        config = {"model": {"id": "test", "user_id": "user1", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["app_id"] == "main"

    def test_default_app_id_not_applied_when_provided(self):
        """Default app_id is not used when explicitly provided via parameter."""
        config = {"model": {"id": "test", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config, app_id="my-custom-app")
        assert result["model"]["app_id"] == "my-custom-app"

    def test_default_app_id_not_applied_when_in_config(self):
        """Default app_id is not used when already in config."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text", "app_id": "existing-app"}
        }
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["app_id"] == "existing-app"

    def test_default_model_type_id_when_missing(self):
        """model_type_id defaults to 'any-to-any' when not in config."""
        config = {"model": {"id": "test"}}
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["model_type_id"] == "any-to-any"

    def test_default_model_type_id_not_applied_when_in_config(self):
        """Existing model_type_id is preserved."""
        config = {"model": {"id": "test", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["model_type_id"] == "text-to-text"

    def test_expand_compute_instance(self):
        """compute.instance expands to inference_compute_info with wildcard accelerator_type."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "compute": {"instance": "A10G"},
        }
        result = ModelBuilder.normalize_config(config)
        assert "inference_compute_info" in result
        assert result["inference_compute_info"]["num_accelerators"] == 1
        # Should use wildcard so model can be scheduled on any NVIDIA GPU
        assert result["inference_compute_info"]["accelerator_type"] == ["NVIDIA-*"]
        assert result["compute"]["instance"] == "A10G"  # compute key preserved for reference

    def test_expand_compute_gpu_legacy(self):
        """Legacy compute.gpu still works and gets normalized to compute.instance."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "compute": {"gpu": "A10G"},
        }
        result = ModelBuilder.normalize_config(config)
        assert "inference_compute_info" in result
        assert result["inference_compute_info"]["num_accelerators"] == 1
        assert result["compute"]["instance"] == "A10G"
        assert "gpu" not in result["compute"]  # legacy key removed after normalization

    def test_inference_compute_info_wins_over_compute_instance(self):
        """If both compute.instance and inference_compute_info exist, inference_compute_info wins."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "compute": {"instance": "A10G"},
            "inference_compute_info": {
                "cpu_limit": "8",
                "num_accelerators": 2,
                "accelerator_type": ["NVIDIA-L40S"],
            },
        }
        result = ModelBuilder.normalize_config(config)
        # inference_compute_info should be unchanged (it existed already)
        assert result["inference_compute_info"]["num_accelerators"] == 2
        assert "NVIDIA-L40S" in result["inference_compute_info"]["accelerator_type"]

    def test_expand_simplified_checkpoints(self):
        """Simplified checkpoints get type and when defaults."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "checkpoints": {"repo_id": "meta-llama/Llama-3-8B"},
        }
        result = ModelBuilder.normalize_config(config)
        assert result["checkpoints"]["type"] == "huggingface"
        assert result["checkpoints"]["when"] == "runtime"

    def test_existing_checkpoints_preserved(self):
        """Existing checkpoints type and when are preserved."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "checkpoints": {
                "type": "custom",
                "repo_id": "some/model",
                "when": "build",
            },
        }
        result = ModelBuilder.normalize_config(config)
        assert result["checkpoints"]["type"] == "custom"
        assert result["checkpoints"]["when"] == "build"

    def test_build_info_defaults(self):
        """build_info defaults are added when missing."""
        config = {"model": {"id": "test", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config)
        assert result["build_info"] == {"python_version": "3.12"}

    def test_existing_build_info_preserved(self):
        """Existing build_info is preserved."""
        config = {
            "model": {"id": "test", "model_type_id": "text-to-text"},
            "build_info": {"python_version": "3.11"},
        }
        result = ModelBuilder.normalize_config(config)
        assert result["build_info"]["python_version"] == "3.11"

    def test_verbose_config_passthrough(self):
        """Verbose config (already has all fields) passes through unchanged."""
        config = {
            "model": {
                "id": "test",
                "user_id": "user1",
                "app_id": "app1",
                "model_type_id": "text-to-text",
            },
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 1,
                "accelerator_type": ["NVIDIA-A10G"],
                "accelerator_memory": "24Gi",
            },
            "build_info": {"python_version": "3.12"},
        }
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["user_id"] == "user1"
        assert result["inference_compute_info"]["num_accelerators"] == 1
        assert result["build_info"]["python_version"] == "3.12"


class TestSimplifyClonedConfig:
    """Test simplify_cloned_config utility."""

    def test_removes_placeholder_user_id(self):
        """Placeholder user_id/app_id values are removed."""
        from clarifai.utils.cli import simplify_cloned_config

        config = {
            "model": {
                "id": "test",
                "user_id": "user_id",
                "app_id": "app_id",
                "model_type_id": "text-to-text",
            },
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 1,
                "accelerator_type": ["NVIDIA-A10G"],
                "accelerator_memory": "24Gi",
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            tmp_path = f.name

        try:
            simplify_cloned_config(tmp_path)
            with open(tmp_path) as f:
                result = yaml.safe_load(f)
            assert "user_id" not in result["model"]
            assert "app_id" not in result["model"]
        finally:
            os.unlink(tmp_path)

    def test_converts_compute_info_to_instance_shorthand(self):
        """inference_compute_info matching A10G becomes compute.instance."""
        from clarifai.utils.cli import simplify_cloned_config

        config = {
            "model": {
                "id": "test",
                "model_type_id": "text-to-text",
            },
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 1,
                "accelerator_type": ["NVIDIA-A10G"],
                "accelerator_memory": "24Gi",
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            tmp_path = f.name

        try:
            simplify_cloned_config(tmp_path)
            with open(tmp_path) as f:
                result = yaml.safe_load(f)
            assert "inference_compute_info" not in result
            assert result["compute"]["instance"] == "A10G"
        finally:
            os.unlink(tmp_path)

    def test_updates_model_id_from_model_name(self):
        """model_name is used to set model.id."""
        from clarifai.utils.cli import simplify_cloned_config

        config = {
            "model": {
                "id": "old-name",
                "model_type_id": "text-to-text",
            },
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            tmp_path = f.name

        try:
            simplify_cloned_config(tmp_path, model_name="meta-llama/Llama-3-8B")
            with open(tmp_path) as f:
                result = yaml.safe_load(f)
            assert result["model"]["id"] == "Llama-3-8B"
        finally:
            os.unlink(tmp_path)


class TestResolveUserId:
    """Test resolve_user_id() from config and API."""

    def test_resolve_from_config(self):
        """Resolves user_id from CLI config file."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        mock_config = MagicMock()
        mock_config.current.get.return_value = "config-user"
        mock_config.context_override = None
        mock_config.current_context = "test-context"

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.utils.config._get_user_id_from_pat', return_value="config-user"),
        ):
            user_id = resolve_user_id()
            assert user_id == "config-user"

    def test_resolve_falls_back_to_api(self):
        """Falls back to API when config has no user_id."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        # Mock config to have no user_id
        mock_config = MagicMock()
        mock_config.current.get.return_value = None

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.utils.config._get_user_id_from_pat', return_value="api-user"),
        ):
            user_id = resolve_user_id(pat="test-pat")
            assert user_id == "api-user"

    def test_resolve_returns_none_on_failure(self):
        """Returns None when both config and API fail."""
        from unittest.mock import patch

        from clarifai.utils.config import resolve_user_id

        with (
            patch('clarifai.utils.config.Config.from_yaml', side_effect=Exception("no config")),
            patch('clarifai.utils.config._get_user_id_from_pat', return_value=None),
        ):
            user_id = resolve_user_id()
            assert user_id is None

    def test_config_user_id_takes_priority_over_api(self):
        """Config file user_id is used when PAT matches."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        mock_config = MagicMock()
        mock_config.current.get.return_value = "config-user"
        mock_config.context_override = None
        mock_config.current_context = "test-context"

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.utils.config._get_user_id_from_pat', return_value="config-user"),
        ):
            user_id = resolve_user_id(pat="test-pat")
            assert user_id == "config-user"

    def test_pat_user_wins_on_mismatch(self):
        """When PAT user differs from config user, PAT user is used with a warning."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        mock_config = MagicMock()
        mock_config.current.get.return_value = "config-user"
        mock_config.context_override = None
        mock_config.current_context = "test-context"

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.utils.config._get_user_id_from_pat', return_value="pat-user"),
        ):
            user_id = resolve_user_id(pat="test-pat")
            assert user_id == "pat-user"


class TestModelDeployerValidation:
    """Test ModelDeployer input validation."""

    def test_no_source_raises(self):
        """No model source raises UserError."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer()
        with pytest.raises(Exception, match="You must specify either MODEL_PATH"):
            deployer.deploy()

    def test_multiple_sources_raises(self):
        """Multiple model sources raises UserError."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(
            model_path="/tmp/model", model_url="https://clarifai.com/u/a/models/m"
        )
        with pytest.raises(Exception, match="Specify only one of"):
            deployer.deploy()

    def test_existing_model_without_gpu_raises(self):
        """Deploying existing model without GPU raises UserError."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(model_url="https://clarifai.com/user1/app1/models/my-model")
        with pytest.raises(Exception, match="You must specify --instance"):
            deployer.deploy()


class TestInstanceOverride:
    """Test that --instance flag properly overrides inference_compute_info."""

    def test_instance_flag_overrides_config(self):
        """--instance l40s should override inference_compute_info even if config had a10g."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer.__new__(ModelDeployer)
        deployer.instance_type = "gpu-nvidia-l40s"
        deployer.pat = None
        deployer.base_url = None

        # Simulate builder with A10G inference_compute_info (set by normalize_config)
        mock_builder = MagicMock()
        mock_builder.config = {
            "inference_compute_info": {
                "cpu_limit": "4",
                "cpu_memory": "16Gi",
                "num_accelerators": 1,
                "accelerator_type": ["NVIDIA-A10G"],
                "accelerator_memory": "24Gi",
            }
        }
        mock_builder.inference_compute_info = MagicMock()  # non-None (already set)
        deployer._builder = mock_builder

        # Mock get_inference_compute_for_gpu to return L40S info
        l40s_ici = {
            "cpu_limit": "8",
            "cpu_memory": "32Gi",
            "num_accelerators": 1,
            "accelerator_type": ["NVIDIA-L40S"],
            "accelerator_memory": "48Gi",
        }
        with patch(
            "clarifai.utils.compute_presets.get_inference_compute_for_gpu",
            return_value=l40s_ici,
        ):
            from clarifai.utils.compute_presets import get_inference_compute_for_gpu

            if deployer.instance_type:
                ici = get_inference_compute_for_gpu(
                    deployer.instance_type, pat=deployer.pat, base_url=deployer.base_url
                )
                if ici.get('num_accelerators', 0) > 0:
                    ici.setdefault('accelerator_type', ['NVIDIA-*'])
                deployer._builder.config['inference_compute_info'] = ici
                deployer._builder.inference_compute_info = (
                    deployer._builder._get_inference_compute_info()
                )

        # Verify inference_compute_info was updated to L40S
        updated_ici = deployer._builder.config['inference_compute_info']
        assert updated_ici['accelerator_memory'] == '48Gi'
        assert updated_ici['cpu_limit'] == '8'

    def test_no_instance_flag_keeps_config(self):
        """Without --instance, inference_compute_info from config is preserved."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer.__new__(ModelDeployer)
        deployer.instance_type = None  # No --instance flag
        deployer.pat = None
        deployer.base_url = None

        # Simulate builder with A10G inference_compute_info
        from unittest.mock import MagicMock

        mock_builder = MagicMock()
        a10g_ici = {
            "cpu_limit": "4",
            "cpu_memory": "16Gi",
            "num_accelerators": 1,
            "accelerator_type": ["NVIDIA-A10G"],
            "accelerator_memory": "24Gi",
        }
        mock_builder.config = {"inference_compute_info": dict(a10g_ici)}
        deployer._builder = mock_builder

        # The override block should NOT execute
        if deployer.instance_type:
            assert False, "Should not reach here"

        # inference_compute_info unchanged
        assert deployer._builder.config['inference_compute_info'] == a10g_ici


class TestDeploymentMonitoring:
    """Test deployment monitoring logic."""

    def test_fetch_runner_logs_deduplicates(self):
        """Runner log fetching deduplicates by (log_type, url/message).

        _fetch_runner_logs only fetches "runner.events" (k8s events).
        Model stdout/stderr ("runner" logs) are reserved for _tail_runner_logs.
        """
        from unittest.mock import MagicMock

        from clarifai_grpc.grpc.api import resources_pb2

        from clarifai.runners.models.model_deploy import ModelDeployer

        # Mock stub and response with real-ish log entries
        mock_stub = MagicMock()
        mock_entry1 = resources_pb2.LogEntry(
            url="http://log1", message="Pod scheduled on node abc"
        )
        mock_entry2 = resources_pb2.LogEntry(
            url="http://log2", message="Pulling image clarifai/runner:latest"
        )

        mock_response = MagicMock()
        mock_response.log_entries = [mock_entry1, mock_entry2]
        mock_stub.ListLogEntries.return_value = mock_response

        user_app_id = resources_pb2.UserAppIDSet(user_id="test-user")
        seen_logs = set()

        # First call - should return log lines
        page, lines = ModelDeployer._fetch_runner_logs(
            mock_stub,
            user_app_id,
            "cc-id",
            "np-id",
            "runner-1",
            seen_logs,
            1,
        )

        # Only "runner.events" is fetched (not "runner" — that's for Startup Logs)
        assert mock_stub.ListLogEntries.call_count == 1
        # 2 entries from runner.events
        assert len(seen_logs) == 2
        assert len(lines) > 0

        # Second call with same logs - should not add new entries
        prev_seen = len(seen_logs)
        mock_stub.ListLogEntries.reset_mock()
        page, lines = ModelDeployer._fetch_runner_logs(
            mock_stub,
            user_app_id,
            "cc-id",
            "np-id",
            "runner-1",
            seen_logs,
            1,
        )
        # No new logs should be added
        assert len(seen_logs) == prev_seen
        assert len(lines) == 0

    def test_fetch_runner_logs_handles_errors(self):
        """Log fetching is best-effort and doesn't raise on errors."""
        from unittest.mock import MagicMock

        from clarifai_grpc.grpc.api import resources_pb2

        from clarifai.runners.models.model_deploy import ModelDeployer

        mock_stub = MagicMock()
        mock_stub.ListLogEntries.side_effect = Exception("API unavailable")
        user_app_id = resources_pb2.UserAppIDSet(user_id="test-user")

        # Should not raise
        page, lines = ModelDeployer._fetch_runner_logs(
            mock_stub,
            user_app_id,
            "cc-id",
            "np-id",
            "runner-1",
            set(),
            1,
        )
        assert page == 1  # Page unchanged on error
        assert len(lines) == 0

    def test_format_event_logs_parses_events(self):
        """Event log parser extracts reason and message from raw events."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        raw = (
            "Name: runner-pod-xyz.abc123, Type: Warning, Source: {karpenter }, "
            "Reason: FailedScheduling, FirstTimestamp: 2026-02-16 15:49:06 +0000 UTC, "
            "LastTimestamp: 2026-02-16 15:49:06 +0000 UTC, "
            'Message: Failed to schedule pod, incompatible requirements'
        )
        # verbose=True preserves the original reason
        lines = _format_event_logs(raw, verbose=True)
        assert len(lines) == 1
        assert "FailedScheduling" in lines[0]
        assert "Failed to schedule pod" in lines[0]
        # Should NOT contain raw pod name or timestamps
        assert "runner-pod-xyz" not in lines[0]
        assert "FirstTimestamp" not in lines[0]

    def test_format_event_logs_multi_events(self):
        """Multiple events separated by newlines are returned as separate lines."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        raw = (
            "Name: pod-1.abc, Type: Warning, Source: {}, Reason: FailedScheduling, "
            "FirstTimestamp: 2026-01-01 00:00:00, LastTimestamp: 2026-01-01 00:00:00, "
            "Message: No nodes available\n"
            "Name: pod-1.def, Type: Normal, Source: {autoscaler}, Reason: ScaleUp, "
            "FirstTimestamp: 2026-01-01 00:01:00, LastTimestamp: 2026-01-01 00:01:00, "
            "Message: Scaling up node group"
        )
        lines = _format_event_logs(raw, verbose=True)
        assert len(lines) == 2
        assert "FailedScheduling" in lines[0]
        assert "ScaleUp" in lines[1]

    def test_format_event_logs_non_verbose_simplifies(self):
        """Non-verbose mode simplifies FailedScheduling messages."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        raw = (
            "Name: pod-1.abc, Type: Warning, Source: {karpenter }, "
            "Reason: FailedScheduling, FirstTimestamp: 2026-02-16 15:49:06, "
            "LastTimestamp: 2026-02-16 15:49:06, "
            "Message: 0/5 nodes are available: 3 had untolerated taint "
            "{infra.clarifai.com/karpenter: }, 2 didn't match Pod topology"
        )
        lines = _format_event_logs(raw, verbose=False)
        assert len(lines) == 1
        assert "Scheduling" in lines[0]
        assert "Waiting for node" in lines[0]
        # Should NOT contain taint details
        assert "untolerated taint" not in lines[0]

    def test_format_event_logs_non_verbose_skips_noise(self):
        """Non-verbose mode skips TaintManagerEviction and other noise events."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        raw = (
            "Name: pod-1.abc, Type: Normal, Source: {scheduler}, "
            "Reason: TaintManagerEviction, FirstTimestamp: 2026-02-16 15:49:06, "
            "LastTimestamp: 2026-02-16 15:49:06, "
            "Message: Taint manager evicted the pod"
        )
        lines = _format_event_logs(raw, verbose=False)
        assert len(lines) == 0

        # Same event in verbose mode should appear
        lines_verbose = _format_event_logs(raw, verbose=True)
        assert len(lines_verbose) == 1

    def test_monitor_constants(self):
        """Monitoring constants are set to reasonable values."""
        from clarifai.runners.models.model_deploy import (
            DEFAULT_LOG_TAIL_DURATION,
            DEFAULT_MONITOR_TIMEOUT,
            DEFAULT_POLL_INTERVAL,
        )

        assert DEFAULT_MONITOR_TIMEOUT == 1200  # 20 minutes
        assert DEFAULT_POLL_INTERVAL == 5  # 5 seconds
        assert DEFAULT_LOG_TAIL_DURATION == 15  # 15 seconds quick check after ready


class TestParseK8sQuantity:
    """Test parse_k8s_quantity helper."""

    def test_gibibytes(self):
        assert parse_k8s_quantity("24Gi") == 24 * 1024**3
        assert parse_k8s_quantity("48Gi") == 48 * 1024**3

    def test_mebibytes(self):
        assert parse_k8s_quantity("1500Mi") == 1500 * 1024**2

    def test_gigabytes(self):
        assert parse_k8s_quantity("16G") == 16e9

    def test_plain_number(self):
        assert parse_k8s_quantity("4") == 4.0
        assert parse_k8s_quantity("4.5") == 4.5

    def test_millicores(self):
        assert parse_k8s_quantity("100m") == 0.1
        assert parse_k8s_quantity("500m") == 0.5

    def test_none_and_empty(self):
        assert parse_k8s_quantity(None) == 0
        assert parse_k8s_quantity("") == 0

    def test_numeric_input(self):
        assert parse_k8s_quantity(24) == 24.0
        assert parse_k8s_quantity(4.5) == 4.5

    def test_tebibytes(self):
        assert parse_k8s_quantity("1Ti") == 1024**4

    def test_kibibytes(self):
        assert parse_k8s_quantity("512Ki") == 512 * 1024


class TestAutoComputeUpdate:
    """Test automatic compute info update logic."""

    def _make_compute_info_proto(
        self, num_accelerators=1, accelerator_memory="24Gi", accelerator_type=None
    ):
        """Create a mock ComputeInfo proto."""
        from unittest.mock import MagicMock

        ci = MagicMock()
        ci.num_accelerators = num_accelerators
        ci.accelerator_memory = accelerator_memory
        ci.accelerator_type = accelerator_type or ["NVIDIA-*"]
        ci.ByteSize.return_value = 1  # Non-empty
        return ci

    def test_a10g_to_l40s_needs_update(self):
        """A10G model → L40S instance: instance exceeds spec, needs update."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        model_ci = self._make_compute_info_proto(num_accelerators=1, accelerator_memory="24Gi")
        instance_ci = FALLBACK_GPU_PRESETS["L40S"]["inference_compute_info"]

        needs_update, reasons = ModelDeployer._needs_compute_update(model_ci, instance_ci)
        assert needs_update is True
        assert any("accelerator_memory" in r for r in reasons)

    def test_l40s_to_a10g_no_update(self):
        """L40S model → A10G instance: instance is below spec, no update needed."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        model_ci = self._make_compute_info_proto(num_accelerators=1, accelerator_memory="48Gi")
        instance_ci = FALLBACK_GPU_PRESETS["A10G"]["inference_compute_info"]

        needs_update, reasons = ModelDeployer._needs_compute_update(model_ci, instance_ci)
        assert needs_update is False
        assert len(reasons) == 0

    def test_same_instance_no_update(self):
        """A10G model → A10G instance: same spec, no update needed."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        model_ci = self._make_compute_info_proto(num_accelerators=1, accelerator_memory="24Gi")
        instance_ci = FALLBACK_GPU_PRESETS["A10G"]["inference_compute_info"]

        needs_update, reasons = ModelDeployer._needs_compute_update(model_ci, instance_ci)
        assert needs_update is False

    def test_no_compute_info_needs_update(self):
        """Model with no compute info → any instance needs update."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        instance_ci = FALLBACK_GPU_PRESETS["A10G"]["inference_compute_info"]

        needs_update, reasons = ModelDeployer._needs_compute_update(None, instance_ci)
        assert needs_update is True
        assert any("no inference_compute_info" in r for r in reasons)

    def test_num_accelerators_triggers_update(self):
        """1-GPU model → 2-GPU instance: needs update."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        model_ci = self._make_compute_info_proto(num_accelerators=1, accelerator_memory="48Gi")
        instance_ci = FALLBACK_GPU_PRESETS["G6E"]["inference_compute_info"]

        needs_update, reasons = ModelDeployer._needs_compute_update(model_ci, instance_ci)
        assert needs_update is True
        assert any("num_accelerators" in r for r in reasons)

    def test_auto_update_patches_when_needed(self):
        """_auto_update_compute_if_needed patches model when instance exceeds spec."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(
            model_url="https://clarifai.com/user1/app1/models/my-model",
            instance_type="L40S",
        )
        deployer.model_version_id = "version-123"

        mock_model = MagicMock()

        # Model has A10G compute info (24Gi) with specific accelerator_type
        model_ci = self._make_compute_info_proto(
            num_accelerators=1, accelerator_memory="24Gi", accelerator_type=["NVIDIA-*"]
        )

        with (
            patch.object(deployer, '_resolve_gpu') as mock_resolve,
            patch.object(deployer, '_get_model_version_compute_info', return_value=model_ci),
        ):
            mock_resolve.return_value = dict(FALLBACK_GPU_PRESETS["L40S"])
            deployer._auto_update_compute_if_needed(mock_model)

            # Should have patched the model version
            mock_model.patch_version.assert_called_once()
            call_kwargs = mock_model.patch_version.call_args
            assert call_kwargs.kwargs["version_id"] == "version-123"
            # num_accelerators and accelerator_memory updated, accelerator_type preserved
            patched_ci = call_kwargs.kwargs["inference_compute_info"]
            assert patched_ci.accelerator_memory == "48Gi"
            assert patched_ci.num_accelerators == 1
            # accelerator_type should be preserved from the model version (not changed)
            assert list(patched_ci.accelerator_type) == ["NVIDIA-*"]

    def test_auto_update_skips_when_compatible(self):
        """_auto_update_compute_if_needed skips patch when instance is within spec."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(
            model_url="https://clarifai.com/user1/app1/models/my-model",
            instance_type="A10G",
        )
        deployer.model_version_id = "version-123"

        mock_model = MagicMock()

        # Model has L40S compute info (48Gi), deploying to A10G (24Gi) → compatible
        model_ci = self._make_compute_info_proto(num_accelerators=1, accelerator_memory="48Gi")

        with (
            patch.object(deployer, '_resolve_gpu') as mock_resolve,
            patch.object(deployer, '_get_model_version_compute_info', return_value=model_ci),
        ):
            mock_resolve.return_value = dict(FALLBACK_GPU_PRESETS["A10G"])
            deployer._auto_update_compute_if_needed(mock_model)

            # Should NOT have patched
            mock_model.patch_version.assert_not_called()

    def test_auto_update_skips_without_gpu_preset(self):
        """_auto_update_compute_if_needed is a no-op when GPU preset can't be resolved."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(
            model_url="https://clarifai.com/user1/app1/models/my-model",
            instance_type="A10G",
        )
        deployer.model_version_id = "version-123"

        mock_model = MagicMock()

        with patch.object(deployer, '_resolve_gpu', return_value=None):
            deployer._auto_update_compute_if_needed(mock_model)
            mock_model.patch_version.assert_not_called()


class TestStreamModelLogs:
    """Test standalone log streaming function."""

    def test_stream_logs_requires_model_info(self):
        """stream_model_logs raises UserError without model info."""
        from clarifai.runners.models.model_deploy import stream_model_logs

        with pytest.raises(Exception, match="You must specify --model-url"):
            stream_model_logs()

    def test_stream_logs_parses_model_url(self):
        """stream_model_logs extracts user/app/model from URL."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_deploy import stream_model_logs

        # Mock the Model client and gRPC stub
        mock_version = MagicMock()
        mock_version.model_version.id = "ver-123"

        mock_stub = MagicMock()
        # ListRunners returns no runners → should raise UserError
        mock_resp = MagicMock()
        mock_resp.runners = []
        mock_stub.ListRunners.return_value = mock_resp

        with (
            patch('clarifai.client.auth.create_stub', return_value=mock_stub),
            patch('clarifai.client.model.Model.__init__', return_value=None),
            patch('clarifai.client.model.Model.list_versions', return_value=[mock_version]),
        ):
            with pytest.raises(Exception, match="No active runner found"):
                stream_model_logs(
                    model_url="https://clarifai.com/user1/app1/models/my-model",
                    pat="test-pat",
                )

    def test_stream_logs_no_follow(self, capsys):
        """stream_model_logs with follow=False prints existing logs and exits."""
        from unittest.mock import MagicMock, patch

        from clarifai_grpc.grpc.api import resources_pb2

        from clarifai.runners.models.model_deploy import stream_model_logs

        # Mock version lookup
        mock_version = MagicMock()
        mock_version.model_version.id = "ver-123"

        # Mock runner
        mock_runner = MagicMock()
        mock_runner.id = "runner-1"
        mock_runner.nodepool.compute_cluster.id = "cc-1"
        mock_runner.nodepool.id = "np-1"
        mock_runners_resp = MagicMock()
        mock_runners_resp.runners = [mock_runner]

        # Mock log entries
        mock_log_entry = resources_pb2.LogEntry(message="Model loaded successfully!")
        mock_log_resp = MagicMock()
        mock_log_resp.log_entries = [mock_log_entry]

        mock_stub = MagicMock()
        mock_stub.ListRunners.return_value = mock_runners_resp
        mock_stub.ListLogEntries.return_value = mock_log_resp

        with (
            patch('clarifai.client.auth.create_stub', return_value=mock_stub),
            patch('clarifai.client.model.Model.__init__', return_value=None),
            patch('clarifai.client.model.Model.list_versions', return_value=[mock_version]),
        ):
            stream_model_logs(
                model_url="https://clarifai.com/user1/app1/models/my-model",
                pat="test-pat",
                follow=False,
            )

        captured = capsys.readouterr()
        assert "Model loaded successfully!" in captured.out
        assert "runner-1" in captured.out


class TestConfigTemplate:
    """Test config template generation."""

    def test_simplified_template(self):
        """Simplified template has no TODOs."""
        from clarifai.cli.templates.model_templates import get_config_template

        template = get_config_template(simplified=True, model_id="test-model")
        assert "TODO" not in template
        assert "compute:" in template
        assert "instance:" in template
        assert "test-model" in template
        # Should NOT have user_id/app_id
        assert "user_id" not in template
        assert "app_id" not in template

    def test_verbose_template(self):
        """Verbose template has full config fields."""
        from clarifai.cli.templates.model_templates import get_config_template

        template = get_config_template(simplified=False, user_id="test-user")
        assert "test-user" in template
        assert "inference_compute_info" in template


class TestCustomImageDockerfile:
    """Test build_info.image custom base Docker image support."""

    def test_custom_image_dockerfile_generated(self):
        """build_info.image triggers custom image Dockerfile template."""
        import shutil
        from pathlib import Path

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["build_info"] = {"image": "nvcr.io/nvidia/pytorch:24.01-py3"}

            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            builder = ModelBuilder(str(target), validate_api_ids=False)
            content = builder._generate_dockerfile_content()

            assert "nvcr.io/nvidia/pytorch:24.01-py3" in content
            assert "FROM --platform=$TARGETPLATFORM nvcr.io/nvidia/pytorch:24.01-py3" in content
            assert 'pip' in content
            # Should NOT contain multi-stage build FROM (no second FROM)
            from_lines = [l for l in content.splitlines() if l.strip().startswith("FROM")]
            assert len(from_lines) == 1

    def test_custom_image_dockerfile_has_required_sections(self):
        """Custom image Dockerfile includes requirements install, config copy, entrypoint."""
        import shutil
        from pathlib import Path

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["build_info"] = {"image": "python:3.12-slim"}

            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            builder = ModelBuilder(str(target), validate_api_ids=False)
            content = builder._generate_dockerfile_content()

            assert "requirements.txt" in content
            assert "config.yaml" in content
            assert "ENTRYPOINT" in content
            assert "clarifai.runners.server" in content
            assert "WORKDIR /home/nonroot/main" in content

    def test_no_custom_image_uses_standard_dockerfile(self):
        """Without build_info.image, standard Dockerfile is generated."""
        import shutil
        from pathlib import Path

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model"
            shutil.copytree(original_dummy_path, target)

            builder = ModelBuilder(str(target), validate_api_ids=False)
            content = builder._generate_dockerfile_content()

            # Standard Dockerfile uses uv and multi-stage build
            assert "uv" in content.lower() or "pip" in content.lower()
            # Should NOT reference a custom image like nvcr.io
            assert "nvcr.io" not in content

    def test_empty_image_uses_standard_dockerfile(self):
        """Empty string build_info.image falls back to standard Dockerfile."""
        import shutil
        from pathlib import Path

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["build_info"] = {"image": ""}

            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            builder = ModelBuilder(str(target), validate_api_ids=False)
            content = builder._generate_dockerfile_content()

            # Should NOT trigger custom image path
            assert "nvcr.io" not in content


class TestParseRunnerLog:
    """Test _parse_runner_log() JSON log parsing and filtering."""

    def test_json_log_extracts_msg(self):
        """JSON runner log extracts the 'msg' field."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = '{"msg": "Starting MCP bridge...", "@timestamp": "2026-02-18T13:15:15Z", "stack_info": null}'
        assert _parse_runner_log(raw) == "Starting MCP bridge..."

    def test_json_log_empty_msg_returns_none(self):
        """JSON runner log with empty msg returns None."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = '{"msg": "", "@timestamp": "2026-02-18T13:15:15Z"}'
        assert _parse_runner_log(raw) is None

    def test_json_log_no_msg_field_passthrough(self):
        """JSON object without 'msg' field passes through as raw string."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = '{"level": "info", "@timestamp": "2026-02-18T13:15:15Z"}'
        assert _parse_runner_log(raw) == raw

    def test_plain_text_passthrough(self):
        """Non-JSON text passes through unchanged."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = "[02/18/26 13:15:22] INFO Starting server on port 8080"
        assert _parse_runner_log(raw) == raw

    def test_deprecation_warning_filtered(self):
        """DeprecationWarning lines are filtered in non-verbose mode."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = "/usr/local/lib/python3.12/site-packages/foo.py:42: DeprecationWarning: datetime.utcnow() is deprecated"
        assert _parse_runner_log(raw, verbose=False) is None
        # Verbose mode keeps it
        assert _parse_runner_log(raw, verbose=True) == raw

    def test_pip_download_filtered(self):
        """pip download lines are filtered in non-verbose mode."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = "Downloading pygments (1.2MiB)"
        assert _parse_runner_log(raw, verbose=False) is None
        assert _parse_runner_log(raw, verbose=True) == raw

    def test_empty_and_none(self):
        """Empty string and None return None."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        assert _parse_runner_log("") is None
        assert _parse_runner_log(None) is None

    def test_installing_packages_filtered(self):
        """'Installing collected packages:' lines are filtered in non-verbose mode."""
        from clarifai.runners.models.model_deploy import _parse_runner_log

        raw = "Installing collected packages: numpy, pandas, torch"
        assert _parse_runner_log(raw, verbose=False) is None


class TestSimplifyK8sMessage:
    """Test _simplify_k8s_message() human-friendly event mapping."""

    def test_failed_scheduling_simplified(self):
        """FailedScheduling becomes a simple 'waiting' message."""
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        msg = _simplify_k8s_message(
            "FailedScheduling",
            "0/5 nodes are available: 3 had untolerated taint {infra.clarifai.com/karpenter: }",
        )
        assert msg == "Waiting for node to become available..."

    def test_scheduled_simplified(self):
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        msg = _simplify_k8s_message(
            "Scheduled", "Successfully assigned to ip-10-7-1-42.ec2.internal"
        )
        assert msg == "Pod scheduled on node"

    def test_pulling_simplified(self):
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        msg = _simplify_k8s_message(
            "Pulling", "Pulling image public.ecr.aws/clarifai/runner:sha-abc123"
        )
        assert msg == "Pulling model image..."

    def test_long_message_truncated(self):
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        long_msg = "x" * 100
        result = _simplify_k8s_message("UnknownReason", long_msg)
        assert len(result) == 80
        assert result.endswith("...")

    def test_short_message_passthrough(self):
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        msg = _simplify_k8s_message("UnknownReason", "Short message")
        assert msg == "Short message"

    def test_nominated_simplified(self):
        """Nominated/NominatedNode hides internal node IPs."""
        from clarifai.runners.models.model_deploy import _simplify_k8s_message

        msg = _simplify_k8s_message(
            "Nominated", "Pod should schedule on: node/ip-10-7-158-85.ec2.internal"
        )
        assert msg == "Node selected for scheduling"
        assert "ip-10" not in msg

        msg2 = _simplify_k8s_message(
            "NominatedNode", "Pod should schedule on: node/ip-10-7-158-85.ec2.internal"
        )
        assert msg2 == "Node selected for scheduling"


class TestEventDedup:
    """Test that simplified event messages are deduplicated across polls."""

    def test_seen_messages_deduplicates_events(self):
        """Repeated simplified events are suppressed when seen_messages is used."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        raw = (
            "Name: pod-1.abc, Type: Warning, Source: {karpenter }, "
            "Reason: FailedScheduling, FirstTimestamp: 2026-02-16 15:49:06, "
            "LastTimestamp: 2026-02-16 15:49:06, "
            "Message: 0/5 nodes are available"
        )
        raw2 = (
            "Name: pod-1.def, Type: Warning, Source: {karpenter }, "
            "Reason: FailedScheduling, FirstTimestamp: 2026-02-16 15:49:11, "
            "LastTimestamp: 2026-02-16 15:49:11, "
            "Message: 0/5 nodes are available (different timestamp)"
        )

        # Both simplify to the same message
        lines1 = _format_event_logs(raw, verbose=False)
        lines2 = _format_event_logs(raw2, verbose=False)
        assert len(lines1) == 1
        assert len(lines2) == 1
        # Both have the same simplified text
        assert lines1[0] == lines2[0]

    def test_event_prefix_alignment(self):
        """[warning] and [event  ] prefixes have consistent width."""
        from clarifai.runners.models.model_deploy import _format_event_logs

        warning_raw = (
            "Name: pod-1, Type: Warning, Source: {}, Reason: FailedScheduling, "
            "FirstTimestamp: 2026-01-01, LastTimestamp: 2026-01-01, "
            "Message: test"
        )
        normal_raw = (
            "Name: pod-1, Type: Normal, Source: {}, Reason: Scheduled, "
            "FirstTimestamp: 2026-01-01, LastTimestamp: 2026-01-01, "
            "Message: test"
        )
        warning_lines = _format_event_logs(warning_raw, verbose=True)
        normal_lines = _format_event_logs(normal_raw, verbose=True)
        assert len(warning_lines) == 1
        assert len(normal_lines) == 1
        # Both prefixes should align — same character position for the reason
        assert "[warning]" in warning_lines[0]
        assert "[event  ]" in normal_lines[0]


class TestDeployOutput:
    """Test deploy_output.py helper functions."""

    def test_phase_header_outputs(self, capsys):
        """phase_header prints a formatted header."""
        from clarifai.runners.models.deploy_output import phase_header

        phase_header("Validate")
        captured = capsys.readouterr()
        assert "Validate" in captured.out
        assert "\u2500" in captured.out  # em dash character

    def test_info_outputs(self, capsys):
        """info prints a labeled line."""
        from clarifai.runners.models.deploy_output import info

        info("Model", "my-model-id")
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "my-model-id" in captured.out

    def test_status_outputs(self, capsys):
        """status prints a status message."""
        from clarifai.runners.models.deploy_output import status

        status("Building image...")
        captured = capsys.readouterr()
        assert "Building image..." in captured.out

    def test_success_outputs(self, capsys):
        """success prints a green message."""
        from clarifai.runners.models.deploy_output import success

        success("Model deployed!")
        captured = capsys.readouterr()
        assert "Model deployed!" in captured.out

    def test_warning_outputs(self, capsys):
        """warning prints a yellow [warning] message."""
        from clarifai.runners.models.deploy_output import warning

        warning("Timeout reached")
        captured = capsys.readouterr()
        assert "[warning]" in captured.out
        assert "Timeout reached" in captured.out


class TestVerboseFlag:
    """Test that verbose flag is properly plumbed through ModelDeployer."""

    def test_deployer_accepts_verbose(self):
        """ModelDeployer accepts verbose parameter."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        deployer = ModelDeployer(verbose=True)
        assert deployer.verbose is True

        deployer2 = ModelDeployer()
        assert deployer2.verbose is False

    def test_fetch_runner_logs_passes_verbose(self):
        """_fetch_runner_logs accepts and passes verbose to formatters."""
        from unittest.mock import MagicMock

        from clarifai_grpc.grpc.api import resources_pb2

        from clarifai.runners.models.model_deploy import ModelDeployer

        mock_stub = MagicMock()
        mock_entry = resources_pb2.LogEntry(
            url="http://log1", message='{"msg": "Hello", "@timestamp": "2026-01-01"}'
        )
        mock_response = MagicMock()
        mock_response.log_entries = [mock_entry]
        mock_stub.ListLogEntries.return_value = mock_response

        user_app_id = resources_pb2.UserAppIDSet(user_id="test-user")

        # Should not raise with verbose=True
        page, lines = ModelDeployer._fetch_runner_logs(
            mock_stub, user_app_id, "cc", "np", "runner-1", set(), 1, verbose=True
        )
        assert page >= 1


class TestQuietSdkLogger:
    """Test the _quiet_sdk_logger context manager."""

    def test_suppresses_info_when_enabled(self):
        """Logger level is raised to WARNING inside the context."""
        import logging

        from clarifai.runners.models.model_deploy import _quiet_sdk_logger
        from clarifai.utils.logging import logger

        original_level = logger.level
        with _quiet_sdk_logger(suppress=True):
            assert logger.level >= logging.WARNING
        # Restored after exiting
        assert logger.level == original_level

    def test_noop_when_disabled(self):
        """Logger level is unchanged when suppress=False."""

        from clarifai.runners.models.model_deploy import _quiet_sdk_logger
        from clarifai.utils.logging import logger

        original_level = logger.level
        with _quiet_sdk_logger(suppress=False):
            assert logger.level == original_level

    def test_restores_on_exception(self):
        """Logger level is restored even if an exception is raised."""
        import logging

        from clarifai.runners.models.model_deploy import _quiet_sdk_logger
        from clarifai.utils.logging import logger

        original_level = logger.level
        with pytest.raises(ValueError):
            with _quiet_sdk_logger(suppress=True):
                assert logger.level >= logging.WARNING
                raise ValueError("test error")
        assert logger.level == original_level


class TestDeployModelQuiet:
    """Test the quiet parameter on deploy_model."""

    def test_deploy_model_quiet_suppresses_print(self, capsys):
        """deploy_model with quiet=True should not print success/failure messages."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_builder import deploy_model

        mock_nodepool = MagicMock()
        mock_deployment = MagicMock()
        mock_nodepool.create_deployment.return_value = mock_deployment

        with patch('clarifai.runners.models.model_builder.Nodepool', return_value=mock_nodepool):
            result = deploy_model(
                model_id="test-model",
                app_id="test-app",
                user_id="test-user",
                deployment_id="deploy-test",
                model_version_id="v1",
                nodepool_id="np-1",
                compute_cluster_id="cc-1",
                cluster_user_id="test-user",
                quiet=True,
            )

        assert result is True
        captured = capsys.readouterr()
        assert "✅" not in captured.out
        assert "Deployment" not in captured.out

    def test_deploy_model_not_quiet_prints(self, capsys):
        """deploy_model with quiet=False should print success message."""
        from unittest.mock import MagicMock, patch

        from clarifai.runners.models.model_builder import deploy_model

        mock_nodepool = MagicMock()
        mock_deployment = MagicMock()
        mock_nodepool.create_deployment.return_value = mock_deployment

        with patch('clarifai.runners.models.model_builder.Nodepool', return_value=mock_nodepool):
            result = deploy_model(
                model_id="test-model",
                app_id="test-app",
                user_id="test-user",
                deployment_id="deploy-test",
                model_version_id="v1",
                nodepool_id="np-1",
                compute_cluster_id="cc-1",
                cluster_user_id="test-user",
                quiet=False,
            )

        assert result is True
        captured = capsys.readouterr()
        assert "Deployment" in captured.out


class TestRecommendInstance:
    """Test auto-selection of GPU instance based on model size."""

    def testget_hf_model_info_success(self):
        """Parses safetensors.total and config from mocked HF API."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "safetensors": {
                "total": 7_000_000_000,
                "parameters": {"BF16": 7_000_000_000},
            },
            "config": {},
            "pipeline_tag": "text-generation",
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            info = get_hf_model_info("meta-llama/Llama-3-8B")
        assert info is not None
        assert info["num_params"] == 7_000_000_000
        assert info["dtype_breakdown"] == {"BF16": 7_000_000_000}
        assert info["pipeline_tag"] == "text-generation"
        assert info["quant_method"] is None

    def testget_hf_model_info_with_quantization(self):
        """Detects AWQ quantization from API response."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "safetensors": {"total": 7_000_000_000, "parameters": {"I32": 7_000_000_000}},
            "config": {"quantization_config": {"quant_method": "awq", "bits": 4}},
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            info = get_hf_model_info("some/model-awq")
        assert info["quant_method"] == "awq"
        assert info["quant_bits"] == 4

    def testget_hf_model_info_api_failure(self):
        """Returns None when API fails."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets.requests.get", side_effect=Exception("timeout")
        ):
            info = get_hf_model_info("nonexistent/model")
        assert info is None

    def testget_hf_model_info_no_safetensors(self):
        """Returns num_params=None when safetensors field missing."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"config": {}, "pipeline_tag": "text-generation"}
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            info = get_hf_model_info("some/old-model")
        assert info is not None
        assert info["num_params"] is None

    def test_detect_quant_awq(self):
        """-awq in repo name → ("awq", 4)"""
        method, bits = _detect_quant_from_repo_name("TheBloke/Llama-7B-AWQ")
        assert method == "awq"
        assert bits == 4

    def test_detect_quant_gptq(self):
        """-GPTQ in repo name → ("gptq", 4)"""
        method, bits = _detect_quant_from_repo_name("TheBloke/Llama-7B-GPTQ")
        assert method == "gptq"
        assert bits == 4

    def test_detect_quant_none(self):
        """Clean name → (None, None)"""
        method, bits = _detect_quant_from_repo_name("meta-llama/Llama-3-8B")
        assert method is None
        assert bits is None

    def test_estimate_vram_7b_bf16(self):
        """~7B * 2 + overhead (50% KV + hybrid overhead)"""
        vram = _estimate_vram_bytes(7_248_023_552)  # 7B params, BF16 default
        weight_bytes = 7_248_023_552 * 2.0
        # weights + 50% KV + overhead (2 GiB fixed + 10% of weights)
        expected_approx = weight_bytes * 1.50 + (2 * 1024**3 + weight_bytes * 0.10)
        assert abs(vram - expected_approx) < 1024  # within 1 KB

    def test_estimate_vram_7b_awq_4bit(self):
        """~7B * 0.5 + overhead (50% KV + hybrid overhead)"""
        vram = _estimate_vram_bytes(7_248_023_552, quant_method="awq", quant_bits=4)
        weight_bytes = 7_248_023_552 * 0.5
        expected_approx = weight_bytes * 1.50 + (2 * 1024**3 + weight_bytes * 0.10)
        assert abs(vram - expected_approx) < 1024

    def test_estimate_vram_70b_bf16(self):
        """~70B BF16 should be very large."""
        vram = _estimate_vram_bytes(70_000_000_000)
        vram_gib = vram / (1024**3)
        assert vram_gib > 150  # ~170 GiB, exceeds all instances

    def test_select_instance_small(self):
        """10 GiB → A10G (24 GiB) via fallback."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
        ):
            inst_id, reason = _select_instance_by_vram(10 * 1024**3)
        assert inst_id == "gpu-nvidia-a10g"
        assert "10.0 GiB" in reason

    def test_select_instance_medium(self):
        """30 GiB → L40S (48 GiB) via fallback."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
        ):
            inst_id, reason = _select_instance_by_vram(30 * 1024**3)
        assert inst_id == "gpu-nvidia-l40s"

    def test_select_instance_large(self):
        """60 GiB → G6E (96 GiB) via fallback."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
        ):
            inst_id, reason = _select_instance_by_vram(60 * 1024**3)
        assert inst_id == "gpu-nvidia-g6e-2x-large"

    def test_select_instance_too_large(self):
        """120 GiB → None via fallback."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
        ):
            inst_id, reason = _select_instance_by_vram(120 * 1024**3)
        assert inst_id is None
        assert "exceeds" in reason

    def test_recommend_mcp_model(self):
        """MCP → CPU instance."""
        config = {"model": {"model_type_id": "mcp"}}
        inst_id, reason = recommend_instance(config)
        assert inst_id == "t3a.2xlarge"
        assert "CPU" in reason

    def test_recommend_no_checkpoints(self):
        """No repo_id, no GPU toolkit → CPU."""
        config = {"model": {"model_type_id": "any-to-any"}}
        inst_id, reason = recommend_instance(config)
        assert inst_id == "t3a.2xlarge"
        assert "CPU" in reason or "cpu" in reason.lower()

    def test_recommend_vllm_no_repo(self):
        """vLLM without repo_id → None."""
        config = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
        }
        inst_id, reason = recommend_instance(config)
        assert inst_id is None
        assert "checkpoints.repo_id" in reason

    def test_recommend_7b_model(self):
        """Mock 7B BF16 → L40S (heuristic path with 90% utilization headroom)."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "meta-llama/Llama-3-8B"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "safetensors": {"total": 7_248_023_552, "parameters": {"BF16": 7_248_023_552}},
            "config": {},
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            with patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ):
                inst_id, reason = recommend_instance(config)
        # 7.2B * 2 * 1.5 + overhead = ~23.6 GiB > A10G usable (21.6 GiB) → L40S
        assert inst_id == "gpu-nvidia-l40s"

    def test_recommend_13b_model(self):
        """Mock 13B BF16 → L40S."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "meta-llama/Llama-13B"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "safetensors": {"total": 13_000_000_000, "parameters": {"BF16": 13_000_000_000}},
            "config": {},
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            with patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ):
                inst_id, reason = recommend_instance(config)
        # 13B * 2 * 1.5 + (2 GiB + 10%) ≈ 40.7 GiB → L40S (48 GiB)
        assert inst_id == "gpu-nvidia-l40s"

    def test_recommend_fallback_file_size(self):
        """HF metadata fails, file size works → selects by file size."""
        from unittest.mock import patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "some/model"},
        }
        with patch(
            "clarifai.utils.compute_presets.get_hf_model_info",
            return_value={
                "num_params": None,
                "quant_method": None,
                "quant_bits": None,
                "dtype_breakdown": None,
                "pipeline_tag": None,
            },
        ):
            with patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.get_huggingface_checkpoint_total_size",
                return_value=10 * 1024**3,  # 10 GiB files
            ):
                with patch(
                    "clarifai.utils.compute_presets._try_list_all_instance_types",
                    return_value=None,
                ):
                    inst_id, reason = recommend_instance(config)
        # 10 GiB * 1.3 + 2 GiB ≈ 15 GiB → A10G (24 GiB)
        assert inst_id == "gpu-nvidia-a10g"

    def test_recommend_both_fail(self):
        """Both APIs fail → (None, reason)."""
        from unittest.mock import patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "nonexistent/model"},
        }
        with patch("clarifai.utils.compute_presets.get_hf_model_info", return_value=None):
            with patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.get_huggingface_checkpoint_total_size",
                return_value=0,
            ):
                inst_id, reason = recommend_instance(config)
        assert inst_id is None
        assert "Could not determine" in reason

    def test_recommend_sglang_skips_pre_ampere(self):
        """SGLang toolkit should skip pre-Ampere instances across all clouds."""
        from unittest.mock import MagicMock, patch

        # AWS T4 instance
        mock_g4dn = MagicMock()
        mock_g4dn.id = "g4dn.xlarge"
        mock_g4dn.cloud_provider.id = "aws"
        mock_g4dn.compute_info.num_accelerators = 1
        mock_g4dn.compute_info.accelerator_memory = "16Gi"

        # Azure T4 instance (not in supported clouds — filtered out by recommendation)
        mock_azure_t4 = MagicMock()
        mock_azure_t4.id = "Standard_NC4as_T4_v3"
        mock_azure_t4.cloud_provider.id = "azure"
        mock_azure_t4.compute_info.num_accelerators = 1
        mock_azure_t4.compute_info.accelerator_memory = "16Gi"

        # AWS A10G instance (Ampere)
        mock_g5 = MagicMock()
        mock_g5.id = "g5.xlarge"
        mock_g5.cloud_provider.id = "aws"
        mock_g5.compute_info.num_accelerators = 1
        mock_g5.compute_info.accelerator_memory = "24Gi"

        config_sglang = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "lmsysorg/sglang:latest"},
            "checkpoints": {"repo_id": "Qwen/Qwen3-0.6B"},
        }
        config_vllm = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
            "checkpoints": {"repo_id": "Qwen/Qwen3-0.6B"},
        }

        small_vram = {"num_params": 600_000_000}  # ~3 GiB, fits T4

        with (
            patch("clarifai.utils.compute_presets.get_hf_model_info", return_value=small_vram),
            patch("clarifai.utils.compute_presets._get_hf_model_config", return_value=None),
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types",
                return_value=[mock_g4dn, mock_azure_t4, mock_g5],
            ),
        ):
            # SGLang should skip g4dn (pre-Ampere), Azure T4 filtered (unsupported cloud) → g5
            inst_id, _ = recommend_instance(config_sglang)
            assert inst_id == "g5.xlarge"

            # vLLM picks g4dn (AWS, cheapest); Azure T4 filtered out
            inst_id, _ = recommend_instance(config_vllm)
            assert inst_id == "g4dn.xlarge"

    def test_recommend_sglang_from_requirements_txt(self):
        """SGLang detected via requirements.txt should also skip pre-Ampere."""
        from unittest.mock import MagicMock, patch

        mock_g4dn = MagicMock()
        mock_g4dn.id = "g4dn.xlarge"
        mock_g4dn.cloud_provider.id = "aws"
        mock_g4dn.compute_info.num_accelerators = 1
        mock_g4dn.compute_info.accelerator_memory = "16Gi"

        mock_g5 = MagicMock()
        mock_g5.id = "g5.xlarge"
        mock_g5.cloud_provider.id = "aws"
        mock_g5.compute_info.num_accelerators = 1
        mock_g5.compute_info.accelerator_memory = "24Gi"

        # Config has no build_info.image — toolkit should be detected from requirements.txt
        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "Qwen/Qwen3-0.6B"},
        }

        small_vram = {"num_params": 600_000_000}

        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("sglang\nclarifai\n")

            with (
                patch("clarifai.utils.compute_presets.get_hf_model_info", return_value=small_vram),
                patch("clarifai.utils.compute_presets._get_hf_model_config", return_value=None),
                patch(
                    "clarifai.utils.compute_presets._try_list_all_instance_types",
                    return_value=[mock_g4dn, mock_g5],
                ),
            ):
                # Should detect sglang from requirements.txt and skip g4dn
                inst_id, _ = recommend_instance(config, model_path=tmpdir)
                assert inst_id == "g5.xlarge"

    def test_write_instance_to_config(self):
        """Verify config.yaml is updated with selected instance."""
        from clarifai.runners.models.model_deploy import ModelDeployer

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"model": {"id": "test"}}, f)

            deployer = ModelDeployer.__new__(ModelDeployer)
            deployer.model_path = tmpdir
            deployer._write_instance_to_config("gpu-nvidia-l40s")

            with open(config_path) as f:
                config = yaml.safe_load(f)
            assert config["compute"]["instance"] == "gpu-nvidia-l40s"


class TestHFTokenValidation:
    """Tests for HuggingFace token validation and error reporting."""

    def test_hf_gated_no_token(self):
        """Gated repo with no token returns (False, 'gated_no_token')."""
        from unittest.mock import patch

        with patch(
            "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access"
        ) as mock:
            # Simulate real behavior
            mock.return_value = (False, "gated_no_token")
            has_access, reason = mock("meta-llama/Llama-3.1-8B-Instruct", token=None)
            assert has_access is False
            assert reason == "gated_no_token"

    def _mock_hf_response(self, status_code=403):
        """Create a mock HTTP response for HF exceptions."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.status_code = status_code
        response.headers = {}
        return response

    def test_hf_gated_no_access(self):
        """Gated repo with token that lacks access returns (False, 'gated_no_access')."""
        from unittest.mock import patch

        from huggingface_hub.utils import GatedRepoError

        err = GatedRepoError("gated", response=self._mock_hf_response(403))
        with patch("huggingface_hub.auth_check", side_effect=err):
            from clarifai.runners.utils.loader import HuggingFaceLoader

            has_access, reason = HuggingFaceLoader.validate_hf_repo_access(
                "meta-llama/Llama-3.1-8B-Instruct", token="hf_fake_token"
            )
            assert has_access is False
            assert reason == "gated_no_access"

    def test_hf_gated_no_token_real(self):
        """Gated repo with no token triggers gated_no_token reason."""
        from unittest.mock import patch

        from huggingface_hub.utils import GatedRepoError

        err = GatedRepoError("gated", response=self._mock_hf_response(403))
        with patch("huggingface_hub.auth_check", side_effect=err):
            from clarifai.runners.utils.loader import HuggingFaceLoader

            has_access, reason = HuggingFaceLoader.validate_hf_repo_access(
                "meta-llama/Llama-3.1-8B-Instruct", token=None
            )
            assert has_access is False
            assert reason == "gated_no_token"

    def test_hf_not_found(self):
        """Non-existent repo returns (False, 'not_found')."""
        from unittest.mock import patch

        from huggingface_hub.utils import RepositoryNotFoundError

        err = RepositoryNotFoundError("not found", response=self._mock_hf_response(404))
        with patch("huggingface_hub.auth_check", side_effect=err):
            from clarifai.runners.utils.loader import HuggingFaceLoader

            has_access, reason = HuggingFaceLoader.validate_hf_repo_access(
                "fake-org/nonexistent-model", token=None
            )
            assert has_access is False
            assert reason == "not_found"

    def test_hf_success(self):
        """Valid repo returns (True, '')."""
        from unittest.mock import patch

        with patch("huggingface_hub.auth_check", return_value=None):
            from clarifai.runners.utils.loader import HuggingFaceLoader

            has_access, reason = HuggingFaceLoader.validate_hf_repo_access(
                "bert-base-uncased", token=None
            )
            assert has_access is True
            assert reason == ""

    def test_validate_config_gated_no_token_raises(self):
        """ModelBuilder raises UserError with 'Set HF_TOKEN' for gated repo without token."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        from clarifai.errors import UserError

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Anonymous check returns gated, no env token available either
            with (
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                    return_value=(False, "gated_no_token"),
                ),
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                    return_value=False,
                ),
            ):
                with pytest.raises(UserError, match="requires authentication"):
                    ModelBuilder(str(target), validate_api_ids=False)

    def test_validate_config_gated_no_access_raises(self):
        """ModelBuilder raises UserError with 'Request access' for gated repo with bad token."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        from clarifai.errors import UserError

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "hf_token": "hf_bad_token",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            # 1st call (anonymous) → gated, 2nd call (with token) → no access
            with (
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                    side_effect=[(False, "gated_no_token"), (False, "gated_no_access")],
                ),
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                    return_value=True,
                ),
            ):
                with pytest.raises(UserError, match="does not have access"):
                    ModelBuilder(str(target), validate_api_ids=False)

    def test_validate_config_not_found_raises(self):
        """ModelBuilder raises UserError with 'not found' for missing repo."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        from clarifai.errors import UserError

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "fake-org/nonexistent-model",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            with (
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                    return_value=(False, "not_found"),
                ),
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                    return_value=False,
                ),
            ):
                with pytest.raises(UserError, match="not found"):
                    ModelBuilder(str(target), validate_api_ids=False)

    def test_validate_config_env_token_persisted_for_runtime(self):
        """When when=runtime and HF_TOKEN only in env, token is validated and written to config."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            # 1st call (anonymous/False) → gated, 2nd call (with env token) → success
            mock_validate = patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                side_effect=[(False, "gated_no_token"), (True, "")],
            )
            mock_env = patch.dict(os.environ, {"HF_TOKEN": "hf_env_only_token"})
            mock_hftoken = patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                return_value=True,
            )

            with mock_validate as mv, mock_env, mock_hftoken:
                ModelBuilder(str(target), validate_api_ids=False)
                # First call anonymous (False), second with env token
                assert mv.call_count == 2
                assert mv.call_args_list[0].kwargs["token"] is False
                assert mv.call_args_list[1].kwargs["token"] == "hf_env_only_token"

            # Token should have been persisted to config.yaml
            with config_path.open("r") as f:
                saved = yaml.safe_load(f)
            assert saved["checkpoints"]["hf_token"] == "hf_env_only_token"

    def test_validate_config_env_token_no_access_raises(self):
        """When when=runtime, env token set but lacks access, raises UserError."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        from clarifai.errors import UserError

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            # 1st call (anonymous) → gated, 2nd call (with env token) → no access
            with (
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                    side_effect=[(False, "gated_no_token"), (False, "gated_no_access")],
                ),
                patch.dict(os.environ, {"HF_TOKEN": "hf_bad_env_token"}),
                patch(
                    "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                    return_value=True,
                ),
            ):
                with pytest.raises(UserError, match="does not have access"):
                    ModelBuilder(str(target), validate_api_ids=False)

    def test_validate_config_config_token_used_for_build_runtime(self):
        """When when=runtime and hf_token IS in config, validate with that token."""
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        tests_dir = Path(__file__).parent.resolve()
        original_dummy_path = tests_dir / "dummy_runner_models"

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "test_model"
            shutil.copytree(original_dummy_path, target)

            config_path = target / "config.yaml"
            with config_path.open("r") as f:
                config = yaml.safe_load(f)

            config["checkpoints"] = {
                "type": "huggingface",
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "hf_token": "hf_config_token",
                "when": "runtime",
            }
            with config_path.open("w") as f:
                yaml.dump(config, f, sort_keys=False)

            # 1st call (anonymous) → gated, 2nd call (with config token) → success
            mock_validate = patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hf_repo_access",
                side_effect=[(False, "gated_no_token"), (True, "")],
            )
            mock_hftoken = patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.validate_hftoken",
                return_value=True,
            )

            with mock_validate as mv, mock_hftoken:
                ModelBuilder(str(target), validate_api_ids=False)
                # 1st anonymous, 2nd with config token
                assert mv.call_count == 2
                assert mv.call_args_list[1].kwargs["token"] == "hf_config_token"


class TestKVCacheEstimation:
    """Tests for accurate KV cache estimation from HF config.json."""

    # Qwen3-4B architecture params (from actual config.json)
    QWEN3_4B_CONFIG = {
        'num_hidden_layers': 36,
        'num_key_value_heads': 8,
        'head_dim': 128,
        'max_position_embeddings': 40960,
    }

    # Llama-3.1-8B architecture params
    LLAMA_8B_CONFIG = {
        'num_hidden_layers': 32,
        'num_key_value_heads': 8,
        'head_dim': 128,
        'max_position_embeddings': 131072,
    }

    # Phi-3-mini-4k (small context)
    PHI3_MINI_CONFIG = {
        'num_hidden_layers': 32,
        'num_key_value_heads': 32,  # MHA (num_kv_heads == num_attention_heads)
        'head_dim': 96,
        'max_position_embeddings': 4096,
    }

    def test_kv_cache_qwen3_4b(self):
        """Qwen3-4B KV cache should be ~5.62 GiB (matches vLLM error)."""
        kv_bytes = _estimate_kv_cache_bytes(self.QWEN3_4B_CONFIG)
        kv_gib = kv_bytes / (1024**3)
        # 2 * 36 * 8 * 128 * 2 * 40960 = 6,039,797,760 bytes = ~5.625 GiB
        assert abs(kv_gib - 5.625) < 0.01

    def test_kv_cache_llama_8b(self):
        """Llama-3.1-8B with 128k context should have ~16 GiB KV cache."""
        kv_bytes = _estimate_kv_cache_bytes(self.LLAMA_8B_CONFIG)
        kv_gib = kv_bytes / (1024**3)
        # 2 * 32 * 8 * 128 * 2 * 131072 = 17,179,869,184 bytes = 16 GiB
        assert abs(kv_gib - 16.0) < 0.01

    def test_kv_cache_phi3_small_context(self):
        """Phi-3-mini with 4k context should have small KV cache."""
        kv_bytes = _estimate_kv_cache_bytes(self.PHI3_MINI_CONFIG)
        kv_gib = kv_bytes / (1024**3)
        # 2 * 32 * 32 * 96 * 2 * 4096 = 1,610,612,736 bytes = 1.5 GiB
        assert abs(kv_gib - 1.5) < 0.01

    def test_estimate_weight_bytes_bf16(self):
        """_estimate_weight_bytes returns just weights, no KV or overhead."""
        weight_bytes = _estimate_weight_bytes(4_000_000_000)  # 4B params, BF16
        # 4B * 2 = 8 GB
        assert weight_bytes == 4_000_000_000 * 2

    def test_estimate_weight_bytes_awq(self):
        """AWQ 4-bit quantization: 0.5 bytes per param."""
        weight_bytes = _estimate_weight_bytes(7_000_000_000, quant_method="awq", quant_bits=4)
        assert weight_bytes == 7_000_000_000 * 0.5

    def test_get_hf_model_config_qwen3(self):
        """Mock HF config.json for Qwen3-4B returns correct architecture."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'num_hidden_layers': 36,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 2560,
            'max_position_embeddings': 40960,
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            config = _get_hf_model_config("Qwen/Qwen3-4B")
        assert config == self.QWEN3_4B_CONFIG

    def test_get_hf_model_config_compute_head_dim(self):
        """When head_dim is not explicit, compute from hidden_size / num_attention_heads."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'hidden_size': 4096,
            'max_position_embeddings': 131072,
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            config = _get_hf_model_config("meta-llama/Llama-3.1-8B")
        assert config is not None
        assert config['head_dim'] == 128  # 4096 / 32
        assert config['num_key_value_heads'] == 8

    def test_get_hf_model_config_mha_fallback(self):
        """MHA model (no num_key_value_heads) falls back to num_attention_heads."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'hidden_size': 3072,
            'max_position_embeddings': 4096,
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            config = _get_hf_model_config("microsoft/phi-3-mini-4k-instruct")
        assert config is not None
        assert config['num_key_value_heads'] == 32  # falls back to num_attention_heads

    def test_get_hf_model_config_missing_max_pos(self):
        """Models without max_position_embeddings return None (safe fallback)."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'num_hidden_layers': 32,
            'num_attention_heads': 71,
            'hidden_size': 4544,
            # No max_position_embeddings
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp):
            config = _get_hf_model_config("tiiuae/falcon-7b")
        assert config is None

    def test_get_hf_model_config_network_failure(self):
        """Network failure returns None gracefully."""
        from unittest.mock import patch

        with patch(
            "clarifai.utils.compute_presets.requests.get",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ):
            config = _get_hf_model_config("any/model")
        assert config is None

    def test_get_hf_model_config_with_token(self):
        """HF token is passed as Bearer header for gated models."""
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'num_hidden_layers': 32,
            'num_attention_heads': 8,
            'hidden_size': 4096,
            'max_position_embeddings': 131072,
        }
        with patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp) as mock:
            _get_hf_model_config("meta-llama/Llama-3.1-8B", hf_token="hf_test_token")
            # Verify token was passed in headers
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs['headers']['Authorization'] == 'Bearer hf_test_token'

    def test_get_hf_token_from_config(self):
        """Token from config takes priority."""
        config = {'checkpoints': {'hf_token': 'hf_from_config'}}
        assert _get_hf_token(config) == 'hf_from_config'

    def test_get_hf_token_from_env(self):
        """Falls back to HF_TOKEN environment variable."""
        from unittest.mock import patch

        with patch.dict(os.environ, {'HF_TOKEN': 'hf_from_env'}):
            assert _get_hf_token({}) == 'hf_from_env'

    def test_get_hf_token_none(self):
        """Returns None when no token available."""
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            # Also ensure no cached token file
            with patch("builtins.open", side_effect=FileNotFoundError):
                assert _get_hf_token({}) is None

    def test_recommend_vllm_qwen3_4b_accurate_kv(self):
        """vLLM + Qwen3-4B: accurate KV cache → A10G instead of g4dn."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
            "checkpoints": {"repo_id": "Qwen/Qwen3-4B"},
        }
        # Mock HF model info (parameter count)
        mock_hf_info = MagicMock()
        mock_hf_info.json.return_value = {
            "safetensors": {"total": 4_020_000_000, "parameters": {"BF16": 4_020_000_000}},
            "config": {},
        }
        with (
            patch("clarifai.utils.compute_presets.requests.get") as mock_get,
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ),
        ):
            # First call: get_hf_model_info (HF API)
            # Second call: _get_hf_model_config (config.json)
            mock_config_resp = MagicMock()
            mock_config_resp.json.return_value = {
                'num_hidden_layers': 36,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'head_dim': 128,
                'hidden_size': 2560,
                'max_position_embeddings': 40960,
            }
            mock_get.side_effect = [mock_hf_info, mock_config_resp]

            inst_id, reason = recommend_instance(config)

        # 4.02B * 2 = 8.04 GiB weights + 5.625 GiB KV + 2 GiB overhead = ~15.7 GiB
        # g4dn (16 GiB) would be too tight → should pick A10G (24 GiB)
        assert inst_id == "gpu-nvidia-a10g"
        assert "KV cache" in reason
        assert "40960 ctx" in reason

    def test_recommend_vllm_short_context_unchanged(self):
        """vLLM + short context model: accurate KV cache is small, same result as heuristic."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
            "checkpoints": {"repo_id": "microsoft/phi-3-mini-4k-instruct"},
        }
        mock_hf_info = MagicMock()
        mock_hf_info.json.return_value = {
            "safetensors": {"total": 3_800_000_000, "parameters": {"BF16": 3_800_000_000}},
            "config": {},
        }
        mock_config_resp = MagicMock()
        mock_config_resp.json.return_value = {
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'hidden_size': 3072,
            'max_position_embeddings': 4096,
        }
        with (
            patch("clarifai.utils.compute_presets.requests.get") as mock_get,
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ),
        ):
            mock_get.side_effect = [mock_hf_info, mock_config_resp]
            inst_id, reason = recommend_instance(config)

        # 3.8B * 2 = 7.6 GiB weights + ~1.5 GiB KV (small ctx) + 2 GiB = ~11 GiB
        # Fits g4dn (16 GiB) - but g4dn is not in fallback tiers, so → A10G (24 GiB)
        assert inst_id == "gpu-nvidia-a10g"
        assert "KV cache" in reason

    def test_recommend_non_vllm_uses_heuristic(self):
        """Non-vLLM toolkit (huggingface) uses heuristic, not accurate KV cache."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "checkpoints": {"repo_id": "some/model"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "safetensors": {"total": 7_000_000_000, "parameters": {"BF16": 7_000_000_000}},
            "config": {},
        }
        with (
            patch("clarifai.utils.compute_presets.requests.get", return_value=mock_resp),
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ),
        ):
            inst_id, reason = recommend_instance(config)

        # Should NOT call _get_hf_model_config (no toolkit detected)
        # 7B * 2 * 1.5 + overhead ≈ 22.9 GiB > A10G usable (21.6 GiB) → L40S
        assert inst_id == "gpu-nvidia-l40s"
        assert "KV cache" not in reason  # heuristic path, no KV detail

    def test_recommend_vllm_config_unavailable_falls_back(self):
        """vLLM with unavailable config.json falls back to heuristic."""
        from unittest.mock import MagicMock, patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
            "checkpoints": {"repo_id": "private/gated-model"},
        }
        mock_hf_info = MagicMock()
        mock_hf_info.json.return_value = {
            "safetensors": {"total": 7_000_000_000, "parameters": {"BF16": 7_000_000_000}},
            "config": {},
        }
        with (
            patch("clarifai.utils.compute_presets.requests.get") as mock_get,
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ),
        ):
            # First call: get_hf_model_info succeeds
            # Second call: _get_hf_model_config fails (gated)
            mock_config_fail = MagicMock()
            mock_config_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("403")
            mock_get.side_effect = [mock_hf_info, mock_config_fail]

            inst_id, reason = recommend_instance(config)

        # Falls back to heuristic: 7B * 2 * 1.5 + overhead ≈ 22.9 GiB > A10G usable → L40S
        assert inst_id == "gpu-nvidia-l40s"
        assert "KV cache" not in reason  # heuristic, no KV detail

    def test_recommend_vllm_file_size_with_kv(self):
        """vLLM file-size fallback also uses accurate KV cache when available."""
        from unittest.mock import patch

        config = {
            "model": {"model_type_id": "any-to-any"},
            "build_info": {"image": "vllm/vllm-openai:latest"},
            "checkpoints": {"repo_id": "Qwen/Qwen3-4B"},
        }
        with (
            patch(
                "clarifai.utils.compute_presets.get_hf_model_info",
                return_value={
                    "num_params": None,
                    "quant_method": None,
                    "quant_bits": None,
                    "dtype_breakdown": None,
                    "pipeline_tag": None,
                },
            ),
            patch(
                "clarifai.utils.compute_presets._get_hf_model_config",
                return_value=self.QWEN3_4B_CONFIG,
            ),
            patch(
                "clarifai.runners.utils.loader.HuggingFaceLoader.get_huggingface_checkpoint_total_size",
                return_value=int(7.5 * 1024**3),  # 7.5 GiB file size
            ),
            patch(
                "clarifai.utils.compute_presets._try_list_all_instance_types", return_value=None
            ),
        ):
            inst_id, reason = recommend_instance(config)

        # 7.5 GiB files + 5.625 GiB KV + 2 GiB overhead = ~15.1 GiB → A10G
        assert inst_id == "gpu-nvidia-a10g"
        assert "KV cache" in reason
        assert "40960 ctx" in reason


class TestNodepoolInstanceTypeValidation:
    """Test validation of instance type against nodepool's available instance types."""

    def _make_deployer(self, **kwargs):
        from clarifai.runners.models.model_deploy import ModelDeployer

        defaults = dict(
            model_url="https://clarifai.com/user/app/models/test-model",
            user_id="test-user",
            instance_type="a10g",
            compute_cluster_id="cc-1",
            nodepool_id="np-1",
            pat="test-pat",
            base_url="https://api.clarifai.com",
        )
        defaults.update(kwargs)
        return ModelDeployer(**defaults)

    def _mock_nodepool(self, instance_type_ids):
        """Create a mock nodepool with given instance type IDs."""
        from unittest.mock import MagicMock

        np = MagicMock()
        instance_types = []
        for it_id in instance_type_ids:
            it = MagicMock()
            it.id = it_id
            instance_types.append(it)
        np.instance_types = instance_types
        return np

    def test_valid_instance_type_passes(self):
        """No error when instance type exists in nodepool."""
        from unittest.mock import MagicMock, patch

        deployer = self._make_deployer(instance_type="a10g")
        deployer._gpu_preset = {
            "instance_type_id": "gpu-nvidia-a10g",
            "inference_compute_info": {},
        }
        deployer._gpu_preset_key = (deployer.cloud_provider, deployer.region, deployer.num_gpus)

        mock_cc = MagicMock()
        mock_cc.nodepool.return_value = self._mock_nodepool(["gpu-nvidia-a10g", "gpu-nvidia-l40s"])

        with patch("clarifai.client.compute_cluster.ComputeCluster", return_value=mock_cc):
            # Should not raise
            deployer._validate_nodepool_instance_type("cc-1", "np-1")

    def test_invalid_instance_type_raises(self):
        """UserError when instance type is not in nodepool."""
        from unittest.mock import MagicMock, patch

        from clarifai.errors import UserError

        deployer = self._make_deployer(instance_type="a10g")
        deployer._gpu_preset = {
            "instance_type_id": "gpu-nvidia-a10g",
            "inference_compute_info": {},
        }
        deployer._gpu_preset_key = (deployer.cloud_provider, deployer.region, deployer.num_gpus)

        mock_cc = MagicMock()
        mock_cc.nodepool.return_value = self._mock_nodepool(["gpu-nvidia-l40s"])

        with patch("clarifai.client.compute_cluster.ComputeCluster", return_value=mock_cc):
            with pytest.raises(UserError, match="not available in nodepool"):
                deployer._validate_nodepool_instance_type("cc-1", "np-1")

    def test_error_message_lists_available_types(self):
        """Error message includes available instance types."""
        from unittest.mock import MagicMock, patch

        from clarifai.errors import UserError

        deployer = self._make_deployer(instance_type="a10g")
        deployer._gpu_preset = {
            "instance_type_id": "gpu-nvidia-a10g",
            "inference_compute_info": {},
        }
        deployer._gpu_preset_key = (deployer.cloud_provider, deployer.region, deployer.num_gpus)

        mock_cc = MagicMock()
        mock_cc.nodepool.return_value = self._mock_nodepool(
            ["gpu-nvidia-l40s", "gpu-nvidia-g6e-2x-large"]
        )

        with patch("clarifai.client.compute_cluster.ComputeCluster", return_value=mock_cc):
            with pytest.raises(UserError, match="gpu-nvidia-l40s"):
                deployer._validate_nodepool_instance_type("cc-1", "np-1")

    def test_nodepool_not_found_raises(self):
        """UserError when nodepool doesn't exist."""
        from unittest.mock import MagicMock, patch

        from clarifai.errors import UserError

        deployer = self._make_deployer(instance_type="a10g")
        deployer._gpu_preset = {
            "instance_type_id": "gpu-nvidia-a10g",
            "inference_compute_info": {},
        }
        deployer._gpu_preset_key = (deployer.cloud_provider, deployer.region, deployer.num_gpus)

        mock_cc = MagicMock()
        mock_cc.nodepool.side_effect = Exception("not found")

        with patch("clarifai.client.compute_cluster.ComputeCluster", return_value=mock_cc):
            with pytest.raises(UserError, match="not found in compute cluster"):
                deployer._validate_nodepool_instance_type("cc-1", "np-1")

    def test_no_instance_type_skips_validation(self):
        """When no instance type specified, validation is skipped."""
        deployer = self._make_deployer(instance_type=None)
        deployer._gpu_preset = None
        # Should not raise — no validation needed
        deployer._validate_nodepool_instance_type("cc-1", "np-1")


class TestAcceleratorWildcard:
    """Test accelerator wildcard detection for NVIDIA vs AMD instances."""

    def test_nvidia_instance_type_id(self):
        assert get_accelerator_wildcard(instance_type_id="gpu-nvidia-a10g") == "NVIDIA-*"

    def test_amd_instance_type_id(self):
        assert get_accelerator_wildcard(instance_type_id="gpu-amd-mi300x") == "AMD-*"

    def test_amd_mi250_instance_type_id(self):
        assert get_accelerator_wildcard(instance_type_id="gpu-amd-mi250") == "AMD-*"

    def test_amd_accelerator_types(self):
        assert get_accelerator_wildcard(accelerator_types=["AMD-MI300X"]) == "AMD-*"

    def test_nvidia_accelerator_types(self):
        assert get_accelerator_wildcard(accelerator_types=["NVIDIA-A10G"]) == "NVIDIA-*"

    def test_no_info_defaults_to_nvidia(self):
        assert get_accelerator_wildcard() == "NVIDIA-*"

    def test_accelerator_types_takes_precedence(self):
        """accelerator_types from API should take precedence over instance_type_id."""
        assert (
            get_accelerator_wildcard(
                instance_type_id="gpu-nvidia-a10g", accelerator_types=["AMD-MI300X"]
            )
            == "AMD-*"
        )
