"""Tests for model deploy, config normalization, and GPU presets."""

import os
import tempfile

import pytest
import yaml

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.compute_presets import (
    FALLBACK_GPU_PRESETS,
    get_compute_cluster_config,
    get_deploy_compute_cluster_id,
    get_deploy_nodepool_id,
    get_inference_compute_for_gpu,
    get_nodepool_config,
    infer_gpu_from_config,
    list_gpu_presets,
    parse_k8s_quantity,
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
        """Instance type IDs should be case-insensitive."""
        preset_lower = resolve_gpu("g5.xlarge")
        preset_upper = resolve_gpu("G5.XLARGE")
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
        """app_id defaults to 'app' when not in config and not provided."""
        config = {"model": {"id": "test", "user_id": "user1", "model_type_id": "text-to-text"}}
        result = ModelBuilder.normalize_config(config)
        assert result["model"]["app_id"] == "app"

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

        with patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config):
            user_id = resolve_user_id()
            assert user_id == "config-user"

    def test_resolve_falls_back_to_api(self):
        """Falls back to API when config has no user_id."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        # Mock config to have no user_id
        mock_config = MagicMock()
        mock_config.current.get.return_value = None

        # Mock User API call
        mock_user = MagicMock()
        mock_user.get_user_info.return_value.user.id = "api-user"

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.client.user.User', return_value=mock_user),
        ):
            user_id = resolve_user_id(pat="test-pat")
            assert user_id == "api-user"

    def test_resolve_returns_none_on_failure(self):
        """Returns None when both config and API fail."""
        from unittest.mock import patch

        from clarifai.utils.config import resolve_user_id

        with (
            patch('clarifai.utils.config.Config.from_yaml', side_effect=Exception("no config")),
            patch('clarifai.client.user.User', side_effect=Exception("no api")),
        ):
            user_id = resolve_user_id()
            assert user_id is None

    def test_config_user_id_takes_priority_over_api(self):
        """Config file user_id is used without making API call."""
        from unittest.mock import MagicMock, patch

        from clarifai.utils.config import resolve_user_id

        mock_config = MagicMock()
        mock_config.current.get.return_value = "config-user"

        mock_user_cls = MagicMock()

        with (
            patch('clarifai.utils.config.Config.from_yaml', return_value=mock_config),
            patch('clarifai.client.user.User', mock_user_cls),
        ):
            user_id = resolve_user_id(pat="test-pat")
            assert user_id == "config-user"
            # User class should NOT have been called
            mock_user_cls.assert_not_called()


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

        assert DEFAULT_MONITOR_TIMEOUT == 600  # 10 minutes
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
