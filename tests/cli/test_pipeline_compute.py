"""Tests for pipeline run compute auto-resolution (--instance flag).

Mirrors the patterns from test_model_deploy.py TestComputeConfigs and
TestModelDeployerValidation, applied to the pipeline run CLI.
"""

from unittest.mock import Mock, patch

import yaml
from click.testing import CliRunner

from clarifai.cli.pipeline import run


class _MockContext:
    """Reusable mock CLI context for pipeline tests."""

    def __init__(self):
        self.pat = 'test-pat'
        self.api_base = 'https://api.clarifai.com'
        self.user_id = 'test-user'

    def get(self, key, default=None):
        return getattr(self, key, default)


class _MockConfig:
    def __init__(self):
        self.current = _MockContext()


# Common CLI args for a valid pipeline identity (used across multiple tests)
_PIPELINE_IDENTITY_ARGS = [
    '--pipeline_id',
    'test-pipeline',
    '--pipeline_version_id',
    'v1',
    '--user_id',
    'test-user',
    '--app_id',
    'test-app',
]


class TestPipelineRunInstanceFlag:
    """Test --instance flag auto-creates compute infrastructure for pipeline run."""

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_instance_flag_triggers_auto_compute(
        self, mock_ensure, mock_validate, mock_pipeline_class
    ):
        """--instance without cc/np calls _ensure_pipeline_compute and passes resolved IDs."""
        mock_ensure.return_value = ('deploy-cc-aws-us-east-1', 'deploy-np-g5-xlarge')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                _PIPELINE_IDENTITY_ARGS + ['--instance', 'g5.xlarge'],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        mock_ensure.assert_called_once()
        # Verify auto-resolved IDs were passed to Pipeline
        call_kwargs = mock_pipeline_class.call_args[1]
        assert call_kwargs['compute_cluster_id'] == 'deploy-cc-aws-us-east-1'
        assert call_kwargs['nodepool_id'] == 'deploy-np-g5-xlarge'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    def test_explicit_cc_np_skips_auto_compute(self, mock_validate, mock_pipeline_class):
        """Explicit --compute_cluster_id + --nodepool_id bypasses auto-creation entirely."""
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                _PIPELINE_IDENTITY_ARGS
                + [
                    '--compute_cluster_id',
                    'my-cc',
                    '--nodepool_id',
                    'my-np',
                ],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_pipeline_class.call_args[1]
        assert call_kwargs['compute_cluster_id'] == 'my-cc'
        assert call_kwargs['nodepool_id'] == 'my-np'

    @patch('clarifai.utils.cli.validate_context')
    def test_no_instance_no_cc_np_raises_error(self, mock_validate):
        """Missing both --instance and --compute_cluster_id/--nodepool_id raises clear error."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                _PIPELINE_IDENTITY_ARGS,
                obj=_MockConfig(),
            )

        assert result.exit_code != 0
        assert '--instance is required' in str(result.exception)

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_instance_with_cloud_region_override(
        self, mock_ensure, mock_validate, mock_pipeline_class
    ):
        """--instance + --cloud + --region passes overrides to _ensure_pipeline_compute."""
        mock_ensure.return_value = ('deploy-cc-gcp-us-central1', 'deploy-np-a10g')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                _PIPELINE_IDENTITY_ARGS
                + [
                    '--instance',
                    'a10g',
                    '--cloud',
                    'gcp',
                    '--region',
                    'us-central1',
                ],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        # Verify cloud/region were forwarded
        call_args = mock_ensure.call_args
        assert call_args[0][3] == 'gcp'  # cloud
        assert call_args[0][4] == 'us-central1'  # region

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_instance_from_config_yaml(self, mock_ensure, mock_validate, mock_pipeline_class):
        """compute.instance in config.yaml is used when --instance is not passed."""
        mock_ensure.return_value = ('deploy-cc-aws-us-east-1', 'deploy-np-g5-xlarge')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        config_data = {
            'pipeline': {
                'id': 'test-pipeline',
                'version_id': 'v1',
                'user_id': 'test-user',
                'app_id': 'test-app',
            },
            'compute': {
                'instance': 'g5.xlarge',
            },
        }

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('run-config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

            result = runner.invoke(
                run,
                ['--config', 'run-config.yaml'],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        mock_ensure.assert_called_once()
        # instance arg should be 'g5.xlarge' from config
        assert mock_ensure.call_args[0][2] == 'g5.xlarge'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_cli_instance_overrides_config(self, mock_ensure, mock_validate, mock_pipeline_class):
        """CLI --instance takes precedence over config.yaml compute.instance."""
        mock_ensure.return_value = ('deploy-cc-aws-us-east-1', 'deploy-np-l40s')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        config_data = {
            'pipeline': {
                'id': 'test-pipeline',
                'version_id': 'v1',
                'user_id': 'test-user',
                'app_id': 'test-app',
            },
            'compute': {
                'instance': 'g5.xlarge',  # config says g5.xlarge
            },
        }

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('run-config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

            result = runner.invoke(
                run,
                ['--config', 'run-config.yaml', '--instance', 'l40s'],  # CLI says l40s
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        # CLI flag should win
        assert mock_ensure.call_args[0][2] == 'l40s'

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_config_cloud_region_used_when_cli_omitted(
        self, mock_ensure, mock_validate, mock_pipeline_class
    ):
        """compute.cloud and compute.region from config used when CLI flags omitted."""
        mock_ensure.return_value = ('deploy-cc-gcp-eu-west1', 'deploy-np-a10g')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        config_data = {
            'pipeline': {
                'id': 'test-pipeline',
                'version_id': 'v1',
                'user_id': 'test-user',
                'app_id': 'test-app',
            },
            'compute': {
                'instance': 'a10g',
                'cloud': 'gcp',
                'region': 'eu-west1',
            },
        }

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('run-config.yaml', 'w') as f:
                yaml.safe_dump(config_data, f)

            result = runner.invoke(
                run,
                ['--config', 'run-config.yaml'],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        call_args = mock_ensure.call_args[0]
        assert call_args[2] == 'a10g'  # instance
        assert call_args[3] == 'gcp'  # cloud from config
        assert call_args[4] == 'eu-west1'  # region from config

    @patch('clarifai.client.pipeline.Pipeline')
    @patch('clarifai.utils.cli.validate_context')
    @patch('clarifai.cli.pipeline._ensure_pipeline_compute')
    def test_pipeline_url_with_instance(self, mock_ensure, mock_validate, mock_pipeline_class):
        """--pipeline_url + --instance works (url-based run with auto-compute)."""
        mock_ensure.return_value = ('deploy-cc-aws-us-east-1', 'deploy-np-g5-xlarge')
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {'status': 'success'}
        mock_pipeline_class.return_value = mock_pipeline

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                run,
                [
                    '--pipeline_url',
                    'https://clarifai.com/user/app/pipelines/my-pl',
                    '--instance',
                    'g5.xlarge',
                ],
                obj=_MockConfig(),
            )

        assert result.exit_code == 0, result.output
        mock_ensure.assert_called_once()
        call_kwargs = mock_pipeline_class.call_args[1]
        assert call_kwargs['nodepool_id'] == 'deploy-np-g5-xlarge'
        assert call_kwargs['compute_cluster_id'] == 'deploy-cc-aws-us-east-1'


class TestEnsurePipelineCompute:
    """Test _ensure_pipeline_compute helper directly (mirrors TestComputeConfigs from model deploy)."""

    @patch('clarifai.client.compute_cluster.ComputeCluster')
    @patch('clarifai.client.user.User')
    @patch('clarifai.utils.compute_presets.resolve_gpu')
    def test_creates_cluster_and_nodepool_when_missing(
        self, mock_resolve_gpu, mock_user_class, mock_cc_class
    ):
        """When cluster and nodepool don't exist, both are auto-created."""
        from clarifai.cli.pipeline import _ensure_pipeline_compute

        mock_resolve_gpu.return_value = {
            'instance_type_id': 'g5.xlarge',
            'cloud_provider': 'aws',
            'region': 'us-east-1',
            'inference_compute_info': {'cpu_limit': '4', 'num_accelerators': 1},
        }
        mock_user = Mock()
        mock_user.compute_cluster.side_effect = Exception("not found")
        mock_user_class.return_value = mock_user

        mock_cc = Mock()
        mock_cc.nodepool.side_effect = Exception("not found")
        mock_cc_class.return_value = mock_cc

        ctx = Mock()
        ctx.obj.current.pat = 'test-pat'
        ctx.obj.current.api_base = 'https://api.clarifai.com'

        cc_id, np_id = _ensure_pipeline_compute(
            ctx, 'test-user', 'g5.xlarge', None, None, None, None
        )

        assert cc_id == 'deploy-cc-aws-us-east-1'
        assert np_id == 'deploy-np-g5-xlarge'
        mock_user.create_compute_cluster.assert_called_once()
        mock_cc.create_nodepool.assert_called_once()

    @patch('clarifai.client.compute_cluster.ComputeCluster')
    @patch('clarifai.client.user.User')
    @patch('clarifai.utils.compute_presets.resolve_gpu')
    def test_reuses_existing_cluster_and_nodepool(
        self, mock_resolve_gpu, mock_user_class, mock_cc_class
    ):
        """When cluster and nodepool already exist, no creation calls are made."""
        from clarifai.cli.pipeline import _ensure_pipeline_compute

        mock_resolve_gpu.return_value = {
            'instance_type_id': 'g5.xlarge',
            'cloud_provider': 'aws',
            'region': 'us-east-1',
            'inference_compute_info': {},
        }
        mock_user = Mock()
        mock_user.compute_cluster.return_value = Mock()  # exists
        mock_user_class.return_value = mock_user

        mock_cc = Mock()
        mock_cc.nodepool.return_value = Mock()  # exists
        mock_cc_class.return_value = mock_cc

        ctx = Mock()
        ctx.obj.current.pat = 'test-pat'
        ctx.obj.current.api_base = 'https://api.clarifai.com'

        cc_id, np_id = _ensure_pipeline_compute(
            ctx, 'test-user', 'g5.xlarge', None, None, None, None
        )

        assert cc_id == 'deploy-cc-aws-us-east-1'
        assert np_id == 'deploy-np-g5-xlarge'
        mock_user.create_compute_cluster.assert_not_called()
        mock_cc.create_nodepool.assert_not_called()

    @patch('clarifai.client.compute_cluster.ComputeCluster')
    @patch('clarifai.client.user.User')
    @patch('clarifai.utils.compute_presets.resolve_gpu')
    def test_custom_cloud_region(self, mock_resolve_gpu, mock_user_class, mock_cc_class):
        """Explicit cloud/region overrides auto-detected values from resolve_gpu."""
        from clarifai.cli.pipeline import _ensure_pipeline_compute

        mock_resolve_gpu.return_value = {
            'instance_type_id': 'gpu-nvidia-a10g',
            'cloud_provider': 'aws',  # will be overridden
            'region': 'us-east-1',  # will be overridden
            'inference_compute_info': {},
        }
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_cc = Mock()
        mock_cc_class.return_value = mock_cc

        ctx = Mock()
        ctx.obj.current.pat = 'test-pat'
        ctx.obj.current.api_base = 'https://api.clarifai.com'

        cc_id, np_id = _ensure_pipeline_compute(
            ctx, 'test-user', 'a10g', 'gcp', 'us-central1', None, None
        )

        assert cc_id == 'deploy-cc-gcp-us-central1'
        assert np_id == 'deploy-np-gpu-nvidia-a10g'

    @patch('clarifai.utils.compute_presets.resolve_gpu')
    def test_unknown_instance_raises_error(self, mock_resolve_gpu):
        """Unknown instance type raises ValueError."""
        from clarifai.cli.pipeline import _ensure_pipeline_compute

        mock_resolve_gpu.side_effect = ValueError("Unknown GPU")

        ctx = Mock()
        ctx.obj.current.pat = 'test-pat'
        ctx.obj.current.api_base = 'https://api.clarifai.com'

        import pytest

        with pytest.raises(ValueError):
            _ensure_pipeline_compute(ctx, 'test-user', 'nonexistent-gpu', None, None, None, None)

    @patch('clarifai.client.compute_cluster.ComputeCluster')
    @patch('clarifai.client.user.User')
    @patch('clarifai.utils.compute_presets.resolve_gpu')
    def test_explicit_cc_id_preserved(self, mock_resolve_gpu, mock_user_class, mock_cc_class):
        """When compute_cluster_id is explicitly provided, it's used instead of auto-generated."""
        from clarifai.cli.pipeline import _ensure_pipeline_compute

        mock_resolve_gpu.return_value = {
            'instance_type_id': 'g5.xlarge',
            'cloud_provider': 'aws',
            'region': 'us-east-1',
            'inference_compute_info': {},
        }
        mock_user = Mock()
        mock_user_class.return_value = mock_user
        mock_cc = Mock()
        mock_cc_class.return_value = mock_cc

        ctx = Mock()
        ctx.obj.current.pat = 'test-pat'
        ctx.obj.current.api_base = 'https://api.clarifai.com'

        cc_id, np_id = _ensure_pipeline_compute(
            ctx, 'test-user', 'g5.xlarge', None, None, 'my-custom-cc', None
        )

        assert cc_id == 'my-custom-cc'
        assert np_id == 'deploy-np-g5-xlarge'
