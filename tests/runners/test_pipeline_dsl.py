from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from clarifai.runners.pipelines import ComputeConfig, Pipeline, step, step_ref
from clarifai.runners.utils.pipeline_validation import PipelineConfigValidator


def normalize_text(value: str) -> str:
    return ' '.join(value.strip().split())


@step(
    id='prepare-text',
    requirements=['transformers>=4.0'],
    compute=ComputeConfig(cpu_limit='500m', cpu_memory='500Mi'),
)
def prepare_text(input_text: str) -> str:
    return normalize_text(input_text)


summarize = step_ref.from_url(
    'users/me/apps/shared-app/pipeline_steps/summarize/versions/summary-v1',
    secrets={'OPENAI_API_KEY': 'users/me/secrets/openai-key'},
)


classify_sentiment = step_ref(
    id='classify-sentiment',
    user_id='me',
    app_id='shared-app',
    version_id='sentiment-v2',
)


@step(id='assemble-report')
def assemble_report(summary: str, sentiment: str) -> str:
    return f'{summary}:{sentiment}'


def build_pipeline() -> Pipeline:
    with Pipeline(id='text-pipeline', user_id='me', app_id='my-app') as pipeline:
        raw_text = pipeline.input('input_text')
        prepared = prepare_text(input_text=raw_text)
        summary = summarize(input_text=prepared.output())
        sentiment = classify_sentiment(input_text=prepared.output())
        report = assemble_report(
            summary=summary.output(),
            sentiment=sentiment.output(),
        )
        prepared >> [summary, sentiment] >> report
    return pipeline


def test_pipeline_to_config_supports_diamond_dag_and_secrets():
    pipeline = build_pipeline()

    config = pipeline.to_config()
    PipelineConfigValidator.validate_config(config)

    argo_spec = yaml.safe_load(config['pipeline']['orchestration_spec']['argo_orchestration_spec'])
    step_groups = argo_spec['spec']['templates'][0]['steps']
    # Flatten into a name→entry map for easy assertions
    tasks = {entry['name']: entry for group in step_groups for entry in group}

    assert set(tasks) == {
        'prepare-text',
        'summarize',
        'classify-sentiment',
        'assemble-report',
    }
    # Verify layer ordering: prepare-text before summarize/sentiment, both before assemble-report
    layer_names = [sorted(entry['name'] for entry in group) for group in step_groups]
    assert layer_names[0] == ['prepare-text']
    assert layer_names[1] == ['classify-sentiment', 'summarize']
    assert layer_names[2] == ['assemble-report']
    assert (
        tasks['summarize']['templateRef']['name']
        == 'users/me/apps/shared-app/pipeline_steps/summarize/versions/summary-v1'
    )
    assert (
        tasks['classify-sentiment']['templateRef']['name']
        == 'users/me/apps/shared-app/pipeline_steps/classify-sentiment/versions/sentiment-v2'
    )
    assert (
        tasks['summarize']['arguments']['parameters'][0]['value']
        == '{{steps.prepare-text.outputs.parameters.result}}'
    )
    assert config['pipeline']['step_directories'] == ['prepare-text', 'assemble-report']
    assert config['pipeline']['config']['step_version_secrets'] == {
        'summarize': {'OPENAI_API_KEY': 'users/me/secrets/openai-key'}
    }


def test_pipeline_generate_writes_helper_functions_and_expected_files(tmp_path: Path):
    pipeline = build_pipeline()

    config_path = Path(pipeline.generate(str(tmp_path)))

    assert config_path.exists()
    assert (tmp_path / 'prepare-text' / 'config.yaml').exists()
    assert (tmp_path / 'prepare-text' / 'requirements.txt').exists()
    step_script = tmp_path / 'prepare-text' / '1' / 'pipeline_step.py'
    assert step_script.exists()

    step_script_content = step_script.read_text(encoding='utf-8')
    requirements_content = (tmp_path / 'prepare-text' / 'requirements.txt').read_text(
        encoding='utf-8'
    )

    assert 'def normalize_text(value: str) -> str:' in step_script_content
    assert '@step' not in step_script_content
    assert 'clarifai==' in requirements_content
    assert 'transformers>=4.0' in requirements_content
    assert not (tmp_path / 'summarize').exists()
    assert not (tmp_path / 'classify-sentiment').exists()


def test_validator_collects_only_managed_steps_without_versions():
    pipeline = build_pipeline()

    steps = PipelineConfigValidator.get_pipeline_steps_without_versions(pipeline.to_config())

    assert steps == ['prepare-text', 'assemble-report']


def test_pipeline_run_uploads_then_delegates_to_client():
    pipeline = build_pipeline()

    with patch.object(pipeline, 'upload', return_value='pipeline-version-123') as mock_upload:
        with patch('clarifai.client.pipeline.Pipeline') as mock_client_class:
            mock_client = Mock()
            mock_client.run.return_value = {'status': 'ok'}
            mock_client_class.return_value = mock_client

            result = pipeline.run(timeout=12, monitor_interval=3)

    assert result == {'status': 'ok'}
    mock_upload.assert_called_once_with(no_lockfile=False)
    mock_client_class.assert_called_once_with(
        pipeline_id='text-pipeline',
        pipeline_version_id='pipeline-version-123',
        user_id='me',
        app_id='my-app',
        nodepool_id=None,
        compute_cluster_id=None,
        log_file=None,
    )
    mock_client.run.assert_called_once_with(
        inputs=None,
        timeout=12,
        monitor_interval=3,
        input_args_override=None,
    )


def test_step_ref_is_not_callable_outside_pipeline():
    with pytest.raises(RuntimeError, match='active Pipeline'):
        summarize(input_text='hello')


def test_step_ref_from_url_parses_versioned_pipeline_step_url():
    step_definition = step_ref.from_url(
        'https://api.clarifai.com/v2/users/demo-user/apps/shared-app/pipeline_steps/summarize/versions/summary-v1',
        secrets={'OPENAI_API_KEY': 'users/demo-user/secrets/openai-key'},
    )

    assert step_definition.id == 'summarize'
    assert step_definition.user_id == 'demo-user'
    assert step_definition.app_id == 'shared-app'
    assert step_definition.version_id == 'summary-v1'
    assert step_definition.secrets == {'OPENAI_API_KEY': 'users/demo-user/secrets/openai-key'}


def test_step_ref_from_url_requires_versioned_pipeline_step_path():
    with pytest.raises(ValueError, match='versioned pipeline step URL or resource path'):
        step_ref.from_url('users/demo-user/apps/shared-app/pipeline_steps/summarize')