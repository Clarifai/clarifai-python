from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from click.testing import CliRunner

from clarifai.cli.pipeline import generate, upload


def test_upload_python_pipeline_file_generates_and_uploads_directory(tmp_path: Path):
    pipeline_file = tmp_path / 'pipeline_def.py'
    pipeline_file.write_text('pipeline = None\n', encoding='utf-8')
    runner = CliRunner()

    with (
        patch('clarifai.runners.pipelines.load_pipeline_from_file') as mock_loader,
        patch('clarifai.runners.pipelines.pipeline_builder.upload_pipeline') as mock_upload,
    ):
        mock_pipeline = Mock()
        mock_pipeline.id = 'test-pipeline'
        mock_loader.return_value = mock_pipeline

        result = runner.invoke(upload, [str(pipeline_file)])

    expected_dir = str(tmp_path / 'generated-test-pipeline')
    assert result.exit_code == 0
    mock_loader.assert_called_once_with(str(pipeline_file))
    mock_pipeline.generate.assert_called_once_with(expected_dir)
    mock_upload.assert_called_once_with(expected_dir, no_lockfile=False)


def test_generate_python_pipeline_file_writes_output(tmp_path: Path):
    pipeline_file = tmp_path / 'pipeline_def.py'
    pipeline_file.write_text('pipeline = None\n', encoding='utf-8')
    output_dir = tmp_path / 'generated'
    runner = CliRunner()

    with patch('clarifai.runners.pipelines.load_pipeline_from_file') as mock_loader:
        mock_pipeline = Mock()
        mock_pipeline.generate.return_value = str(output_dir / 'config.yaml')
        mock_loader.return_value = mock_pipeline

        result = runner.invoke(generate, [str(pipeline_file), '--output-dir', str(output_dir)])

    assert result.exit_code == 0
    mock_loader.assert_called_once_with(str(pipeline_file))
    mock_pipeline.generate.assert_called_once_with(str(output_dir))


def test_generate_real_example_pipeline_writes_mixed_step_config(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_file = repo_root / 'examples' / 'pipeline_dsl_text_pipeline.py'
    output_dir = tmp_path / 'generated'
    runner = CliRunner()

    result = runner.invoke(generate, [str(pipeline_file), '--output-dir', str(output_dir)])

    assert result.exit_code == 0, result.output

    config = yaml.safe_load((output_dir / 'config.yaml').read_text(encoding='utf-8'))
    argo_spec = yaml.safe_load(config['pipeline']['orchestration_spec']['argo_orchestration_spec'])
    step_groups = argo_spec['spec']['templates'][0]['steps']
    tasks = {entry['name']: entry for group in step_groups for entry in group}

    assert config['pipeline']['step_directories'] == ['prepare-text', 'assemble-report']
    assert tasks['summarize']['templateRef']['name'].endswith('/versions/summary-v1')
    assert tasks['classify-sentiment']['templateRef']['name'].endswith('/versions/sentiment-v3')
    assert not (output_dir / 'summarize').exists()
    assert not (output_dir / 'classify-sentiment').exists()
