from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from clarifai.cli.pipeline import generate, upload


def test_upload_python_pipeline_file_uses_pipeline_loader(tmp_path: Path):
    pipeline_file = tmp_path / 'pipeline_def.py'
    pipeline_file.write_text('pipeline = None\n', encoding='utf-8')
    runner = CliRunner()

    with patch('clarifai.runners.pipelines.load_pipeline_from_file') as mock_loader:
        mock_pipeline = Mock()
        mock_loader.return_value = mock_pipeline

        result = runner.invoke(upload, [str(pipeline_file)])

    assert result.exit_code == 0
    mock_loader.assert_called_once_with(str(pipeline_file))
    mock_pipeline.upload.assert_called_once_with(no_lockfile=False)


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