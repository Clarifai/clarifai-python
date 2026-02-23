"""Tests for model init with vllm toolkit (embedded templates, no GitHub clone)."""

import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli


def test_model_init_vllm_toolkit(monkeypatch, tmp_path):
    """Happy path: --model-name provided -> checkpoints.repo_id set, model.id sanitized."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'vllm_model'
    result = runner.invoke(
        cli,
        [
            'model',
            'init',
            str(model_dir),
            '--toolkit',
            'vllm',
            '--model-name',
            'microsoft/phi-1_5',
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    assert cfg_path.exists(), 'config.yaml not created'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'checkpoints' in data and isinstance(data['checkpoints'], dict), (
        'checkpoints section missing'
    )
    assert data['checkpoints']['repo_id'] == 'microsoft/phi-1_5'
    assert data['model']['id'] == 'phi-1-5'  # sanitized: dots and underscores handled

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    assert 'VLLMModel' in model_py.read_text(), 'embedded vllm model class missing'

    requirements = model_dir / 'requirements.txt'
    assert requirements.exists(), 'requirements.txt missing'
    assert 'vllm' in requirements.read_text()


def test_model_init_vllm_no_model_name(monkeypatch, tmp_path):
    """No --model-name: default checkpoint from embedded template remains."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'vllm_model2'
    result = runner.invoke(
        cli,
        ['model', 'init', str(model_dir), '--toolkit', 'vllm'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    data = yaml.safe_load(cfg_path.read_text())
    # Default checkpoint from embedded template
    assert 'checkpoints' in data
    assert data['checkpoints']['repo_id'] == 'google/gemma-3-1b-it'
    assert (model_dir / '1' / 'model.py').exists()
