"""Tests for model init with huggingface toolkit (embedded templates, no GitHub clone)."""

import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli


def test_model_init_huggingface_toolkit(monkeypatch, tmp_path):
    """Happy path: --model-name provided -> checkpoints.repo_id set, model.id sanitized."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'hf_model'
    result = runner.invoke(
        cli,
        [
            'model',
            'init',
            str(model_dir),
            '--toolkit',
            'huggingface',
            '--model-name',
            'UnsLOTH/Llama-1B',
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
    assert data['checkpoints']['repo_id'] == 'UnsLOTH/Llama-1B'
    assert data['model']['id'] == 'llama-1b'

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    assert 'HuggingFaceModel' in model_py.read_text(), 'embedded hf model class missing'

    requirements = model_dir / 'requirements.txt'
    assert requirements.exists(), 'requirements.txt missing'
    assert 'transformers' in requirements.read_text()


def test_model_init_hf_no_model_name(monkeypatch, tmp_path):
    """No --model-name: default checkpoint from embedded template remains."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'hf_model2'
    result = runner.invoke(
        cli,
        ['model', 'init', str(model_dir), '--toolkit', 'huggingface'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'checkpoints' in data
    assert data['checkpoints']['repo_id'] == 'unsloth/Llama-3.2-1B-Instruct'
    assert (model_dir / '1' / 'model.py').exists()
