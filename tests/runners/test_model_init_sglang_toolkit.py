"""Tests for model init with sglang toolkit (embedded templates, no GitHub clone)."""

import yaml
from click.testing import CliRunner

from clarifai.cli.base import cli


def test_model_init_sglang_toolkit(monkeypatch, tmp_path):
    """Happy path: --model-name provided -> checkpoints.repo_id set, model.id sanitized."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'sglang_model'
    result = runner.invoke(
        cli,
        [
            'model',
            'init',
            str(model_dir),
            '--toolkit',
            'sglang',
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
    assert data['model']['id'] == 'phi-1-5'

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    assert 'SGLangModel' in model_py.read_text(), 'embedded sglang model class missing'


def test_model_init_sglang_no_model_name(monkeypatch, tmp_path):
    """No --model-name: default checkpoint from embedded template remains."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])

    model_dir = tmp_path / 'sglang_model2'
    result = runner.invoke(
        cli,
        ['model', 'init', str(model_dir), '--toolkit', 'sglang'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'checkpoints' in data
    assert data['checkpoints']['repo_id'] == 'google/gemma-3-1b-it'
    assert (model_dir / '1' / 'model.py').exists()
