"""Tests for model init with lmstudio toolkit (embedded templates, no GitHub clone)."""

import yaml
from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.base import cli


def test_model_init_lmstudio_with_model_name(monkeypatch, tmp_path):
    """Happy path: --model-name provided -> toolkit.model set in config."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])
    monkeypatch.setattr(model_module, 'check_lmstudio_installed', lambda: True)

    model_dir = tmp_path / 'lmstudio_model'
    result = runner.invoke(
        cli,
        [
            'model',
            'init',
            str(model_dir),
            '--toolkit',
            'lmstudio',
            '--model-name',
            'qwen/qwen3-4b',
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    assert cfg_path.exists(), 'config.yaml not created'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'toolkit' in data and isinstance(data['toolkit'], dict), 'toolkit section missing'
    assert data['toolkit']['model'] == 'qwen/qwen3-4b'
    assert data['model']['id'] == 'qwen3-4b'

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    assert 'LMStudioModel' in model_py.read_text(), 'embedded lmstudio model class missing'


def test_model_init_lmstudio_defaults(monkeypatch, tmp_path):
    """No --model-name: defaults from embedded template remain."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])
    monkeypatch.setattr(model_module, 'check_lmstudio_installed', lambda: True)

    model_dir = tmp_path / 'lmstudio_model_default'
    result = runner.invoke(
        cli,
        ['model', 'init', str(model_dir), '--toolkit', 'lmstudio'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    content = (model_dir / '1' / 'model.py').read_text()
    assert "LMS_MODEL_NAME = \"LiquidAI/LFM2-1.2B\"" in content
    assert "LMS_PORT = 11434" in content

    files = {p.name for p in model_dir.iterdir()}
    assert {'1', 'config.yaml', 'requirements.txt'}.issubset(files)
