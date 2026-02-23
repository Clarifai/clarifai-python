"""Tests for model init with ollama toolkit (embedded templates, no GitHub clone)."""

import yaml
from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.base import cli


def test_model_init_ollama_with_model_name(monkeypatch, tmp_path):
    """Happy path: --model-name provided -> toolkit.model set in config."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])
    monkeypatch.setattr(model_module, 'check_ollama_installed', lambda: True)

    model_dir = tmp_path / 'ollama_custom'
    result = runner.invoke(
        cli,
        [
            'model',
            'init',
            str(model_dir),
            '--toolkit',
            'ollama',
            '--model-name',
            'llama3.1',
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    cfg_path = model_dir / 'config.yaml'
    assert cfg_path.exists(), 'config.yaml not created'
    data = yaml.safe_load(cfg_path.read_text())
    assert data['toolkit']['model'] == 'llama3.1'
    assert data['model']['id'] == 'llama31'  # sanitized: dots stripped

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    content = model_py.read_text()
    assert 'OllamaModel' in content, 'embedded ollama model class missing'
    # model name should be replaced in the model.py
    assert 'llama3.1' in content

    files = {p.name for p in model_dir.iterdir()}
    assert {'1', 'config.yaml', 'requirements.txt'}.issubset(files)


def test_model_init_ollama_defaults(monkeypatch, tmp_path):
    """No --model-name: defaults from embedded template remain."""
    runner = CliRunner()
    runner.invoke(cli, ["login", "--user_id", "test_user"])
    monkeypatch.setattr(model_module, 'check_ollama_installed', lambda: True)

    model_dir = tmp_path / 'ollama_default'
    result = runner.invoke(
        cli,
        ['model', 'init', str(model_dir), '--toolkit', 'ollama'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output

    model_py = model_dir / '1' / 'model.py'
    content = model_py.read_text()
    # defaults remain
    assert 'llama3.2' in content

    files = {p.name for p in model_dir.iterdir()}
    assert {'1', 'config.yaml', 'requirements.txt'}.issubset(files)
