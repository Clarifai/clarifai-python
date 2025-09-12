import os

from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.model import init as model_init


def test_model_init_lmstudio_toolkit(monkeypatch, tmp_path):
    """Happy path: all customization flags provided; placeholders replaced."""
    runner = CliRunner()
    called = {'clone': False, 'repo_url': None, 'branch': None}

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        called['clone'] = True
        called['repo_url'] = repo_url
        called['branch'] = branch
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        model_py = os.path.join(version_dir, 'model.py')
        with open(model_py, 'w') as f:
            f.write(
                "LMS_MODEL_NAME = 'LiquidAI/LFM2-1.2B'\n"
                "LMS_PORT = 11434\n"
                "LMS_CONTEXT_LENGTH = 4096\n"
            )
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write('model:\n  id: dummy\n')
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write('# none')
        return True

    # Patches
    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_lmstudio_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )

    model_dir = tmp_path / 'lmstudio_model'

    result = runner.invoke(
        model_init,
        [
            str(model_dir),
            '--toolkit',
            'lmstudio',
            '--model-name',
            'qwen/qwen3-4b',
            '--port',
            '11888',
            '--context-length',
            '16000',
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert called['clone'] is True
    assert called['repo_url'] is not None

    model_py_path = model_dir / '1' / 'model.py'
    assert model_py_path.exists()
    content = model_py_path.read_text()

    # New values
    assert "LMS_MODEL_NAME = 'qwen/qwen3-4b'" in content
    assert "LMS_PORT = 11888" in content
    assert "LMS_CONTEXT_LENGTH = 16000" in content

    # Originals removed
    assert "LMS_MODEL_NAME = 'LiquidAI/LFM2-1.2B'" not in content
    assert "LMS_PORT = 11434" not in content
    assert "LMS_CONTEXT_LENGTH = 4096" not in content

    # config.yaml present
    assert (model_dir / 'config.yaml').read_text().startswith('model:')


def test_model_init_lmstudio_defaults(monkeypatch, tmp_path):
    """No customization flags: placeholders remain unchanged."""
    runner = CliRunner()
    called = {'clone': True}

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(version_dir, 'model.py'), 'w') as f:
            f.write(
                "LMS_MODEL_NAME = 'LiquidAI/LFM2-1.2B'\n"
                "LMS_PORT = 11434\n"
                "LMS_CONTEXT_LENGTH = 4096\n"
            )
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write('model:\n  id: dummy\n')
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write('# none')
        return True

    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_lmstudio_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )

    model_dir = tmp_path / 'lmstudio_model_default'
    result = runner.invoke(
        model_init,
        [str(model_dir), '--toolkit', 'lmstudio'],  # no customization args
        standalone_mode=False,
    )
    assert result.exit_code == 0, result.output

    content = (model_dir / '1' / 'model.py').read_text()
    assert "LMS_MODEL_NAME = 'LiquidAI/LFM2-1.2B'" in content
    assert "LMS_PORT = 11434" in content
    assert "LMS_CONTEXT_LENGTH = 4096" in content
