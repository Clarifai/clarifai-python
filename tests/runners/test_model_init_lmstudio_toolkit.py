import os

import requests
import yaml
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
            f.write('pass')
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write('model:\n  id: dummy\n')
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write('# none')
        return True

    # --- NEW: stub GitHub API listing to avoid real network call ---
    class _FakeResp:
        def __init__(self, data, status_code=200):
            self._data = data
            self.status_code = status_code
            self.text = 'ok'

        def json(self):
            return self._data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def fake_requests_get(url, *a, **kw):
        # Simulate the directory listing the code expects
        if "/contents" in url:
            return _FakeResp(
                [
                    {'name': '1', 'type': 'dir', 'path': '1'},
                    {'name': 'config.yaml', 'type': 'file', 'path': 'config.yaml'},
                    {'name': 'requirements.txt', 'type': 'file', 'path': 'requirements.txt'},
                ]
            )
        return _FakeResp({})

    # Patch requests.get inside the module under test
    monkeypatch.setattr(requests, "get", fake_requests_get, raising=True)
    # ----------------------------------------------------------------

    # Patches
    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_lmstudio_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

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

    cfg_path = model_dir / 'config.yaml'
    assert cfg_path.exists(), 'config.yaml not created'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'toolkit' in data and isinstance(data['toolkit'], dict), 'toolkit section missing'

    # New values
    assert data['toolkit']['model'] == 'qwen/qwen3-4b'
    assert data['toolkit']['port'] == '11888'
    assert data['toolkit']['context_length'] == '16000'

    # Originals removed
    assert data['toolkit']['model'] != 'LiquidAI/LFM2-1.2B'
    assert data['toolkit']['port'] != '11434'
    assert data['toolkit']['context_length'] != '2048'


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
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

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
