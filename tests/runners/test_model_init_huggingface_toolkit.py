import os

import requests
import yaml
from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.model import init as model_init


def test_model_init_huggingface_toolkit(monkeypatch, tmp_path):
    """Happy path: model-name provided -> checkpoints.repo_id created and set."""
    runner = CliRunner()
    called = {'clone': False, 'repo_url': None, 'branch': None}

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        called['clone'] = True
        called['repo_url'] = repo_url
        called['branch'] = branch
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        # minimal model file (content should remain unchanged by huggingface customization)
        with open(os.path.join(version_dir, 'model.py'), 'w') as f:
            f.write('pass')
        # config WITHOUT checkpoints so code path adds it
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
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    model_dir = tmp_path / 'hf_model'
    result = runner.invoke(
        model_init,
        [str(model_dir), '--toolkit', 'huggingface', '--model-name', 'UnsLOTH/Llama-1B'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert called['clone'] is True
    assert called['repo_url'] is not None  # sanity that our fake saw a value

    cfg_path = model_dir / 'config.yaml'
    assert cfg_path.exists(), 'config.yaml not created'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'checkpoints' in data and isinstance(data['checkpoints'], dict), (
        'checkpoints section missing'
    )
    assert data['checkpoints']['repo_id'] == 'UnsLOTH/Llama-1B'

    model_py = model_dir / '1' / 'model.py'
    assert model_py.exists(), 'model.py missing'
    assert model_py.read_text() == 'pass', 'model.py unexpectedly modified'


def test_model_init_hf_no_model_name(monkeypatch, tmp_path):
    """No --model-name: checkpoints section should NOT be added (mirrors current logic)."""
    runner = CliRunner()
    called = {'clone': False}

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        called['clone'] = True
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(version_dir, 'model.py'), 'w') as f:
            f.write('pass')
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write('model:\n  id: dummy\n')
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write('# none')
        return True

    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    model_dir = tmp_path / 'hf_model2'
    result = runner.invoke(
        model_init,
        [str(model_dir), '--toolkit', 'huggingface'],  # no --model-name
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert called['clone'] is True

    cfg_path = model_dir / 'config.yaml'
    data = yaml.safe_load(cfg_path.read_text())
    assert 'checkpoints' not in data, 'checkpoints unexpectedly added without model-name'
    assert (model_dir / '1' / 'model.py').exists()
