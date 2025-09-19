import os

import requests
from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.model import init as model_init


def test_model_init_ollama_toolkit(monkeypatch, tmp_path):
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
                "# placeholder template\nimport os\n\nclass Dummy:\n"
                "    def __init__(self):\n"
                "        self.model = os.environ.get(\"OLLAMA_MODEL_NAME\", 'llama3.2')\n\n"
                "PORT = '23333'\ncontext_length = '8192'\n"
            )
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write("model:\n  id: dummy\n")
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write("# none\n")
        return True

    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_ollama_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    model_dir = tmp_path / 'ollama_model'

    result = runner.invoke(
        model_init,
        [
            str(model_dir),
            '--toolkit',
            'ollama',
            '--model-name',
            'my-ollama',
            '--port',
            '4567',
            '--context-length',
            '9999',
        ],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    assert called['clone'] is True
    assert called['repo_url'] is not None

    model_py_path = model_dir / '1' / 'model.py'
    assert model_py_path.exists()
    content = model_py_path.read_text()
    assert "my-ollama" in content
    assert "PORT = '4567'" in content
    assert "context_length = '9999'" in content
    # Originals removed
    assert "llama3.2" not in content
    assert "PORT = '23333'" not in content
    assert "context_length = '8192'" not in content


def test_model_init_ollama_defaults(monkeypatch, tmp_path):
    """No customization flags: placeholders remain unchanged."""
    runner = CliRunner()

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(version_dir, 'model.py'), 'w') as f:
            f.write(
                "# placeholder template\nimport os\n\nclass Dummy:\n"
                "    def __init__(self):\n"
                "        self.model = os.environ.get(\"OLLAMA_MODEL_NAME\", 'llama3.2')\n\n"
                "PORT = '23333'\ncontext_length = '8192'\n"
            )
        with open(os.path.join(clone_dir, 'config.yaml'), 'w') as f:
            f.write("model:\n  id: dummy\n")
        with open(os.path.join(clone_dir, 'requirements.txt'), 'w') as f:
            f.write("# none\n")
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

    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_ollama_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "")

    model_dir = tmp_path / 'ollama_model_defaults'
    result = runner.invoke(
        model_init,
        [str(model_dir), '--toolkit', 'ollama'],
        standalone_mode=False,
    )

    assert result.exit_code == 0, result.output
    content = (model_dir / '1' / 'model.py').read_text()
    assert "self.model = os.environ.get(\"OLLAMA_MODEL_NAME\", 'llama3.2')" in content
    assert "PORT = '23333'" in content
    assert "context_length = '8192'" in content
