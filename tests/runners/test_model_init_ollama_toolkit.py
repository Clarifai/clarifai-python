import os

import pytest
from click.testing import CliRunner

import clarifai.cli.model as model_module
from clarifai.cli.model import init as model_init


@pytest.mark.parametrize(
    "custom,model_name,port,context_length",
    [
        (True, "my-ollama", "4567", "9999"),
        (False, None, None, None),
    ],
)
def test_model_init_ollama(monkeypatch, tmp_path, custom, model_name, port, context_length):
    """Test ollama toolkit init with and without customization flags."""
    runner = CliRunner()
    called = {'clone': False}

    def fake_clone(repo_url, clone_dir, github_pat, branch):
        called['clone'] = True
        version_dir = os.path.join(clone_dir, '1')
        os.makedirs(version_dir, exist_ok=True)
        # template model file
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

    monkeypatch.setattr(model_module, 'clone_github_repo', fake_clone)
    monkeypatch.setattr(model_module, 'check_ollama_installed', lambda: True)
    monkeypatch.setattr(
        model_module, 'check_requirements_installed', lambda path: True, raising=False
    )
    # Avoid real GitHub listing by stubbing folder contents.
    monkeypatch.setattr(
        model_module.GitHubDownloader,
        'get_folder_contents',
        lambda self, owner, repo, path, branch: [
            {'name': '1', 'type': 'dir', 'path': '1'},
            {'name': 'config.yaml', 'type': 'file', 'path': 'config.yaml'},
            {'name': 'requirements.txt', 'type': 'file', 'path': 'requirements.txt'},
        ],
        raising=True,
    )

    # Simulate user pressing Enter at interactive prompts
    monkeypatch.setattr('builtins.input', lambda *a, **k: '\n')

    args = [
        str(tmp_path / ('ollama_custom' if custom else 'ollama_default')),
        '--toolkit',
        'ollama',
    ]
    if custom:
        args.extend(
            ['--model-name', model_name, '--port', port, '--context-length', context_length]
        )

    result = runner.invoke(model_init, args, standalone_mode=False)

    assert result.exit_code == 0, result.output
    assert called['clone'] is True
    # We allow prompts; just ensure command succeeds.

    model_py = tmp_path / ('ollama_custom' if custom else 'ollama_default') / '1' / 'model.py'
    content = model_py.read_text()

    if custom:
        assert model_name in content
        assert f"PORT = '{port}'" in content
        assert f"context_length = '{context_length}'" in content
        # old defaults replaced
        assert "PORT = '23333'" not in content
        assert "context_length = '8192'" not in content
    else:
        # defaults remain
        assert "llama3.2" in content
        assert "PORT = '23333'" in content
        assert "context_length = '8192'" in content

    # baseline file set
    root = tmp_path / ('ollama_custom' if custom else 'ollama_default')
    files = {p.name for p in root.iterdir()}
    assert {'1', 'config.yaml', 'requirements.txt'}.issubset(files)
