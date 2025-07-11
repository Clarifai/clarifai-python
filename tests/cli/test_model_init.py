import tempfile
from unittest import mock

from click.testing import CliRunner

from clarifai.cli.base import cli


class TestModelInitCommand:
    """Test cases for the model init functionality."""

    def test_toolkit_option_help(self):
        """Test that --toolkit option appears in help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'init', '--help'])

        assert result.exit_code == 0
        assert '--toolkit' in result.output
        assert '--model-name' in result.output

    def test_model_name_without_toolkit_fails(self):
        """Test that --model-name without --toolkit or --local-ollama-model fails."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', '--model-name', 'test', temp_dir])

        assert result.exit_code == 1
        assert "--model-name can only be used with --toolkit or --local-ollama-model" in result.output

    def test_toolkit_and_local_ollama_conflict(self):
        """Test that --toolkit and --local-ollama-model cannot be used together."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', '--toolkit', 'ollama', '--local-ollama-model', temp_dir])

        assert result.exit_code == 1
        assert "Cannot specify both --toolkit and --local-ollama-model" in result.output

    def test_toolkit_and_github_repo_conflict(self):
        """Test that --toolkit and --github-repo cannot be used together."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', '--toolkit', 'ollama', '--github-repo', 'test/repo', temp_dir])

        assert result.exit_code == 1
        assert "Cannot specify both --toolkit and --github-repo/--branch" in result.output

    @mock.patch('clarifai.cli.model.clone_github_repo')
    @mock.patch('os.listdir')
    @mock.patch('shutil.copytree')
    @mock.patch('shutil.copy2')
    def test_toolkit_ollama_success(self, mock_copy2, mock_copytree, mock_listdir, mock_clone):
        """Test that --toolkit ollama works correctly."""
        mock_clone.return_value = True
        mock_listdir.return_value = ['1', 'config.yaml', 'requirements.txt']

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', '--toolkit', 'ollama', temp_dir])

        assert result.exit_code == 0
        assert mock_clone.called

    @mock.patch('clarifai.cli.model.clone_github_repo')
    @mock.patch('os.listdir')
    @mock.patch('shutil.copytree')
    @mock.patch('shutil.copy2')
    def test_local_ollama_with_model_name_success(self, mock_copy2, mock_copytree, mock_listdir, mock_clone):
        """Test that --local-ollama-model with --model-name works correctly."""
        mock_clone.return_value = True
        mock_listdir.return_value = ['1', 'config.yaml', 'requirements.txt']

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', '--local-ollama-model', temp_dir])

        assert result.exit_code == 0
        assert mock_clone.called
