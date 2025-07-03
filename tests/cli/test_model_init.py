import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from clarifai.cli.base import cli


class TestModelInit:
    """Test cases for clarifai model init command."""

    def test_init_basic_template(self):
        """Test basic model initialization with templates."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', temp_dir])
            
            assert result.exit_code == 0
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(temp_dir, "requirements.txt"))
            assert os.path.exists(os.path.join(temp_dir, "config.yaml"))

    def test_init_with_model_type_mcp(self):
        """Test model initialization with MCP model type."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', temp_dir, '--model-type-id', 'mcp'])
            
            assert result.exit_code == 0
            
            # Check that MCP-specific content is in model.py
            with open(os.path.join(temp_dir, "1", "model.py"), 'r') as f:
                content = f.read()
                assert "MCPModelClass" in content
                assert "FastMCP" in content
            
            # Check that requirements.txt includes fastmcp
            with open(os.path.join(temp_dir, "requirements.txt"), 'r') as f:
                content = f.read()
                assert "fastmcp" in content

    def test_init_with_model_type_openai(self):
        """Test model initialization with OpenAI model type."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['model', 'init', temp_dir, '--model-type-id', 'openai'])
            
            assert result.exit_code == 0
            
            # Check that OpenAI-specific content is in model.py
            with open(os.path.join(temp_dir, "1", "model.py"), 'r') as f:
                content = f.read()
                assert "OpenAIModelClass" in content
                assert "OpenAI" in content
            
            # Check that requirements.txt includes openai
            with open(os.path.join(temp_dir, "requirements.txt"), 'r') as f:
                content = f.read()
                assert "openai" in content

    def test_init_with_local_repo(self):
        """Test model initialization with local repository."""
        runner = CliRunner()
        
        # Create a test repository
        with tempfile.TemporaryDirectory() as test_repo:
            # Create model structure
            os.makedirs(os.path.join(test_repo, "1"))
            
            # Create test files
            with open(os.path.join(test_repo, "1", "model.py"), 'w') as f:
                f.write('# Test model from repo')
            
            with open(os.path.join(test_repo, "config.yaml"), 'w') as f:
                f.write('model:\n  id: "test-model"')
            
            with open(os.path.join(test_repo, "requirements.txt"), 'w') as f:
                f.write('test-requirement>=1.0.0')
            
            # Test initialization from local repo
            with tempfile.TemporaryDirectory() as target_dir:
                # Mock pip install to avoid actual installation
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                    
                    result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', test_repo])
                    
                    assert result.exit_code == 0
                    
                    # Check that files were copied
                    assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
                    assert os.path.exists(os.path.join(target_dir, "config.yaml"))
                    assert os.path.exists(os.path.join(target_dir, "requirements.txt"))
                    
                    # Check content
                    with open(os.path.join(target_dir, "1", "model.py"), 'r') as f:
                        assert "Test model from repo" in f.read()

    @patch('subprocess.run')
    def test_init_with_github_repo_url(self, mock_run):
        """Test model initialization with GitHub URL."""
        runner = CliRunner()
        
        # Mock git clone
        def mock_subprocess(cmd, **kwargs):
            if cmd[0] == 'git' and cmd[1] == 'clone':
                # Create a fake cloned repository
                target_dir = cmd[-1]
                os.makedirs(os.path.join(target_dir, "1"))
                
                with open(os.path.join(target_dir, "1", "model.py"), 'w') as f:
                    f.write('# Test model from GitHub')
                
                with open(os.path.join(target_dir, "config.yaml"), 'w') as f:
                    f.write('model:\n  id: "github-model"')
                
                with open(os.path.join(target_dir, "requirements.txt"), 'w') as f:
                    f.write('github-requirement>=1.0.0')
                
                return MagicMock(returncode=0, stdout="", stderr="")
            elif cmd[0] == 'python' and 'pip' in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = mock_subprocess
        
        with tempfile.TemporaryDirectory() as target_dir:
            result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', 'user/repo'])
            
            assert result.exit_code == 0
            # Check that files were created from the GitHub repo
            assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(target_dir, "config.yaml"))
            
            # Check content
            with open(os.path.join(target_dir, "1", "model.py"), 'r') as f:
                assert "Test model from GitHub" in f.read()

    @patch('subprocess.run')
    def test_init_with_github_repo_with_pat(self, mock_run):
        """Test model initialization with GitHub URL and PAT."""
        runner = CliRunner()
        
        # Mock git clone with PAT
        def mock_subprocess(cmd, **kwargs):
            if cmd[0] == 'git' and cmd[1] == 'clone':
                # Verify PAT is in the URL
                assert 'test_pat@github.com' in cmd[2]
                
                # Create a fake cloned repository
                target_dir = cmd[-1]
                os.makedirs(os.path.join(target_dir, "1"))
                
                with open(os.path.join(target_dir, "1", "model.py"), 'w') as f:
                    f.write('# Test model from private GitHub')
                
                with open(os.path.join(target_dir, "config.yaml"), 'w') as f:
                    f.write('model:\n  id: "private-model"')
                
                return MagicMock(returncode=0, stdout="", stderr="")
            elif cmd[0] == 'python' and 'pip' in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = mock_subprocess
        
        with tempfile.TemporaryDirectory() as target_dir:
            result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', 'user/private-repo', '--pat', 'test_pat'])
            
            assert result.exit_code == 0
            # Check that files were created
            assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(target_dir, "config.yaml"))

    @patch('subprocess.run')
    def test_init_fallback_on_clone_failure(self, mock_run):
        """Test fallback to templates when repository cloning fails."""
        runner = CliRunner()
        
        # Mock failed git clone
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Repository not found")
        
        with tempfile.TemporaryDirectory() as target_dir:
            result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', 'nonexistent/repo'])
            
            assert result.exit_code == 0
            
            # Should still create template files (fallback)
            assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(target_dir, "config.yaml"))

    def test_init_fallback_on_no_model_structure(self):
        """Test fallback to templates when repository has no model structure."""
        runner = CliRunner()
        
        # Create a test repository without model structure
        with tempfile.TemporaryDirectory() as test_repo:
            # Create some files but not a model structure
            with open(os.path.join(test_repo, "README.md"), 'w') as f:
                f.write('# Test repo without model structure')
            
            with tempfile.TemporaryDirectory() as target_dir:
                result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', test_repo])
                
                assert result.exit_code == 0
                
                # Should create template files (fallback)
                assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
                assert os.path.exists(os.path.join(target_dir, "config.yaml"))

    def test_init_existing_files_warning(self):
        """Test that existing files generate warnings."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create existing files
            os.makedirs(os.path.join(temp_dir, "1"))
            with open(os.path.join(temp_dir, "1", "model.py"), 'w') as f:
                f.write('# Existing model')
            
            result = runner.invoke(cli, ['model', 'init', temp_dir])
            
            assert result.exit_code == 0
            # Verify that the existing file wasn't overwritten
            with open(os.path.join(temp_dir, "1", "model.py"), 'r') as f:
                assert "Existing model" in f.read()

    def test_help_shows_new_options(self):
        """Test that help shows the new PAT and GitHub repo options."""
        runner = CliRunner()
        result = runner.invoke(cli, ['model', 'init', '--help'])
        
        assert result.exit_code == 0
        assert "--pat" in result.output
        assert "--github-repo" in result.output
        assert "--branch" in result.output
        assert "Personal Access Token" in result.output
        assert "GitHub repository URL" in result.output
        assert "Git branch to clone from" in result.output

    @patch('subprocess.run')
    def test_init_with_github_repo_and_branch(self, mock_run):
        """Test model initialization with GitHub URL and specific branch."""
        runner = CliRunner()
        
        # Mock git clone with branch
        def mock_subprocess(cmd, **kwargs):
            if cmd[0] == 'git' and cmd[1] == 'clone':
                # Verify branch is specified in the command
                assert '-b' in cmd
                assert 'feature-branch' in cmd
                
                # Create a fake cloned repository
                target_dir = cmd[-1]
                os.makedirs(os.path.join(target_dir, "1"))
                
                with open(os.path.join(target_dir, "1", "model.py"), 'w') as f:
                    f.write('# Test model from GitHub feature branch')
                
                with open(os.path.join(target_dir, "config.yaml"), 'w') as f:
                    f.write('model:\n  id: "branch-model"')
                
                with open(os.path.join(target_dir, "requirements.txt"), 'w') as f:
                    f.write('branch-requirement>=1.0.0')
                
                return MagicMock(returncode=0, stdout="", stderr="")
            elif cmd[0] == 'python' and 'pip' in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = mock_subprocess
        
        with tempfile.TemporaryDirectory() as target_dir:
            result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', 'user/repo', '--branch', 'feature-branch'])
            
            assert result.exit_code == 0
            # Check that files were created from the GitHub repo
            assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(target_dir, "config.yaml"))
            
            # Check content
            with open(os.path.join(target_dir, "1", "model.py"), 'r') as f:
                assert "Test model from GitHub feature branch" in f.read()

    @patch('subprocess.run')
    def test_init_with_github_repo_branch_and_pat(self, mock_run):
        """Test model initialization with GitHub URL, specific branch, and PAT."""
        runner = CliRunner()
        
        # Mock git clone with branch and PAT
        def mock_subprocess(cmd, **kwargs):
            if cmd[0] == 'git' and cmd[1] == 'clone':
                # Verify branch is specified in the command
                assert '-b' in cmd
                assert 'dev-branch' in cmd
                # Verify PAT is in the URL
                assert 'test_pat@github.com' in cmd[4]  # URL should be at index 4 after -b branch
                
                # Create a fake cloned repository
                target_dir = cmd[-1]
                os.makedirs(os.path.join(target_dir, "1"))
                
                with open(os.path.join(target_dir, "1", "model.py"), 'w') as f:
                    f.write('# Test model from private GitHub dev branch')
                
                with open(os.path.join(target_dir, "config.yaml"), 'w') as f:
                    f.write('model:\n  id: "private-branch-model"')
                
                return MagicMock(returncode=0, stdout="", stderr="")
            elif cmd[0] == 'python' and 'pip' in cmd:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="Command not found")
        
        mock_run.side_effect = mock_subprocess
        
        with tempfile.TemporaryDirectory() as target_dir:
            result = runner.invoke(cli, ['model', 'init', target_dir, '--github-repo', 'user/private-repo', '--branch', 'dev-branch', '--pat', 'test_pat'])
            
            assert result.exit_code == 0
            # Check that files were created
            assert os.path.exists(os.path.join(target_dir, "1", "model.py"))
            assert os.path.exists(os.path.join(target_dir, "config.yaml"))