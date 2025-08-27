import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from clarifai.runners.models.model_builder import ModelBuilder


class TestGitInfo(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

        # Create a basic model structure
        self.model_dir = os.path.join(self.test_dir, "test_model")
        os.makedirs(self.model_dir)

        # Create minimal config.yaml
        config_content = """
model:
  id: test-model
  app_id: test-app
  user_id: test-user
  model_type_id: text-to-text

build_info:
  python_version: "3.12"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  num_accelerators: 0
"""
        with open(os.path.join(self.model_dir, "config.yaml"), "w") as f:
            f.write(config_content)

        # Create required "1" subdirectory and model.py
        os.makedirs(os.path.join(self.model_dir, "1"))
        with open(os.path.join(self.model_dir, "1", "model.py"), "w") as f:
            f.write("""
from clarifai.runners.models.model_class import ModelClass

class TestModel(ModelClass):
    def load_model(self):
        pass

    def predict(self, inputs):
        return []
""")

        # Create requirements.txt
        with open(os.path.join(self.model_dir, "requirements.txt"), "w") as f:
            f.write("numpy\n")

    def test_get_git_info_non_git_folder(self):
        """Test _get_git_info on a non-git folder"""
        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        git_info = builder._get_git_info()
        self.assertIsNone(git_info)

    def test_get_git_info_git_folder(self):
        """Test _get_git_info on a git repository"""
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(['git', 'add', '.'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'remote', 'add', 'origin', 'https://github.com/test/repo.git'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )

        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        git_info = builder._get_git_info()

        self.assertIsNotNone(git_info)
        self.assertIn('commit', git_info)
        self.assertIn('branch', git_info)
        self.assertIn('url', git_info)
        self.assertEqual(git_info['url'], 'https://github.com/test/repo.git')
        self.assertEqual(len(git_info['commit']), 40)  # Git commit hash length

    def test_check_git_status_clean_repo(self):
        """Test _check_git_status_and_prompt with clean repository"""
        # Initialize git repo with clean state
        subprocess.run(['git', 'init'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(['git', 'add', '.'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )

        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        result = builder._check_git_status_and_prompt()
        self.assertTrue(result)

    @patch('builtins.input', return_value='y')
    def test_check_git_status_dirty_repo_accept(self, mock_input):
        """Test _check_git_status_and_prompt with uncommitted changes - user accepts"""
        # Initialize git repo with uncommitted changes
        subprocess.run(['git', 'init'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(['git', 'add', '.'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )

        # Add uncommitted changes
        with open(os.path.join(self.model_dir, "uncommitted.txt"), "w") as f:
            f.write("uncommitted content")

        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        result = builder._check_git_status_and_prompt()
        self.assertTrue(result)
        mock_input.assert_called_once()

    @patch('builtins.input', return_value='n')
    def test_check_git_status_dirty_repo_decline(self, mock_input):
        """Test _check_git_status_and_prompt with uncommitted changes - user declines"""
        # Initialize git repo with uncommitted changes
        subprocess.run(['git', 'init'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(['git', 'add', '.'], cwd=self.model_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=self.model_dir,
            capture_output=True,
            check=False,
        )

        # Add uncommitted changes
        with open(os.path.join(self.model_dir, "uncommitted.txt"), "w") as f:
            f.write("uncommitted content")

        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        result = builder._check_git_status_and_prompt()
        self.assertFalse(result)
        mock_input.assert_called_once()

    def test_get_model_version_proto_with_git_info(self):
        """Test that git info is properly added to model version proto"""
        builder = ModelBuilder(self.model_dir, download_validation_only=True)

        git_info = {
            'url': 'https://github.com/test/repo.git',
            'commit': 'abc123def456',
            'branch': 'main',
        }

        proto = builder.get_model_version_proto(git_info)

        # Check that metadata contains git_registry information
        self.assertTrue(proto.metadata)
        metadata_dict = dict(proto.metadata)
        self.assertIn('git_registry', metadata_dict)

        git_registry = metadata_dict['git_registry']
        self.assertEqual(git_registry['url'], 'https://github.com/test/repo.git')
        self.assertEqual(git_registry['commit'], 'abc123def456')
        self.assertEqual(git_registry['branch'], 'main')

    def test_get_model_version_proto_without_git_info(self):
        """Test that proto works normally without git info"""
        builder = ModelBuilder(self.model_dir, download_validation_only=True)

        proto = builder.get_model_version_proto()

        # Metadata should be empty or not contain git_registry
        metadata_dict = dict(proto.metadata) if proto.metadata else {}
        self.assertNotIn('git_registry', metadata_dict)

    def test_git_status_limited_to_model_path(self):
        """Test that git status checking is limited to model path only"""
        # Create a parent directory for the git repo
        parent_dir = os.path.join(self.test_dir, "git_repo")
        os.makedirs(parent_dir)

        # Initialize git repo in parent directory
        subprocess.run(['git', 'init'], cwd=parent_dir, capture_output=True, check=True)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'],
            cwd=parent_dir,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ['git', 'config', 'user.name', 'Test User'],
            cwd=parent_dir,
            capture_output=True,
            check=False,
        )

        # Move model to be inside the git repo
        model_in_git = os.path.join(parent_dir, "my_model")
        shutil.move(self.model_dir, model_in_git)
        self.model_dir = model_in_git

        # Commit all files
        subprocess.run(['git', 'add', '.'], cwd=parent_dir, capture_output=True, check=False)
        subprocess.run(
            ['git', 'commit', '-m', 'Initial commit'],
            cwd=parent_dir,
            capture_output=True,
            check=False,
        )

        # Add uncommitted file OUTSIDE the model directory
        outside_file = os.path.join(parent_dir, "outside_model.txt")
        with open(outside_file, "w") as f:
            f.write("This file is outside the model directory")

        # Test that ModelBuilder ignores files outside model path
        builder = ModelBuilder(self.model_dir, download_validation_only=True)
        result = builder._check_git_status_and_prompt()

        # Should return True because no uncommitted changes within model path
        self.assertTrue(result)

        # Now add uncommitted file inside model directory
        inside_file = os.path.join(self.model_dir, "inside_model.txt")
        with open(inside_file, "w") as f:
            f.write("This file is inside the model directory")

        # Mock user declining the prompt
        with patch('builtins.input', return_value='n'):
            result = builder._check_git_status_and_prompt()
            self.assertFalse(result)

        # Mock user accepting the prompt
        with patch('builtins.input', return_value='y'):
            result = builder._check_git_status_and_prompt()
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
