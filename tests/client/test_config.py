import os
import tempfile
import unittest
from unittest.mock import patch

from clarifai.utils.config import Config, Context


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file for testing
        self.temp_config = tempfile.NamedTemporaryFile(delete=False)
        self.temp_config.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_PAT: "config_pat"
      CLARIFAI_UI: "config_ui"
      api_base: "config_api_base"
      pat: "direct_pat"
      ui: "direct_ui"
""")
        self.temp_config.close()

    def tearDown(self):
        # Clean up the temporary config file
        os.unlink(self.temp_config.name)

    def test_config_from_yaml(self):
        config = Config.from_yaml(self.temp_config.name)
        self.assertEqual(config.current_context, "test_context")
        self.assertIn("test_context", config.contexts)

    @patch.dict('os.environ', {"CLARIFAI_PAT": "env_pat", "CLARIFAI_UI": "env_ui"})
    def test_env_var_override(self):
        # Set environment variables that should override config values
        config = Config.from_yaml(self.temp_config.name)
        context = config.current

        # Check that environment variables override config values
        self.assertEqual(context.pat, "env_pat")
        self.assertEqual(context.ui, "env_ui")

    @patch.dict('os.environ', {}, clear=True)
    def test_direct_access(self):
        config = Config.from_yaml(self.temp_config.name)
        context = config.current

        # Test direct access to config values
        self.assertEqual(context.pat, "config_pat")
        self.assertEqual(context.ui, "config_ui")

    @patch.dict('os.environ', {"CLARIFAI_PAT": "env_pat"})
    def test_envvar_fallback(self):
        # Create a config with ENVVAR placeholder
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_PAT: "ENVVAR"
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Check that ENVVAR is replaced with the actual environment variable
        self.assertEqual(context.pat, "env_pat")

        # Clean up
        os.unlink(temp_file_path)

    def test_attribute_error(self):
        config = Config.from_yaml(self.temp_config.name)
        context = config.current

        # Test that accessing a non-existent attribute raises AttributeError
        with self.assertRaises(AttributeError):
            _ = context.nonexistent

    def test_set_to_env(self):
        # Create a config with a custom value
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_CUSTOM: "custom_value"
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Set the environment variables from the config
        context.set_to_env()

        # Check that the environment variable was set correctly
        self.assertEqual(os.environ["CLARIFAI_CUSTOM"], "custom_value")

        # Clean up
        del os.environ["CLARIFAI_CUSTOM"]
        os.unlink(temp_file_path)

    @patch.dict('os.environ', {}, clear=True)
    def test_default_fallbacks(self):
        # Create a config with no UI or API_BASE values
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env: {}
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Check that default values are used
        self.assertEqual(context.ui, "https://clarifai.com")
        self.assertEqual(context.api_base, "https://api.clarifai.com")

        # Clean up
        os.unlink(temp_file_path)

    @patch.dict('os.environ', {}, clear=True)
    def test_context_creation(self):
        # Test creating a context directly
        context = Context("test_context", CLARIFAI_PAT="context_pat", CLARIFAI_UI="context_ui")

        self.assertEqual(context.name, "test_context")
        self.assertEqual(context.pat, "context_pat")
        self.assertEqual(context.ui, "context_ui")

    def test_env_var_name_access(self):
        # Test accessing config values using their environment variable names
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_PAT: "config_pat"
      CLARIFAI_UI: "config_ui"
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Test accessing values using environment variable names
        self.assertEqual(context.CLARIFAI_PAT, "config_pat")
        self.assertEqual(context.CLARIFAI_UI, "config_ui")

        # Clean up
        os.unlink(temp_file_path)

    @patch.dict('os.environ', clear=True)
    def test_shorthand_and_env_var_access(self):
        # Create a config with both shorthand and full environment variable names
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_PAT: "config_pat"
      CLARIFAI_UI: "config_ui"
      pat: "direct_pat"
      ui: "direct_ui"
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Test accessing values using shorthand names
        # When using shorthand, it first looks for CLARIFAI_<KEY> in config, then <key> in config
        self.assertEqual(context.pat, "config_pat")  # Finds CLARIFAI_PAT in config
        self.assertEqual(context.ui, "config_ui")  # Finds CLARIFAI_UI in config

        # Test accessing values using environment variable names
        self.assertEqual(context.CLARIFAI_PAT, "config_pat")
        self.assertEqual(context.CLARIFAI_UI, "config_ui")

        # Clean up
        os.unlink(temp_file_path)

    @patch.dict('os.environ', {"CLARIFAI_PAT": "env_pat", "CLARIFAI_UI": "env_ui"})
    def test_environment_variable_override(self):
        # Create a config file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"""
current_context: test_context
contexts:
  test_context:
    env:
      CLARIFAI_PAT: "config_pat"
      CLARIFAI_UI: "config_ui"
      pat: "direct_pat"
      ui: "direct_ui"
""")
            temp_file_path = temp_file.name

        config = Config.from_yaml(temp_file_path)
        context = config.current

        # Environment variables should take precedence when using shorthand access
        self.assertEqual(context.pat, "env_pat")  # From environment
        self.assertEqual(context.ui, "env_ui")  # From environment

        # When accessing through the environment variable name, it checks the config first
        self.assertEqual(context.CLARIFAI_PAT, "config_pat")  # From config
        self.assertEqual(context.CLARIFAI_UI, "config_ui")  # From config

        # Clean up
        os.unlink(temp_file_path)


if __name__ == "__main__":
    unittest.main()
