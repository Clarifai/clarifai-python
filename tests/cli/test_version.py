"""Test version functionality for the CLI."""
from click.testing import CliRunner

from clarifai import __version__
from clarifai.cli.base import cli


class TestVersionCommand:
    """Test cases for the version functionality."""

    def test_version_option(self):
        """Test that --version flag displays the correct version."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_in_help(self):
        """Test that --version appears in the help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert '--version' in result.output
