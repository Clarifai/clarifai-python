import subprocess
import sys
import unittest.mock
import pytest


class TestCLIVersionValidation:
    """Test cases for CLI Python version validation."""

    def test_cli_with_supported_version(self):
        """Test that CLI works with supported Python version."""
        # Import after ensuring we have a supported version
        from click.testing import CliRunner
        from clarifai.cli.base import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        # Should complete successfully and show help
        assert result.exit_code == 0
        assert "Clarifai CLI" in result.output

    def test_package_validation_function_behavior(self):
        """Test the core validation logic with mocked versions."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            with unittest.mock.patch('builtins.print') as mock_print:
                with unittest.mock.patch('sys.exit') as mock_exit:
                    # Set version to below minimum (Python 3.7)
                    mock_version.major = 3
                    mock_version.minor = 7
                    mock_version.micro = 0
                    
                    # Test the validation function directly
                    from clarifai.versions import validate_python_version
                    validate_python_version()
                    
                    # Check that print was called with error messages
                    assert mock_print.call_count == 3
                    calls = mock_print.call_args_list
                    
                    # Check error message content
                    assert "Error: Clarifai requires Python" in str(calls[0])
                    assert "You are currently using Python 3.7" in str(calls[1])
                    assert "Please upgrade your Python version" in str(calls[2])
                    
                    # Check sys.exit was called with code 1
                    mock_exit.assert_called_once_with(1)