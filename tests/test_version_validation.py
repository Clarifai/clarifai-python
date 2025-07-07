import sys
import unittest.mock
import pytest

from clarifai.versions import validate_python_version, MINIMUM_PYTHON_VERSION


class TestVersionValidation:
    """Test cases for Python version validation."""

    def test_validate_python_version_current_version_passes(self):
        """Test that current Python version passes validation."""
        # This should not raise any exception
        validate_python_version()

    def test_validate_python_version_minimum_version_passes(self):
        """Test that minimum supported version passes validation."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            # Set version to exactly the minimum supported version
            mock_version.major = MINIMUM_PYTHON_VERSION[0]
            mock_version.minor = MINIMUM_PYTHON_VERSION[1]
            mock_version.micro = 0
            
            # This should not raise any exception
            validate_python_version()

    def test_validate_python_version_below_minimum_exits(self):
        """Test that version below minimum causes system exit."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            with unittest.mock.patch('sys.stderr') as mock_stderr:
                # Set version to below minimum (Python 3.7)
                mock_version.major = 3
                mock_version.minor = 7
                mock_version.micro = 0
                
                # Should exit with code 1
                with pytest.raises(SystemExit) as excinfo:
                    validate_python_version()
                
                assert excinfo.value.code == 1

    def test_validate_python_version_error_message_format(self):
        """Test that error message has correct format."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            with unittest.mock.patch('builtins.print') as mock_print:
                with unittest.mock.patch('sys.exit') as mock_exit:
                    # Set version to below minimum (Python 3.6)
                    mock_version.major = 3
                    mock_version.minor = 6
                    mock_version.micro = 0
                    
                    validate_python_version()
                    
                    # Check that print was called with error messages
                    assert mock_print.call_count == 3
                    calls = mock_print.call_args_list
                    
                    # Check error message content
                    assert "Error: Clarifai requires Python" in str(calls[0])
                    assert "You are currently using Python 3.6" in str(calls[1])
                    assert "Please upgrade your Python version" in str(calls[2])
                    
                    # Check stderr was used
                    for call in calls:
                        assert call[1]['file'] == sys.stderr
                    
                    # Check sys.exit was called with code 1
                    mock_exit.assert_called_once_with(1)

    def test_validate_python_version_very_old_version(self):
        """Test validation with very old Python version."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            with unittest.mock.patch('sys.stderr') as mock_stderr:
                # Set version to very old (Python 2.7)
                mock_version.major = 2
                mock_version.minor = 7
                mock_version.micro = 0
                
                # Should exit with code 1
                with pytest.raises(SystemExit) as excinfo:
                    validate_python_version()
                
                assert excinfo.value.code == 1

    def test_validate_python_version_future_version_passes(self):
        """Test that future Python version passes validation."""
        with unittest.mock.patch('sys.version_info') as mock_version:
            # Set version to future version (Python 4.0)
            mock_version.major = 4
            mock_version.minor = 0
            mock_version.micro = 0
            
            # This should not raise any exception
            validate_python_version()