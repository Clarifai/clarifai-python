"""Tests for logging configuration changes."""

import logging
import os
from unittest import mock


class TestLoggingConfiguration:
    """Test cases for logging level configuration."""

    def test_default_log_level_is_warning(self):
        """Test that the default log level is WARNING (not INFO)."""
        # Clear any LOG_LEVEL env var for this test
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh logger with default settings
            import importlib

            from clarifai.utils import logging as logging_module

            importlib.reload(logging_module)

            # Check that the default logger level is WARNING
            logger = logging_module.logger
            assert logger.level == logging.WARNING

    def test_log_level_env_var_override(self):
        """Test that LOG_LEVEL environment variable overrides default."""
        test_cases = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
        ]

        for env_value, expected_level in test_cases:
            with mock.patch.dict(os.environ, {'LOG_LEVEL': env_value}):
                # Re-import to get fresh logger with new env var
                import importlib

                from clarifai.utils import logging as logging_module

                importlib.reload(logging_module)

                logger = logging_module.logger
                assert logger.level == expected_level, (
                    f"Expected {expected_level} for LOG_LEVEL={env_value}, got {logger.level}"
                )

    def test_info_logs_not_shown_by_default(self):
        """Test that INFO level logs are not displayed with default WARNING level."""
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib
            from io import StringIO

            from clarifai.utils import logging as logging_module

            importlib.reload(logging_module)

            logger = logging_module.logger

            # Capture log output
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger.addHandler(handler)

            # Try to log at INFO level
            logger.info("This INFO message should not appear")

            # Log at WARNING level (should appear)
            logger.warning("This WARNING message should appear")

            log_output = log_capture.getvalue()

            # INFO should not be in output
            assert "This INFO message should not appear" not in log_output
            # WARNING should be in output
            assert "This WARNING message should appear" in log_output

            logger.removeHandler(handler)

    def test_debug_logs_shown_with_debug_level(self):
        """Test that DEBUG logs are shown when LOG_LEVEL=DEBUG."""
        with mock.patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}):
            import importlib
            from io import StringIO

            from clarifai.utils import logging as logging_module

            importlib.reload(logging_module)

            logger = logging_module.logger

            # Capture log output
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger.addHandler(handler)

            # Log at DEBUG level
            logger.debug("This DEBUG message should appear")

            log_output = log_capture.getvalue()

            # DEBUG should be in output
            assert "This DEBUG message should appear" in log_output

            logger.removeHandler(handler)

    def test_cli_uses_clean_output_by_default(self):
        """Test that CLI commands have clean output without verbose logs."""
        from click.testing import CliRunner

        from clarifai.cli.base import cli

        runner = CliRunner()

        # Test a simple command that would normally produce INFO logs
        with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}, clear=False):
            result = runner.invoke(cli, ['--help'])

        # Help output should not contain log formatting
        assert '[INFO]' not in result.output
        assert 'thread=' not in result.output
        # Should just show clean help text
        assert 'Usage:' in result.output
