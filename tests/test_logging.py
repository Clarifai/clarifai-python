"""Tests for logging configuration changes."""

import logging
import os
from unittest import mock

import pytest


class TestLoggingConfiguration:
    """Test cases for logging level configuration."""

    def test_default_log_level_is_info(self):
        """Test that the default log level is INFO."""
        # Clear any LOG_LEVEL env var for this test
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh logger with default settings
            import importlib

            from clarifai.utils import logging as logging_module

            importlib.reload(logging_module)

            # Check that the default logger level is INFO
            logger = logging_module.logger
            assert logger.level == logging.INFO

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
                assert (
                    logger.level == expected_level
                ), f"Expected {expected_level} for LOG_LEVEL={env_value}, got {logger.level}"

    def test_info_logs_shown_by_default(self):
        """Test that INFO level logs are displayed with default INFO level."""
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

            # Log at INFO level (should appear)
            logger.info("This INFO message should appear")

            # Log at DEBUG level (should not appear)
            logger.debug("This DEBUG message should not appear")

            log_output = log_capture.getvalue()

            # INFO should be in output
            assert "This INFO message should appear" in log_output
            # DEBUG should not be in output
            assert "This DEBUG message should not appear" not in log_output

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

    def test_login_uses_clean_output(self):
        """Test that login command has clean output without verbose validation logs."""
        from click.testing import CliRunner
        from unittest import mock

        from clarifai.cli.base import cli

        runner = CliRunner()

        # Test login command - validation logs should be DEBUG level (not shown)
        with runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                    with mock.patch('clarifai.utils.cli.validate_context_auth'):
                        result = runner.invoke(cli, ['--config', './config.yaml', 'login'], input='testuser\ny\n')

        # Should not show validation debug logs
        assert 'Validating the Context Credentials' not in result.output
        assert 'Context is valid' not in result.output
        # Should show clean success message
        assert 'Success!' in result.output
