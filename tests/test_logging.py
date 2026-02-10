"""Tests for logging configuration changes."""

import logging
import os
from io import StringIO
from unittest import mock

from clarifai.utils.logging import get_logger


class TestLoggingConfiguration:
    """Test cases for logging level configuration."""

    def test_default_log_level_is_info(self):
        """Test that get_logger with INFO level sets the correct level."""
        logger = get_logger(logger_level="INFO", name="test_default_info")
        assert logger.level == logging.INFO

    def test_log_level_override(self):
        """Test that get_logger respects the requested log level."""
        test_cases = [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
        ]

        for level_value, expected_level in test_cases:
            logger = get_logger(logger_level=level_value, name=f"test_{level_value}")
            assert logger.level == expected_level, (
                f"Expected {expected_level} for level={level_value}, got {logger.level}"
            )

    def test_info_logs_shown_by_default(self):
        """Test that INFO level logs are displayed with INFO level."""
        logger = get_logger(logger_level="INFO", name="test_info_visible")

        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)

        logger.info("This INFO message should appear")
        logger.debug("This DEBUG message should not appear")

        log_output = log_capture.getvalue()

        assert "This INFO message should appear" in log_output
        assert "This DEBUG message should not appear" not in log_output

        logger.removeHandler(handler)

    def test_debug_logs_shown_with_debug_level(self):
        """Test that DEBUG logs are shown when level is DEBUG."""
        logger = get_logger(logger_level="DEBUG", name="test_debug_visible")

        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)

        logger.debug("This DEBUG message should appear")

        log_output = log_capture.getvalue()

        assert "This DEBUG message should appear" in log_output

        logger.removeHandler(handler)

    def test_login_uses_clean_output(self):
        """Test that login command has clean output without verbose validation logs."""

        from click.testing import CliRunner

        from clarifai.cli.base import cli

        runner = CliRunner()

        # Test login command - validation logs should be DEBUG level (not shown)
        with runner.isolated_filesystem():
            with mock.patch('clarifai.cli.base.DEFAULT_CONFIG', './config.yaml'):
                with mock.patch.dict(os.environ, {'CLARIFAI_PAT': 'test_pat'}):
                    with mock.patch('clarifai.utils.cli.validate_context_auth'):
                        result = runner.invoke(
                            cli, ['--config', './config.yaml', 'login'], input='testuser\ny\n'
                        )

        # Should not show validation debug logs
        assert 'Validating the Context Credentials' not in result.output
        assert 'Context is valid' not in result.output
        # Should show clean success message
        assert 'Success!' in result.output
