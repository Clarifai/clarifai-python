"""Tests for CLI utility functions."""

import sys
from io import StringIO
from unittest import mock

import pytest

from clarifai.utils.cli import masked_input


@pytest.fixture
def unix_terminal():
    """Mock a Unix terminal environment for masked_input testing."""

    def _run(input_chars):
        mock_stdin_read = mock.Mock(side_effect=input_chars)
        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            mock_stdin.isatty.return_value = True
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            return masked_input('Enter password: ')

    return _run


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific tests")
class TestMaskedInput:
    """Test cases for the masked_input function."""

    def test_normal_input(self, unix_terminal):
        assert unix_terminal(list("test123") + ['\n']) == "test123"

    def test_empty_input(self, unix_terminal):
        assert unix_terminal(['\n']) == ""

    def test_backspace(self, unix_terminal):
        # Type "test", backspace, "ing" -> "tesing"
        assert unix_terminal(list("test") + ['\x7f'] + list("ing") + ['\n']) == "tesing"

    def test_ctrl_u_clears_line(self, unix_terminal):
        # Type "wrong", Ctrl+U, "correct" -> "correct"
        assert unix_terminal(list("wrong") + ['\x15'] + list("correct") + ['\n']) == "correct"

    def test_ctrl_w_deletes_word(self, unix_terminal):
        # Type "hello world", Ctrl+W, "test" -> "hello test"
        assert (
            unix_terminal(list("hello world") + ['\x17'] + list("test") + ['\n']) == "hello test"
        )

    def test_ctrl_c_raises_keyboard_interrupt(self, unix_terminal):
        with pytest.raises(KeyboardInterrupt):
            unix_terminal(list("test") + ['\x03'])

    def test_piped_stdin_fallback(self):
        """When stdin is not a tty, reads a plain line instead."""
        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.isatty.return_value = False
            mock_stdin.readline.return_value = 'piped_token\n'
            with mock.patch('sys.stdout', new=StringIO()):
                result = masked_input('Enter password: ')

        assert result == 'piped_token'
