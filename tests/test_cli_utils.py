"""Tests for CLI utility functions."""

import sys
from io import StringIO
from unittest import mock

import pytest

from clarifai.utils.cli import masked_input


class TestMaskedInput:
    """Test cases for the masked_input function."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_normal_input_unix(self):
        """Test that normal input returns the correct password on Unix."""
        test_password = "test123"
        mock_stdin_read = mock.Mock(side_effect=list(test_password) + ['\n'])

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()) as mock_stdout:
                            result = masked_input('Enter password: ')

        assert result == test_password

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_backspace_unix(self):
        """Test that backspace deletes characters on Unix."""
        # Simulate typing "test", backspace, "ing"
        # Result should be "tesing" (backspace removes the 't' at position 4)
        inputs = list("test") + ['\x7f'] + list("ing") + ['\n']
        mock_stdin_read = mock.Mock(side_effect=inputs)

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            result = masked_input('Enter password: ')

        assert result == "tesing"

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_ctrl_u_clears_line_unix(self):
        """Test that Ctrl+U clears the entire line on Unix."""
        # Simulate typing "wrong", Ctrl+U, "correct"
        inputs = list("wrong") + ['\x15'] + list("correct") + ['\n']
        mock_stdin_read = mock.Mock(side_effect=inputs)

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            result = masked_input('Enter password: ')

        assert result == "correct"

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_ctrl_w_deletes_word_unix(self):
        """Test that Ctrl+W deletes the last word on Unix."""
        # Simulate typing "hello world", Ctrl+W, "test"
        inputs = list("hello world") + ['\x17'] + list("test") + ['\n']
        mock_stdin_read = mock.Mock(side_effect=inputs)

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            result = masked_input('Enter password: ')

        assert result == "hello test"

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_ctrl_c_raises_keyboard_interrupt_unix(self):
        """Test that Ctrl+C raises KeyboardInterrupt on Unix."""
        inputs = list("test") + ['\x03']
        mock_stdin_read = mock.Mock(side_effect=inputs)

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            with pytest.raises(KeyboardInterrupt):
                                masked_input('Enter password: ')

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_masked_input_empty_input_unix(self):
        """Test that empty input (just Enter) returns empty string on Unix."""
        mock_stdin_read = mock.Mock(side_effect=['\n'])

        with mock.patch('sys.stdin') as mock_stdin:
            mock_stdin.read = mock_stdin_read
            mock_stdin.fileno.return_value = 0
            with mock.patch('termios.tcgetattr', return_value=[]):
                with mock.patch('termios.tcsetattr'):
                    with mock.patch('tty.setraw'):
                        with mock.patch('sys.stdout', new=StringIO()):
                            result = masked_input('Enter password: ')

        assert result == ""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_masked_input_normal_input_windows(self):
        """Test that normal input returns the correct password on Windows."""
        test_password = "test123"
        inputs = [bytes([c]) for c in test_password.encode()] + [b'\r']

        with mock.patch('os.name', 'nt'):
            with mock.patch('sys.stdin') as mock_stdin:
                mock_stdin.isatty.return_value = True
                with mock.patch('msvcrt.getch', side_effect=inputs):
                    with mock.patch('sys.stdout', new=StringIO()):
                        result = masked_input('Enter password: ')

        assert result == test_password

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_masked_input_ctrl_u_clears_line_windows(self):
        """Test that Ctrl+U clears the entire line on Windows."""
        inputs = [b'w', b'r', b'o', b'n', b'g', b'\x15', b'o', b'k', b'\r']

        with mock.patch('os.name', 'nt'):
            with mock.patch('sys.stdin') as mock_stdin:
                mock_stdin.isatty.return_value = True
                with mock.patch('msvcrt.getch', side_effect=inputs):
                    with mock.patch('sys.stdout', new=StringIO()):
                        result = masked_input('Enter password: ')

        assert result == "ok"

    @pytest.mark.skipif(sys.platform == "win32", reason="Requires termios (Unix-only)")
    def test_masked_input_custom_mask(self):
        """Test that custom mask character is used."""
        with mock.patch('os.name', 'posix'):
            with mock.patch('sys.stdin') as mock_stdin:
                mock_stdin.isatty.return_value = True
                mock_stdin.read = mock.Mock(side_effect=['\n'])
                mock_stdin.fileno.return_value = 0
                with mock.patch('termios.tcgetattr', return_value=[]):
                    with mock.patch('termios.tcsetattr'):
                        with mock.patch('tty.setraw'):
                            with mock.patch('sys.stdout', new=StringIO()):
                                result = masked_input('Enter password: ', mask='#')
                                assert result == ""
