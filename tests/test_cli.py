"""Simple tests for CLI functionality."""

from unittest.mock import Mock, patch

import pytest

from os_assistant.__main__ import main
from os_assistant.command_parser import CommandParser


class TestCommandParser:
    """Test basic command parsing functionality."""

    def test_chat_command_parsing(self):
        """Test parsing of chat command."""
        parser = CommandParser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"

    def test_trace_start_command_parsing(self):
        """Test parsing of trace start command."""
        with patch("os_assistant.command_parser.LogDomain") as mock_log_domain:
            # Mock LogDomain to return some test domains
            mock_log_domain.__iter__.return_value = [
                Mock(value="file_system"),
                Mock(value="network"),
            ]

            parser = CommandParser()
            args = parser.parse_args(["trace", "start", "file_system", "--dir", "/tmp"])

            assert args.command == "trace"
            assert args.trace_command == "start"
            assert args.domain == "file_system"
            assert args.dir == "/tmp"

    def test_help_command_exits(self):
        parser = CommandParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_invalid_command_exits(self):
        parser = CommandParser()
        with pytest.raises(SystemExit):
            parser.parse_args(["invalid_command"])


class TestMainFunction:
    """Test main function basic functionality."""

    @patch("os_assistant.__main__.OSAssistant")
    def test_main_chat_command(self, mock_assistant_class):
        """Test main function handles chat command."""
        mock_assistant = Mock()
        mock_assistant_class.return_value = mock_assistant

        main(["chat"])

        mock_assistant_class.assert_called_once()
        mock_assistant.run.assert_called_once()

    def test_main_help_exits(self):
        with pytest.raises(SystemExit):
            main(["--help"])

    def test_main_invalid_command_exits(self):
        with pytest.raises(SystemExit):
            main(["invalid_command"])
