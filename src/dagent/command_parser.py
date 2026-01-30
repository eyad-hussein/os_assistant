import argparse
from collections.abc import Sequence

from tracer.config import LogDomain


class CommandParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="osassis", description="A device-aware Assistant"
        )
        self.subparsers = self.parser.add_subparsers(dest="command", required=True)

        # Subcommand: trace
        trace_parser = self.subparsers.add_parser(
            "trace", help="System tracing commands"
        )
        trace_subparsers = trace_parser.add_subparsers(
            dest="trace_command", required=True
        )

        # Trace subcommand: start
        start_parser = trace_subparsers.add_parser("start", help="Start tracing")
        CommandParser.add_domain_argument(
            start_parser,
            help="Domain to trace",
        )
        start_parser.add_argument(
            "-d",
            "--dir",
            dest="dir",
            metavar="DIR",
            required=False,
            help="Directory to watch (required for 'file_system' domain)",
        )

        # Trace subcommand: show
        show_parser = trace_subparsers.add_parser("show", help="Print logs")
        CommandParser.add_domain_argument(
            show_parser,
            help="Domain to print logs for",
        )
        show_parser.add_argument(
            "-s",
            "--start",
            metavar="START",
            default=None,
            help="Start time for filtering logs (ISO format or fuzzy: now, yesterday, or [number][s/m/h/d]).",
        )
        show_parser.add_argument(
            "-e",
            "--end",
            metavar="END",
            default=None,
            help="End time for filtering logs (ISO format or fuzzy: now, yesterday, or [number][s/m/h/d]).",
        )

        # Trace subcommand: clear
        clear_parser = trace_subparsers.add_parser("clear", help="Clear all logs")
        CommandParser.add_domain_argument(
            clear_parser,
            help="Domain to clear logs for",
        )

        # Subcommand: chat
        self.subparsers.add_parser("chat", help="OS Assistant chating commands")

    @staticmethod
    def add_domain_argument(parser: argparse.ArgumentParser, help: str) -> None:
        parser.add_argument(
            "domain",
            choices=[d.value for d in LogDomain],
            help=help,
        )

    def parse_args(self, argv: Sequence[str] | None) -> argparse.Namespace:
        return self.parser.parse_args(argv)
