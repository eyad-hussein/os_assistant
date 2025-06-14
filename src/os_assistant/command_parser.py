import argparse

from tracer.config import LogDomain
from tracer.tracer_core import TracerCore

from .os_assistant import OSAssistant


class CommandParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="osassis", description="OS Assistant CLI tool"
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
        start_parser.add_argument(
            "domain",
            choices=[d.value for d in LogDomain],
            help="Domain to print logs for",
        )
        start_parser.add_argument(
            "-d",
            "--dir",
            metavar="DIR",
            required=False,
            help="Directory to watch (required for 'file_system' domain)",
        )

        # Trace subcommand: show
        show_parser = trace_subparsers.add_parser("show", help="Print logs")
        show_parser.add_argument(
            "domain",
            choices=[d.value for d in LogDomain],
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
        clear_parser.add_argument(
            "domain",
            choices=[d.value for d in LogDomain],
            help="Domain to clear logs for",
        )

        # Subcommand: chat
        self.subparsers.add_parser("chat", help="OS Assistant chating commands")

    def parse_args(self):
        args = self.parser.parse_args()
        tracer = TracerCore()
        if args.command == "trace":
            if args.trace_command == "start":
                tracer.start_tracing(args.domain, args.dir)
            elif args.trace_command == "show":
                tracer.print_logs(args.domain, args.start, args.end)
            elif args.trace_command == "clear":
                # FIXME: tracer.clear_logs is not implemented
                tracer.clear_logs(args.domain)
        if args.command == "chat":
            assistant = OSAssistant()
            assistant.run()
