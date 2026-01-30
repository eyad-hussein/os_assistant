from collections.abc import Sequence

from tracer.tracer_core import TracerCore

from dagent.utils import LOGGER

from .command_parser import CommandParser
from .os_assistant import OSAssistant


def main(argv: Sequence[str] | None = None) -> None:
    args = CommandParser().parse_args(argv)
    LOGGER.debug(f"Parsed arguments: {args}")

    if args.command == "trace":
        tracer = TracerCore()
        if args.trace_command == "start":
            tracer.start_tracing(args.domain, args.dir)
        elif args.trace_command == "show":
            tracer.print_logs(args.domain, args.start, args.end)
        elif args.trace_command == "clear":
            tracer.clear_logs(args.domain)
    if args.command == "chat":
        assistant = OSAssistant()
        assistant.run()


if __name__ == "__main__":
    raise SystemExit(main(None))
