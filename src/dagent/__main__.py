from collections.abc import Sequence

from tracer.tracer_core import TracerCore

from dagent.utils import LOGGER

from .command_parser import CommandParser
from .os_assistant import OSAssistant


def main(argv: Sequence[str] | None = None) -> None:
    args = CommandParser().parse_args(argv)
    LOGGER.debug(f"Parsed arguments: {args}")
    match args.command:
        case "trace":
            tracer = TracerCore()
            match args.trace_command:
                case "start":
                    tracer.start_tracing(args.domain, args.dir)
                case "show":
                    tracer.print_logs(args.domain, args.start, args.end)
                case "clear":
                    tracer.clear_logs(args.domain)
        case "chat":
            assistant = OSAssistant()
            assistant.run()


if __name__ == "__main__":
    raise SystemExit(main(None))
