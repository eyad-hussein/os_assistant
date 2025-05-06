"""CLI module for the model training and evaluation."""

import argparse
from collections.abc import Sequence

from os_assistant.core.os_assistant.main import kickoff


def create_cli_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """A custom argument parser for the library

    Args:
        argv(Optional[Sequence[str]], optional): Command line arguments to parse. Default to None.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="os_assistant",
        description="An os_assistant cli tool that connects you to the OS in many ways!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    configure_parser(parser)

    args = parser.parse_args(argv)
    return args


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add arguments and subcommands to the parser.

    Args:
        parser (argparse.ArgumentParser): parser to add arguments to.

    """
    sub_parsers = parser.add_subparsers(dest="command")

    run_parser = sub_parsers.add_parser("run")
    run_parser.add_argument("prompt")

    trace_parser = sub_parsers.add_parser("trace")
    trace_parser.add_argument("start")


def main() -> None:
    args = create_cli_parser()

    if args.command == "run":
        kickoff()
    elif args.command == "trace":
        print("tracing")
    else:
        print("nothing")


if __name__ == "__main__":
    main()
