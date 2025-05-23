import argparse
from .run_code import run_code_execution

# TODO: add folders to make the coding agent more organized.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code execution agent")
    parser.add_argument("question", type=str, help="The question or code to execute")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (no prompts for dangerous operations)",
    )

    args = parser.parse_args()
    run_code_execution(args.question, args.verbose, not args.non_interactive)
