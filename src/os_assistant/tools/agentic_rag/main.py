import argparse
import sys

from agentic_rag.application.init import initialize_database
from agentic_rag.application.search import parse_domains, search_logs
from agentic_rag.config.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
)


def create_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG - Search and analyze system logs with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize filesystem logs database:
    python main.py init --domain file_system
    
  Search for file read operations across all domains:
    python main.py search "file read operation" --domain file_system,process --summarize
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the log database")
    init_parser.add_argument(
        "--domain",
        type=str,
        default="file_system",
        help="Domain(s) to initialize (comma-separated). Options: file_system, networking",
    )
    init_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Size of text chunks in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    init_parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks as a fraction (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    init_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of logs to process in each batch (default: 5)",
    )
    init_parser.add_argument(
        "--start-time",
        type=str,
        help="Start time filter in ISO format (YYYY-MM-DDTHH:MM:SS.ffffff)",
    )
    init_parser.add_argument(
        "--end-time",
        type=str,
        help="End time filter in ISO format (YYYY-MM-DDTHH:MM:SS.ffffff)",
    )
    init_parser.add_argument(
        "--no-continue",
        action="store_false",
        dest="continue_from_last",
        help="Don't continue from last processed timestamp",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for relevant logs")
    search_parser.add_argument("query", type=str, help="Search query text")
    search_parser.add_argument(
        "--domain",
        type=str,
        default="file_system",
        help="Domain(s) to search in (comma-separated). Options: file_system, networking",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top results to return (default: {DEFAULT_TOP_K})",
    )
    search_parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate summaries of search results using AI",
    )
    search_parser.add_argument(
        "--no-auto-init",
        action="store_false",
        dest="auto_init",
        help="Don't automatically initialize databases if needed",
    )

    return parser


def handle_commands(args):
    """Handle the parsed command-line arguments."""
    if args.command == "init":
        # Parse domains for initialization
        domains = parse_domains(args.domain)

        # Initialize each specified domain
        for domain in domains:
            print(f"\n{'=' * 50}")
            print(f"Initializing {domain.name} domain...")
            print(f"{'=' * 50}\n")

            initialize_database(
                log_domain=domain,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                start_time=args.start_time,
                end_time=args.end_time,
                continue_from_last=args.continue_from_last,
            )

        print(f"\n{'=' * 50}")
        print(f"Initialization complete for {', '.join([d.name for d in domains])}")
        print(f"{'=' * 50}\n")

    elif args.command == "search":
        # Parse domains for searching
        domains = parse_domains(args.domain)

        print(f"\n{'=' * 50}")
        print(f"Searching for: '{args.query}'")
        print(f"Domains: {', '.join([d.name for d in domains])}")
        print(f"{'=' * 50}\n")

        # Perform search
        logs, summaries = search_logs(
            query=args.query,
            domains=domains,
            top_k=args.top_k,
            summarize=args.summarize,
            auto_init=args.auto_init,
        )

        # Display result count
        if logs:
            print(f"\n{'=' * 50}")
            print(f"Found {len(logs)} relevant logs")
            print(f"{'=' * 50}\n")
        else:
            print(f"\n{'=' * 50}")
            print("No relevant logs found")
            print(f"{'=' * 50}\n")
    else:
        # If no command provided, show help
        print("Error: No command specified. Use 'init' or 'search'.")
        print("\nFor more information, run: python main.py --help")
        sys.exit(1)


def main():
    """Main entry point for the Agentic RAG CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handle_commands(args)


if __name__ == "__main__":
    main()
