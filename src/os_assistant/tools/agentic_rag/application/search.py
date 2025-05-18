from typing import Any

from tracer.config import LogDomain

from ..config.config import DEFAULT_TOP_K
from ..core.agent_rag import summarize_logs
from ..core.embedding import EmbeddingGenerator
from ..core.retrieval import LogRetriever, MultiDomainRetriever
from ..database.database import LogDatabase
from .init import initialize_database


def parse_domains(domain_str: str) -> list[LogDomain]:
    """
    Parse a comma-separated list of domains into LogDomain objects.

    Args:
        domain_str: Comma-separated list of domain names

    Returns:
        List of LogDomain enum values
    """
    if not domain_str:
        return [LogDomain.FS]  # Default to FS

    domains = []
    for name in domain_str.split(","):
        try:
            domains.append(LogDomain(name.strip()))
        except KeyError:
            print(f"Warning: Unknown domain '{name}'. Skipping.")

    # Default to FS if no valid domains provided
    if not domains:
        domains = [LogDomain.FS]

    return domains


def search_logs(
    query: str,
    domains: list[LogDomain],
    top_k: int = DEFAULT_TOP_K,
    summarize: bool = False,
    auto_init: bool = True,
) -> tuple[list[dict[str, Any]], list[str] | None]:
    """
    Search for logs similar to the query and optionally summarize them.

    Args:
        query: The search query
        domains: List of domains to search in
        top_k: Number of top results to return
        summarize: Whether to generate summaries
        auto_init: Whether to automatically initialize DBs if needed

    Returns:
        Tuple of (aggregated_logs, summaries)
    """
    # Check if we need to automatically initialize any domains
    for domain in domains:
        db = LogDatabase(domain)
        print(f"Database for {domain.name} will be Auto-initializing...")
        initialize_database(log_domain=domain)

    # Create embedding generator to be shared across all retrievers
    embedding_generator = EmbeddingGenerator()

    # Use MultiDomainRetriever if multiple domains, otherwise use regular LogRetriever
    if len(domains) > 1:
        retriever = MultiDomainRetriever(domains, embedding_generator)
    else:
        db = LogDatabase(domains[0])
        retriever = LogRetriever(db, embedding_generator)

    print(f"Searching for: '{query}' across {', '.join([d.name for d in domains])}")
    similar_chunks = retriever.find_similar_chunks(query, top_k)

    if not similar_chunks:
        print("No similar logs found.")
        return [], None

    print(f"\nFound {len(similar_chunks)} similar chunks:")
    for i, chunk in enumerate(similar_chunks):
        domain_info = (
            f"Domain: {chunk.get('domain', 'unknown')}, " if len(domains) > 1 else ""
        )
        print(
            f"\n[{i + 1}] {domain_info}Log #{chunk['log_number']}, Chunk #{chunk['chunk_number']} (Similarity: {chunk['similarity']:.4f})"
        )
        print(f"Text: {chunk['chunk_text']}")

    print("\nAggregating logs...")
    aggregated_logs = retriever.aggregate_logs(similar_chunks)

    print(f"\nTop {len(aggregated_logs)} logs:")
    for i, log in enumerate(aggregated_logs):
        domain_info = (
            f"Domain: {log.get('domain', 'unknown')}, " if len(domains) > 1 else ""
        )
        print(
            f"\n[{i + 1}] {domain_info}Log #{log['log_number']} (Similarity: {log['similarity']:.4f})"
        )
        print(f"Timestamp: {log['timestamp']}")
        print(f"Matched chunks: {log['matched_chunks']}")
        print(f"Text: {log['log_text']}")

    # Generate summaries if requested
    if summarize and aggregated_logs:
        print("\nGenerating summaries...")
        summaries = summarize_logs(aggregated_logs)

        print("\nLog summaries:")
        for i, summary in enumerate(summaries):
            domain_info = (
                f"Domain: {aggregated_logs[i].get('domain', 'unknown')}, "
                if len(domains) > 1
                else ""
            )
            print(
                f"\n[{i + 1}] {domain_info}Summary of Log #{aggregated_logs[i]['log_number']}:"
            )
            print(f"{summary}")

        return aggregated_logs, summaries

    return aggregated_logs, None
