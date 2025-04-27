import argparse
import os
import time
from typing import List, Dict, Any

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K
from src.chunking import load_logs, chunk_logs
from src.embedding import EmbeddingGenerator
from src.database import LogDatabase
from src.retrieval import LogRetriever
from src.agent_rag import summarize_logs

def initialize_database(log_file: str, chunk_size: int, overlap: float, force_rebuild: bool = False) -> LogDatabase:
    """Initialize the database with chunked logs and embeddings."""
    start_time = time.time()
    print(f"Loading logs from {log_file}...")
    logs = load_logs(log_file)
    print(f"Loaded {len(logs)} log entries.")
    
    db = LogDatabase()
    
    # Check if database is already initialized and we're not forcing a rebuild
    if db.is_initialized() and not force_rebuild:
        print("Database already contains logs. Checking for new entries...")
        highest_log_number = db.get_highest_log_number()
        print(f"Highest log number in database: {highest_log_number}")
        
        # Filter logs to only process new ones
        new_logs = []
        for i, log in enumerate(logs):
            log_number = i + 1  # Assuming logs are numbered starting from 1
            if log_number > highest_log_number:
                new_logs.append(log)
        
        if not new_logs:
            print("No new logs to process. Database is up to date.")
            return db
            
        print(f"Found {len(new_logs)} new logs to process.")
        logs = new_logs
        offset_log_number = highest_log_number
    else:
        if force_rebuild and db.is_initialized():
            print("Forced rebuild requested. Reprocessing all logs...")
        else:
            print("Initializing new database...")
        offset_log_number = 0
    
    print(f"Chunking logs with size {chunk_size} and overlap {overlap}...")
    chunks = chunk_logs(logs, chunk_size, overlap, start_log_number=offset_log_number+1)
    print(f"Created {len(chunks)} chunks.")
    
    embedding_generator = EmbeddingGenerator()
    
    print("Generating embeddings...")
    chunks_with_embeddings = []
    batch_size = 50  # Process in batches to avoid memory issues
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)} chunks)...")
        
        for chunk in batch:
            embedding = embedding_generator.get_embedding(chunk["chunk_text"])
            chunk["embedding"] = embedding
            chunks_with_embeddings.append(chunk)
            
        # Insert each batch immediately to keep memory usage low
        print(f"Inserting batch into database...")
        db.bulk_insert_chunks(chunks_with_embeddings)
        chunks_with_embeddings = []  # Clear for next batch
    
    elapsed_time = time.time() - start_time
    print(f"Database initialization complete in {elapsed_time:.2f} seconds.")
    return db

def search_logs(query: str, top_k: int, summarize: bool = False) -> None:
    """Search for logs similar to the query and optionally summarize them."""
    db = LogDatabase()
    if not db.is_initialized():
        print("Database not initialized. Please run 'python main.py init --log-file your_logs.json' first.")
        return
        
    embedding_generator = EmbeddingGenerator()
    retriever = LogRetriever(db, embedding_generator)
    
    print(f"Searching for: '{query}'")
    similar_chunks = retriever.find_similar_chunks(query, top_k)
    
    if not similar_chunks:
        print("No similar logs found.")
        return
    
    print(f"\nFound {len(similar_chunks)} similar chunks:")
    for i, chunk in enumerate(similar_chunks):
        print(f"\n[{i+1}] Log #{chunk['log_number']}, Chunk #{chunk['chunk_number']} (Similarity: {chunk['similarity']:.4f})")
        print(f"Text: {chunk['chunk_text']}")
    
    print("\nAggregating logs...")
    aggregated_logs = retriever.aggregate_logs(similar_chunks)
    
    print(f"\nTop {len(aggregated_logs)} logs:")
    for i, log in enumerate(aggregated_logs):
        print(f"\n[{i+1}] Log #{log['log_number']} (Similarity: {log['similarity']:.4f})")
        print(f"Timestamp: {log['timestamp']}")
        print(f"Matched chunks: {log['matched_chunks']}")
        print(f"Text: {log['log_text']}")
    
    # Generate summaries if requested
    if summarize and aggregated_logs:
        print("\nGenerating summaries...")
        summaries = summarize_logs(aggregated_logs)
        
        print("\nLog summaries:")
        for i, summary in enumerate(summaries):
            print(f"\n[{i+1}] Summary of Log #{aggregated_logs[i]['log_number']}:")
            print(f"{summary}")
        
        return aggregated_logs, summaries
    
    return aggregated_logs, None

def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description="Log Analysis RAG System - Semantic search for log files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command 
    init_parser = subparsers.add_parser(
        "init", 
        help="Initialize database with logs",
        description="Process log files and build the vector database for searching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    init_parser.add_argument(
        "--log-file", 
        type=str, 
        required=True, 
        metavar="PATH",
        help="Path to the log JSON file (REQUIRED)"
    )
    init_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=DEFAULT_CHUNK_SIZE,
        metavar="INT",
        help="Text chunk size in characters"
    )
    init_parser.add_argument(
        "--overlap", 
        type=float, 
        default=DEFAULT_CHUNK_OVERLAP,
        metavar="FLOAT",
        help="Chunk overlap as fraction (0.0-1.0)"
    )
    init_parser.add_argument(
        "--force", 
        action="store_true",
        help="Force rebuilding the database from scratch"
    )
    
    # Search command
    search_parser = subparsers.add_parser(
        "search", 
        help="Search logs by semantic similarity",
        description="Find logs similar to your query using semantic search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    search_parser.add_argument(
        "query", 
        type=str,
        metavar="QUERY",
        help="Search query text (REQUIRED)"
    )
    search_parser.add_argument(
        "--top-k", 
        type=int, 
        default=DEFAULT_TOP_K, 
        metavar="INT",
        help="Number of top results to return"
    )
    search_parser.add_argument(
        "--summarize", 
        action="store_true",
        help="Generate summaries for the retrieved logs"
    )
    
    # Parse and execute commands
    args = parser.parse_args()
    
    if args.command == "init":
        initialize_database(args.log_file, args.chunk_size, args.overlap, args.force)
    elif args.command == "search":
        search_logs(args.query, args.top_k, args.summarize)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
