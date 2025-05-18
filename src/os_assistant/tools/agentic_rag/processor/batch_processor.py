import json
from collections.abc import Generator, Iterator
from typing import Any

from tracer.config import LogDomain
from tracer.store.log_reader import LogReader

from ..core.chunking import chunk_logs
from ..core.embedding import EmbeddingGenerator
from ..database.database import LogDatabase


def read_logs_in_batches(
    domain: LogDomain = LogDomain.FS,
    batch_size: int = 5,
    start_time=None,
    end_time=None,
) -> Generator[list[dict[str, Any]], None, None]:
    """
    Reads logs from the LogReader in batches and yields each batch.

    Args:
        domain: LogDomain to read from
        batch_size: Number of logs to include in each batch
        start_time: Optional start time filter
        end_time: Optional end time filter

    Yields:
        Lists of log dictionaries, each containing a batch of logs
    """
    reader = LogReader(domain)
    batch = []

    print(f"Reading logs from {domain.name} in batches of {batch_size}...")
    if start_time:
        print(f"Starting from timestamp: {start_time}")

    for event in reader.read_logs_iter(start_time, end_time):
        # Convert event to the format expected by the chunking module
        log_entry = {
            "timestamp": event.get("timestamp", ""),
            "log": json.dumps(event),  # Convert the entire event to a JSON string
        }

        batch.append(log_entry)

        # When we reach batch_size, yield the batch and reset
        if len(batch) >= batch_size:
            print(f"Yielding batch of {len(batch)} logs...")
            yield batch
            batch = []

    # Don't forget to yield the last partial batch if it exists
    if batch:
        print(f"Yielding final batch of {len(batch)} logs...")
        yield batch


def process_log_batches(
    batch_iterator: Iterator[list[dict[str, Any]]],
    db: LogDatabase,
    chunk_size: int,
    overlap: float,
) -> None:
    """
    Process batches of logs through chunking, embedding, and DB insertion.

    Args:
        batch_iterator: Iterator that yields batches of logs
        db: LogDatabase instance to store the processed logs
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks as a fraction (0-1)
    """
    embedding_generator = EmbeddingGenerator()
    current_log_number = db.get_highest_log_number() + 1

    print(f"Starting batch processing with log number {current_log_number}...")

    for batch_idx, batch in enumerate(batch_iterator):
        print(f"Processing batch {batch_idx + 1} with {len(batch)} logs...")

        # Chunk the batch
        chunks = chunk_logs(
            batch, chunk_size, overlap, start_log_number=current_log_number
        )
        print(f"Created {len(chunks)} chunks for this batch.")

        # Generate embeddings for chunks
        chunks_with_embeddings = []
        for chunk in chunks:
            embedding = embedding_generator.get_embedding(chunk["chunk_text"])
            chunk["embedding"] = embedding
            chunks_with_embeddings.append(chunk)

        # Insert chunks into the database
        print(f"Storing batch {batch_idx + 1} in database...")
        db.bulk_insert_chunks(chunks_with_embeddings)

        # Update the current_log_number for the next batch
        current_log_number += len(batch)

        print(f"Completed processing batch {batch_idx + 1}.")


def streamline_log_processing(
    domain: LogDomain = LogDomain.FS,
    batch_size: int = 5,
    chunk_size: int = 256,
    overlap: float = 0.1,
    start_time=None,
    end_time=None,
    continue_from_last: bool = True,
) -> int:
    """
    Streamlined function to read logs in batches and process them end-to-end.

    Args:
        domain: LogDomain to read from
        batch_size: Number of logs per batch
        chunk_size: Size of text chunks in characters
        overlap: Overlap between chunks (0-1)
        start_time: Optional start time filter (ISO timestamp)
        end_time: Optional end time filter (ISO timestamp)
        continue_from_last: If True, continue from the last timestamp in DB

    Returns:
        Number of batches processed
    """
    db = LogDatabase(domain)

    # If continue_from_last is True and no explicit start_time, get next timestamp
    if continue_from_last and not start_time:
        next_timestamp = db.get_next_timestamp()
        if next_timestamp:
            start_time = next_timestamp
            print(f"Continuing from last timestamp: {start_time}")

    batch_iterator = read_logs_in_batches(domain, batch_size, start_time, end_time)

    # Count batches for reporting
    batch_count = 0

    # Create a wrapper iterator that counts batches
    def counting_iterator():
        nonlocal batch_count
        for batch in batch_iterator:
            batch_count += 1
            yield batch

    # Process the batches
    process_log_batches(counting_iterator(), db, chunk_size, overlap)

    return batch_count
