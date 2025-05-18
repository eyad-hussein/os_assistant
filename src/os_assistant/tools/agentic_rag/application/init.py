import time

from tracer.config import LogDomain

from ..config.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from ..database.database import LogDatabase
from ..processor.batch_processor import streamline_log_processing


def initialize_database(
    log_domain: LogDomain,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: float = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 5,
    start_time: str = None,
    end_time: str = None,
    continue_from_last: bool = True,
) -> LogDatabase:
    """
    Initialize the database by streaming logs directly from the tracer.

    Args:
        log_domain: Domain to read logs from
        chunk_size: Text chunk size in characters
        overlap: Chunk overlap as fraction (0-1)
        batch_size: Number of logs to process in each batch
        start_time: Optional start time filter (ISO timestamp)
        end_time: Optional end time filter (ISO timestamp)
        continue_from_last: If True, continue from the last timestamp in DB
    """
    start_processing_time = time.time()

    print(f"Initializing database for {log_domain.name} domain...")

    # Process logs using batch processor
    batch_count = streamline_log_processing(
        domain=log_domain,
        batch_size=batch_size,
        chunk_size=chunk_size,
        overlap=overlap,
        start_time=start_time,
        end_time=end_time,
        continue_from_last=continue_from_last,
    )

    elapsed_time = time.time() - start_processing_time
    print(
        f"Database initialization complete in {elapsed_time:.2f} seconds. Processed {batch_count} batches."
    )

    return LogDatabase(log_domain)
