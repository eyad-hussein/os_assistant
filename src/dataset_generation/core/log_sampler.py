import random
from datetime import datetime
from typing import Any

from tracer.config import LogDomain

from dagent.tools.agentic_rag.database.database import LogDatabase

from ..config.config import (
    MAX_SEQUENTIAL_LOGS,
    MIN_SEQUENTIAL_LOGS,
    TIME_WINDOW_SECONDS,
)


class SequentialLogSampler:
    """Samples logs that are sequentially connected (by time and possibly context)"""

    def __init__(self, domain: LogDomain = LogDomain.FS):
        """Initialize with file system domain by default"""
        self.domain = domain
        self.db = None
        self.domains_dbs = {}

        # Initialize database for specific domain if provided
        if domain:
            try:
                self.db = LogDatabase(domain)
                if not self.db.is_initialized():
                    print(
                        f"[WARNING] Database for domain {domain.name} is not initialized or empty"
                    )
            except Exception as e:
                print(
                    f"[ERROR] Error initializing database for domain {domain}: {str(e)}"
                )

        # If no specific domain, initialize only FS domain
        else:
            try:
                db = LogDatabase(LogDomain.FS)
                if db.is_initialized():
                    self.domains_dbs[LogDomain.FS] = db
                    print(
                        f"Successfully initialized database for domain {LogDomain.FS.name}"
                    )
                else:
                    print(
                        f"[WARNING] Database for domain {LogDomain.FS.name} is not initialized or empty"
                    )
            except Exception as e:
                print(
                    f"[ERROR] Error initializing database for domain {LogDomain.FS}: {str(e)}"
                )

    def validate_db_access(
        self, domain: LogDomain | None = None
    ) -> tuple[bool, LogDatabase | None, str]:
        """
        Validate database access for the specified domain.

        Args:
            domain: Domain to validate

        Returns:
            Tuple of (success, database, error_message)
        """
        use_domain = domain or self.domain

        if use_domain:
            # Use the class db if domain matches
            if self.domain == use_domain and self.db:
                db = self.db
            # Otherwise create a new db connection
            else:
                try:
                    db = LogDatabase(use_domain)
                except Exception as e:
                    return (
                        False,
                        None,
                        f"Failed to initialize database for {use_domain}: {str(e)}",
                    )

            # Check if the database is initialized
            if not db.is_initialized():
                return (
                    False,
                    db,
                    f"Database for {use_domain.name} is not initialized or empty",
                )

            return True, db, ""
        else:
            # If no domain specified, check if any domain is available
            if not self.domains_dbs:
                return False, None, "No initialized domain databases available"

            # Choose a random domain
            random_domain = random.choice(list(self.domains_dbs.keys()))
            return True, self.domains_dbs[random_domain], ""

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse timestamp string into datetime object"""
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            # Fallback for other timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            # If all formats fail, return current time as fallback
            return datetime.now()

    def _get_all_log_numbers(self, db: LogDatabase) -> list[int]:
        """
        Get all unique log numbers from the database.
        This is a helper method to work around the missing get_all_log_numbers method in LogDatabase.

        Args:
            db: LogDatabase instance

        Returns:
            List of unique log numbers
        """
        try:
            # Use get_all_chunks method to get all chunks, then extract unique log numbers
            all_chunks = db.get_all_chunks()
            # Extract unique log numbers from chunks
            log_numbers = list(set(chunk["log_number"] for chunk in all_chunks))
            # Sort log numbers
            log_numbers.sort()
            return log_numbers
        except Exception as e:
            print(f"[ERROR] Error getting log numbers: {str(e)}")
            return []

    def get_sequential_logs(
        self,
        start_log_number: int | None = None,
        count: int = 3,
        domain: LogDomain | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get a sequence of logs that follow each other chronologically.

        Args:
            start_log_number: Optional starting log number (random if None)
            count: Number of sequential logs to retrieve (between MIN and MAX)
            domain: Optional domain to override the instance domain

        Returns:
            List of log dictionaries in chronological order
        """
        # Validate database access
        success, db, error_message = self.validate_db_access(domain)
        if not success:
            print(f"[ERROR] Database access validation failed: {error_message}")
            return []

        # Adjust count to be within bounds
        count = max(MIN_SEQUENTIAL_LOGS, min(count, MAX_SEQUENTIAL_LOGS))

        try:
            # Get all log numbers from the domain - use our helper method instead
            all_log_numbers = self._get_all_log_numbers(db)
            if not all_log_numbers:
                print(
                    f"[WARNING] No logs found for domain {db.domain.name if hasattr(db, 'domain') else 'unknown'}"
                )
                return []

            # If start log not specified, choose a random one that has enough room for sequence
            if start_log_number is None:
                potential_starts = (
                    all_log_numbers[: -count + 1]
                    if len(all_log_numbers) > count
                    else all_log_numbers
                )
                if not potential_starts:
                    print("[WARNING] Not enough logs to create a sequence")
                    return []
                start_log_number = random.choice(potential_starts)

            # Get initial log
            start_log = self._get_complete_log(db, start_log_number)
            if not start_log:
                print(f"[WARNING] Could not retrieve log {start_log_number}")
                return []

            result_logs = [start_log]
            current_timestamp = self._parse_timestamp(start_log["timestamp"])

            # Find subsequent logs based on timestamp within window
            all_logs = []
            for log_num in all_log_numbers:
                if log_num != start_log_number:  # Skip the start log
                    log = self._get_complete_log(db, log_num)
                    if log:
                        all_logs.append(log)

            # Sort logs by timestamp
            all_logs.sort(key=lambda x: self._parse_timestamp(x["timestamp"]))

            # Find logs that are within the time window from the current timestamp
            for log in all_logs:
                log_timestamp = self._parse_timestamp(log["timestamp"])
                time_diff = (log_timestamp - current_timestamp).total_seconds()

                # Check if this log is within our time window and is later than current
                if 0 < time_diff < TIME_WINDOW_SECONDS:
                    result_logs.append(log)
                    current_timestamp = log_timestamp

                    # Stop if we have enough logs
                    if len(result_logs) >= count:
                        break

            # If we couldn't find enough sequential logs, supplement with random ones
            if len(result_logs) < count:
                remaining_logs = [
                    log
                    for log in all_logs
                    if log["log_number"]
                    not in [res_log["log_number"] for res_log in result_logs]
                ]
                remaining_count = count - len(result_logs)

                if remaining_logs and remaining_count > 0:
                    random_additional = random.sample(
                        remaining_logs, min(remaining_count, len(remaining_logs))
                    )
                    result_logs.extend(random_additional)

            # Sort final result by timestamp to ensure proper sequence
            result_logs.sort(key=lambda x: self._parse_timestamp(x["timestamp"]))

            return result_logs[:count]  # Ensure we return exactly the requested count

        except Exception as e:
            print(f"[ERROR] Error retrieving sequential logs: {str(e)}")
            return []

    def _get_complete_log(
        self, db: LogDatabase, log_number: int
    ) -> dict[str, Any] | None:
        """
        Get a complete log by combining all its chunks.

        Args:
            db: LogDatabase instance
            log_number: Log number to retrieve

        Returns:
            Complete log dictionary or None if not found
        """
        chunks = db.get_chunks_by_log_number(log_number)
        if not chunks:
            return None

        # Sort chunks by chunk number
        chunks.sort(key=lambda x: x["chunk_number"])

        # Extract timestamp from first chunk
        timestamp = chunks[0]["timestamp"]

        # Concatenate all chunk texts
        full_text = ""
        for chunk in chunks:
            chunk_text = chunk["chunk_text"]
            # Remove timestamp from beginning if it's duplicated
            if chunk_text.startswith(timestamp):
                chunk_text = chunk_text[len(timestamp) :].strip()
            full_text += chunk_text + " "

        return {
            "log_number": log_number,
            "timestamp": timestamp,
            "log_text": full_text.strip(),
            "domain": db.domain.name if hasattr(db, "domain") else None,
        }

    def sample_logs_by_domain(
        self,
        domains: list[LogDomain] | None = None,
        count_per_domain: int = 3,
        sequential: bool = True,
    ) -> dict[LogDomain, list[dict[str, Any]]]:
        """
        Sample logs from domains (restricted to FS if not specified)

        Args:
            domains: List of domains to sample from (defaults to FS only)
            count_per_domain: Number of logs to sample per domain
            sequential: Whether to sample logs sequentially

        Returns:
            Dictionary mapping domains to lists of sampled logs
        """
        result = {}
        # Use only FS domain if none specified
        use_domains = domains if domains else [LogDomain.FS]

        # Filter to only include FS domain
        use_domains = [domain for domain in use_domains if domain == LogDomain.FS]

        if not use_domains:
            use_domains = [LogDomain.FS]

        for domain in use_domains:
            if sequential:
                logs = self.get_sequential_logs(count=count_per_domain, domain=domain)
            else:
                # For non-sequential, just get random logs
                success, db, error_message = self.validate_db_access(domain)
                if not success:
                    print(
                        f"[ERROR] Database access validation failed for domain {domain.name}: {error_message}"
                    )
                    logs = []
                else:
                    all_log_numbers = self._get_all_log_numbers(db)
                    if all_log_numbers:
                        sample_numbers = random.sample(
                            all_log_numbers, min(count_per_domain, len(all_log_numbers))
                        )
                        logs = [
                            self._get_complete_log(db, num) for num in sample_numbers
                        ]
                        logs = [log for log in logs if log]  # Filter out None values
                    else:
                        logs = []

            if logs:
                result[domain] = logs

        return result


# Example usage function
def sample_connected_logs(
    domain: LogDomain | None = None, count: int = 3
) -> list[dict[str, Any]]:
    """
    Sample a sequence of connected logs from the specified domain.

    Args:
        domain: Optional specific domain to sample from
        count: Number of sequential logs to retrieve

    Returns:
        List of chronologically connected logs
    """
    sampler = SequentialLogSampler(domain)
    return sampler.get_sequential_logs(count=count)
