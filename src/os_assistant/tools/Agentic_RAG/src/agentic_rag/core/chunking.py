from typing import List, Dict, Any, Tuple
import json
import math
from ..config.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def load_logs(file_path: str) -> List[Dict[str, Any]]:
    """Load logs from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            logs = json.load(f)
        return logs
    except Exception as e:
        print(f"Error loading logs: {e}")
        return []

def chunk_logs(logs: List[Dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, 
               overlap: float = DEFAULT_CHUNK_OVERLAP, start_log_number: int = 1) -> List[Dict[str, Any]]:
    """
    Chunk logs with specified size and overlap.
    Each chunk will have timestamp at the beginning.
    
    Args:
        logs: List of log entries to chunk
        chunk_size: Size of each chunk in characters
        overlap: Fraction of overlap between chunks (0-1)
        start_log_number: Starting log number for the chunks (for incremental updates)
    """
    chunks = []
    log_number = start_log_number
    
    for log_entry in logs:
        timestamp = log_entry.get("timestamp", "")
        log_text = log_entry.get("log", "")
        
        # Format text with timestamp
        full_text = f"{timestamp} {log_text}"
        print(full_text)
        # If text is shorter than chunk size, use it as one chunk
        if len(full_text) <= chunk_size:
            chunks.append({
                "log_number": log_number,
                "chunk_number": 1,
                "chunk_text": full_text,
                "timestamp": timestamp
            })
        else:
            # Calculate actual overlap in characters
            overlap_size = int(chunk_size * overlap)
            stride = chunk_size - overlap_size
            
            # Calculate number of chunks needed
            num_chunks = math.ceil((len(full_text) - overlap_size) / stride)
            
            for i in range(num_chunks):
                start_idx = i * stride
                end_idx = min(start_idx + chunk_size, len(full_text))
                
                chunk_text = full_text[start_idx:end_idx]
                
                chunks.append({
                    "log_number": log_number,
                    "chunk_number": i + 1,
                    "chunk_text": chunk_text,
                    "timestamp": timestamp
                })
        
        log_number += 1
    
    return chunks
