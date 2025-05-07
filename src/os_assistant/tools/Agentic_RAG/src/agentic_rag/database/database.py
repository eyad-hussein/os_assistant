import sqlite3
import json
import os
import numpy as np
import datetime
from typing import List, Dict, Any, Tuple, Optional
from ..config.config import get_db_path, TIMESTAMP_FORMAT
from tracer.config import LogDomain

class LogDatabase:
    def __init__(self, domain: LogDomain = None):
        self.domain = domain
        self.db_path = get_db_path(domain)
        self._ensure_db_directory()
        self._initialize_db()
    
    def _ensure_db_directory(self):
        """Ensure the directory for the database exists."""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _initialize_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for log chunks with embeddings - without indexing
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS log_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_number INTEGER NOT NULL,
            chunk_number INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            embedding TEXT,
            UNIQUE(log_number, chunk_number)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_domain(self) -> Optional[LogDomain]:
        """Return the domain associated with this database."""
        return self.domain

    def is_initialized(self) -> bool:
        """Check if the database has been initialized with any data."""
        if not os.path.exists(self.db_path):
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='log_chunks'")
        has_table = cursor.fetchone()[0] > 0
        
        if not has_table:
            conn.close()
            return False
            
        cursor.execute("SELECT count(*) FROM log_chunks")
        row_count = cursor.fetchone()[0]
        conn.close()
        
        return row_count > 0
    
    def get_highest_log_number(self) -> int:
        """Get the highest log number in the database."""
        if not self.is_initialized():
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(log_number) FROM log_chunks")
        result = cursor.fetchone()[0]
        conn.close()
        
        return result or 0
    
    def get_latest_timestamp(self) -> Optional[str]:
        """Get the latest timestamp in the database."""
        if not self.is_initialized():
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM log_chunks")
        result = cursor.fetchone()[0]
        conn.close()
        
        return result
    
    def get_next_timestamp(self) -> Optional[str]:
        """Get the timestamp for the next log (latest + 1 microsecond)."""
        latest = self.get_latest_timestamp()
        if not latest:
            return None
        
        try:
            # Parse the timestamp and add 1 microsecond
            dt = datetime.datetime.strptime(latest, TIMESTAMP_FORMAT)
            next_dt = dt + datetime.timedelta(microseconds=1)
            return next_dt.strftime(TIMESTAMP_FORMAT)
        except (ValueError, TypeError):
            # If parsing fails, return None
            return None
    
    def log_exists(self, log_number: int) -> bool:
        """Check if a log with the given number exists in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM log_chunks WHERE log_number = ?", (log_number,))
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def bulk_insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert multiple chunks in bulk for better performance."""
        if not chunks:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Use a single transaction for all inserts
            cursor.execute('BEGIN TRANSACTION')
            
            for chunk in chunks:
                embedding_json = json.dumps(chunk.get('embedding')) if chunk.get('embedding') is not None else None
                cursor.execute('''
                INSERT OR REPLACE INTO log_chunks 
                (log_number, chunk_number, chunk_text, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    chunk['log_number'], 
                    chunk['chunk_number'], 
                    chunk['chunk_text'], 
                    chunk['timestamp'], 
                    embedding_json
                ))
            
            conn.commit()
            print(f"Bulk inserted {len(chunks)} chunks successfully")
        except Exception as e:
            conn.rollback()
            print(f"Error during bulk insert: {e}")
        finally:
            conn.close()
    
    def insert_chunk(self, log_number: int, chunk_number: int, chunk_text: str, 
                    timestamp: str, embedding: Optional[List[float]] = None):
        """Insert a log chunk with its embedding into the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_json = json.dumps(embedding) if embedding else None
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO log_chunks 
            (log_number, chunk_number, chunk_text, timestamp, embedding)
            VALUES (?, ?, ?, ?, ?)
            ''', (log_number, chunk_number, chunk_text, timestamp, embedding_json))
            
            conn.commit()
        except Exception as e:
            print(f"Error inserting chunk into database: {e}")
        finally:
            conn.close()
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all chunks from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT log_number, chunk_number, chunk_text, timestamp, embedding
        FROM log_chunks
        ORDER BY log_number DESC, chunk_number ASC
        ''')
        
        results = []
        for row in cursor:
            chunk = dict(row)
            if chunk['embedding']:
                chunk['embedding'] = json.loads(chunk['embedding'])
            results.append(chunk)
        
        conn.close()
        return results
    
    def get_chunk_by_id(self, log_number: int, chunk_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by log_number and chunk_number."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT log_number, chunk_number, chunk_text, timestamp, embedding
        FROM log_chunks
        WHERE log_number = ? AND chunk_number = ?
        ''', (log_number, chunk_number))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            chunk = dict(row)
            if chunk['embedding']:
                chunk['embedding'] = json.loads(chunk['embedding'])
            return chunk
        return None
    
    def get_chunks_by_log_number(self, log_number: int) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific log number."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT log_number, chunk_number, chunk_text, timestamp, embedding
        FROM log_chunks
        WHERE log_number = ?
        ORDER BY chunk_number ASC
        ''', (log_number,))
        
        results = []
        for row in cursor:
            chunk = dict(row)
            if chunk['embedding']:
                chunk['embedding'] = json.loads(chunk['embedding'])
            results.append(chunk)
        
        conn.close()
        return results
    
    def get_all_embeddings(self) -> List[Tuple[int, int, List[float]]]:
        """Retrieve all embeddings with their log and chunk numbers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT log_number, chunk_number, embedding
        FROM log_chunks
        WHERE embedding IS NOT NULL
        ''')
        
        results = []
        for log_number, chunk_number, embedding_json in cursor:
            if embedding_json:
                embedding = json.loads(embedding_json)
                results.append((log_number, chunk_number, embedding))
        
        conn.close()
        return results
