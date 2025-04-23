from typing import List, Dict, Any, Tuple, Set
import numpy as np
from src.embedding import EmbeddingGenerator
from src.database import LogDatabase
from src.config import DEFAULT_TOP_K

class LogRetriever:
    def __init__(self, db: LogDatabase, embedding_generator: EmbeddingGenerator):
        self.db = db
        self.embedding_generator = embedding_generator
    
    def find_similar_chunks(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Find the most similar chunks to the query based on cosine similarity."""
        # Get embedding for query
        query_embedding = self.embedding_generator.get_embedding(query)
        if not query_embedding:
            return []
        
        # Get all embeddings from database
        all_embeddings = self.db.get_all_embeddings()
        if not all_embeddings:
            return []
            
        # Calculate similarities all at once using numpy for faster processing
        log_numbers, chunk_numbers = [], []
        embeddings = []
        
        for log_number, chunk_number, embedding in all_embeddings:
            log_numbers.append(log_number)
            chunk_numbers.append(chunk_number)
            embeddings.append(embedding)
        
        # Calculate similarities in batch
        similarities = [
            self.embedding_generator.cosine_similarity(query_embedding, embedding)
            for embedding in embeddings
        ]
        
        # Create triplets of (log_number, chunk_number, similarity)
        similarity_data = list(zip(log_numbers, chunk_numbers, similarities))
        
        # Sort by similarity (highest first)
        similarity_data.sort(key=lambda x: x[2], reverse=True)
        
        # Get top_k chunks
        results = []
        for log_number, chunk_number, similarity in similarity_data[:top_k]:
            chunk = self.db.get_chunk_by_id(log_number, chunk_number)
            if chunk:
                chunk['similarity'] = similarity
                results.append(chunk)
        
        return results
    
    def aggregate_logs(self, similar_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate chunks from the same log_number into complete logs.
        Remove duplicate timestamps and concatenate log text.
        """
        # Group chunks by log_number
        log_groups = {}
        for chunk in similar_chunks:
            log_number = chunk['log_number']
            if log_number not in log_groups:
                log_groups[log_number] = []
            log_groups[log_number].append(chunk)
        
        aggregated_logs = []
        for log_number, chunks in log_groups.items():
            # Get all chunks for this log from database to ensure completeness
            all_log_chunks = self.db.get_chunks_by_log_number(log_number)
            
            # Sort by chunk number
            all_log_chunks.sort(key=lambda x: x['chunk_number'])
            
            # Extract timestamp from first chunk
            timestamp = all_log_chunks[0]['timestamp']
            
            # Concatenate the text, removing timestamps
            full_text = ""
            for chunk in all_log_chunks:
                # Remove timestamp from beginning of chunk text
                chunk_text = chunk['chunk_text']
                if chunk_text.startswith(timestamp):
                    chunk_text = chunk_text[len(timestamp):].strip()
                full_text += chunk_text + " "
            
            # Find the highest similarity score from the matched chunks
            similarity = max([c.get('similarity', 0) for c in chunks])
            
            # Create aggregated log entry
            aggregated_log = {
                'log_number': log_number,
                'timestamp': timestamp,
                'log_text': full_text.strip(),
                'similarity': similarity,
                'matched_chunks': [c['chunk_number'] for c in chunks]
            }
            
            aggregated_logs.append(aggregated_log)
        
        # Sort by similarity
        aggregated_logs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return aggregated_logs
