import json
import os
import random
from datetime import datetime
from typing import Any

import numpy as np


class TextChunker:
    """Splits text into appropriate chunks for processing"""

    def __init__(self, chunk_size=200, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks intelligently"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find the end of the chunk
            end = min(start + self.chunk_size, len(text))

            # Try to find natural break points if not at the end of text
            if end < len(text):
                # Try paragraph, sentence, then word breaks
                paragraph_break = text.rfind("\n\n", start, end)
                sentence_break = text.rfind(". ", start, end)
                word_break = text.rfind(" ", start, end)

                # Use the most appropriate break point
                if (
                    paragraph_break != -1
                    and paragraph_break > start + self.chunk_size // 2
                ):
                    end = paragraph_break + 2
                elif (
                    sentence_break != -1
                    and sentence_break > start + self.chunk_size // 2
                ):
                    end = sentence_break + 2
                elif word_break != -1:
                    end = word_break + 1

            # Add the chunk
            chunks.append(text[start:end])

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

        return chunks


class FakeEmbedding:
    """Generates deterministic fake embeddings"""

    def __init__(self, dim=384):
        self.dim = dim
        self.seed = 42  # Fixed seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_deterministic_seed(self, text: str) -> int:
        """Create a deterministic hash from text content"""
        return sum(ord(c) * (i + 1) for i, c in enumerate(text[:100])) % 10000

    def embed(self, text: str) -> list[Any]:
        """Generate fake but deterministic embedding vector"""
        # Set seed based on text content for reproducibility
        text_seed = self.get_deterministic_seed(text)
        np.random.seed(text_seed)

        # Generate a random but deterministic embedding
        embedding = np.random.normal(0, 1, self.dim)

        # Normalize the embedding to unit length (cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Reset seed to avoid affecting other operations
        np.random.seed(self.seed)

        return embedding.tolist()

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = (sum(a * a for a in emb1)) ** 0.5
        norm2 = (sum(b * b for b in emb2)) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)


class RAGManager:
    """Manages retrieval-augmented generation operations"""

    def __init__(self, logs_dir="domain_logs"):
        """Initialize the RAG manager"""
        self.logs_dir = logs_dir
        self.chunker = TextChunker()
        self.embedding_model = FakeEmbedding()
        self.domain_indices = {}  # Will store domain -> [documents]
        self.domain_embeddings = {}  # Will store domain -> [embeddings]

        # Create logs directory if needed
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize domain data
        self._initialize_domains()

    def _initialize_domains(self):
        """Load and index domain data"""
        # Check available domain files
        for filename in os.listdir(self.logs_dir):
            if filename.endswith("_logs.json"):
                domain = filename.replace("_logs.json", "")
                self._index_domain(domain)

        print(f"Initialized RAG system with {len(self.domain_indices)} domains")

    def _index_domain(self, domain: str):
        """Index a specific domain's logs"""
        file_path = os.path.join(self.logs_dir, f"{domain}_logs.json")

        if not os.path.exists(file_path):
            print(f"Warning: No log file found for domain '{domain}'")
            return

        try:
            with open(file_path) as f:
                logs = json.load(f)

            self.domain_indices[domain] = []
            self.domain_embeddings[domain] = []

            # Process each log entry
            for log_entry in logs:
                # Extract text and timestamp
                timestamp = log_entry.get("timestamp", "")
                log_text = log_entry.get("log", "")

                if not log_text:
                    continue

                # Format timestamp if available
                formatted_time = ""
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        formatted_time = timestamp

                # Prepare the document text (with timestamp if available)
                if formatted_time:
                    doc_text = f"{formatted_time}: {log_text}"
                else:
                    doc_text = log_text

                # Chunk the document if needed
                chunks = self.chunker.split_text(doc_text)

                # Index each chunk with its embedding
                for chunk in chunks:
                    embedding = self.embedding_model.embed(chunk)

                    self.domain_indices[domain].append(
                        {
                            "text": chunk,
                            "timestamp": timestamp,
                            "formatted_time": formatted_time,
                            "original_log": log_text,
                        }
                    )

                    self.domain_embeddings[domain].append(embedding)

            print(
                f"Indexed {len(self.domain_indices[domain])} chunks for domain '{domain}'"
            )

        except Exception as e:
            print(f"Error indexing domain '{domain}': {str(e)}")

    def get_available_domains(self) -> list[str]:
        """Get list of available indexed domains"""
        return list(self.domain_indices.keys())

    def retrieve(self, domain: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query from a specific domain"""
        if domain not in self.domain_indices or not query:
            return []

        if not self.domain_indices[domain]:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)

        # Calculate similarities with all documents in the domain
        similarities = []
        for i, doc_embedding in enumerate(self.domain_embeddings[domain]):
            similarity = self.embedding_model.similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top_k results
        results = []
        for i, similarity in similarities[:top_k]:
            document = self.domain_indices[domain][i]
            results.append(
                {
                    "text": document["text"],
                    "timestamp": document["timestamp"],
                    "formatted_time": document["formatted_time"],
                    "original_log": document["original_log"],
                    "score": similarity,
                }
            )

        return results

    def retrieve_formatted(self, domain: str, query: str, top_k: int = 3) -> str:
        """Retrieve and format results as a readable string"""
        results = self.retrieve(domain, query, top_k)

        if not results:
            return f"No relevant information found for '{query}' in domain '{domain}'."

        # Format the results
        output = f"Found {len(results)} relevant items for '{query}' in domain '{domain}':\n\n"

        for i, result in enumerate(results):
            score = result["score"]
            text = result["text"]
            output += f"{i + 1}. [{score:.2f}] {text}\n\n"

        return output

    def multi_domain_retrieve(
        self, query: str, domains: list[str], top_k_per_domain: int = 2
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve results across multiple domains"""
        results = {}

        for domain in domains:
            domain_results = self.retrieve(domain, query, top_k_per_domain)
            if domain_results:
                results[domain] = domain_results

        return results

    def multi_domain_retrieve_formatted(
        self, query: str, domains: list[str], top_k_per_domain: int = 2
    ) -> str:
        """Retrieve and format results from multiple domains"""
        all_results = self.multi_domain_retrieve(query, domains, top_k_per_domain)

        if not all_results:
            return f"No relevant information found for '{query}' across the specified domains."

        output = f"Results for '{query}' across {len(all_results)} domains:\n\n"

        for domain, results in all_results.items():
            output += f"--- {domain.upper()} DOMAIN ---\n"
            for i, result in enumerate(results):
                score = result["score"]
                text = result["text"]
                output += f"{i + 1}. [{score:.2f}] {text}\n"
            output += "\n"

        return output


# Create a global instance for easy access
rag_manager = RAGManager()


def get_rag_manager() -> RAGManager:
    """Get the global RAG manager instance"""
    return rag_manager
