import numpy as np
import requests

from ..config.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


class EmbeddingGenerator:
    def __init__(
        self, model_name: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/embeddings"

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text using Ollama."""
        try:
            response = requests.post(
                self.api_endpoint, json={"model": self.model_name, "prompt": text}
            )

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                return embedding
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        return [self.get_embedding(text) for text in texts]

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors using optimized numpy approach."""
        if not vec1 or not vec2:
            return 0.0

        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)

        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm == 0:
            return 0.0

        return np.dot(vec1, vec2) / norm
