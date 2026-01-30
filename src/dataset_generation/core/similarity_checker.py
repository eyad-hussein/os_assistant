import json
import os

from dagent.tools.agentic_rag.core.embedding import EmbeddingGenerator

from ..config.config import DATASET_OUTPUT_DIR, SIMILARITY_THRESHOLD, VECTOR_CACHE_SIZE


class QuestionSimilarityChecker:
    """Checks for duplicate or too-similar questions using embedding similarity"""

    def __init__(
        self, threshold: float = SIMILARITY_THRESHOLD, cache_file: str | None = None
    ):
        """
        Initialize the similarity checker.

        Args:
            threshold: Similarity threshold (0-1) - higher means more strict duplicate detection
            cache_file: Optional file to store/load embeddings cache
        """
        self.threshold = threshold
        self.embedding_generator = EmbeddingGenerator()
        self.question_embeddings = {}  # question_text -> embedding
        self.cache_file = cache_file or os.path.join(
            DATASET_OUTPUT_DIR, "embedding_cache.json"
        )

        # Try to load existing cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load embedding cache from file if available"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                # Convert strings back to list embeddings
                for question, embedding_str in cache_data.items():
                    if isinstance(embedding_str, list):
                        self.question_embeddings[question] = embedding_str

                print(f"Loaded {len(self.question_embeddings)} embeddings from cache")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding cache: {str(e)}")

    def _save_cache(self) -> None:
        """Save embedding cache to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        try:
            # Limit cache size if needed
            if len(self.question_embeddings) > VECTOR_CACHE_SIZE:
                # Keep only the most recent entries
                questions = list(self.question_embeddings.keys())
                for old_key in questions[:-VECTOR_CACHE_SIZE]:
                    del self.question_embeddings[old_key]

            with open(self.cache_file, "w") as f:
                json.dump(self.question_embeddings, f)

            print(f"Saved {len(self.question_embeddings)} embeddings to cache")
        except Exception as e:
            print(f"[WARNING] Failed to save embedding cache: {str(e)}")

    def get_embedding(self, question: str) -> list[float]:
        """Get embedding for a question, using cache if available"""
        if question in self.question_embeddings:
            return self.question_embeddings[question]

        embedding = self.embedding_generator.get_embedding(question)
        if embedding:
            self.question_embeddings[question] = embedding
            # Save cache periodically (every 10 new embeddings)
            if len(self.question_embeddings) % 10 == 0:
                self._save_cache()

        return embedding

    def is_duplicate(
        self, question: str, existing_questions: list[str]
    ) -> tuple[bool, float, str]:
        """
        Check if a question is too similar to any existing questions.

        Args:
            question: The question to check
            existing_questions: List of existing questions to compare against

        Returns:
            Tuple of (is_duplicate, highest_similarity, most_similar_question)
        """
        if not existing_questions:
            return False, 0.0, ""

        # Get embedding for the new question
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            print(f"[WARNING] Could not generate embedding for question: {question}")
            return False, 0.0, ""  # Can't determine similarity without embedding

        # Find highest similarity among existing questions
        highest_similarity = 0.0
        most_similar_question = ""

        for existing in existing_questions:
            existing_embedding = self.get_embedding(existing)
            if not existing_embedding:
                continue

            similarity = self.embedding_generator.cosine_similarity(
                question_embedding, existing_embedding
            )

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_question = existing

        # Check if the highest similarity exceeds the threshold
        is_duplicate = highest_similarity >= self.threshold

        if is_duplicate:
            print(f"Duplicate detected (similarity: {highest_similarity:.3f}):")
            print(f"New: {question}")
            print(f"Existing: {most_similar_question}")

        return is_duplicate, highest_similarity, most_similar_question

    def filter_duplicates(self, questions: list[dict]) -> list[dict]:
        """
        Filter out duplicate questions from a list.

        Args:
            questions: List of question dictionaries (must have 'question' key)

        Returns:
            Filtered list with duplicates removed
        """
        if not questions:
            return []

        filtered_questions = []
        seen_questions = []

        for q in questions:
            question_text = q.get("question", "")
            if not question_text:
                continue

            is_dup, similarity, similar_q = self.is_duplicate(
                question_text, seen_questions
            )

            if not is_dup:
                filtered_questions.append(q)
                seen_questions.append(question_text)

        print(
            f"Filtered out {len(questions) - len(filtered_questions)} duplicates from {len(questions)} questions"
        )
        return filtered_questions


def check_duplicate_with_dataset(
    new_questions: list[dict], dataset_file: str
) -> list[dict]:
    """
    Check for duplicates against an existing dataset file.

    Args:
        new_questions: List of new question dictionaries
        dataset_file: Path to existing dataset JSON file

    Returns:
        Filtered list with duplicates removed
    """
    # Initialize checker
    checker = QuestionSimilarityChecker()

    # Load existing questions from dataset
    existing_questions = []
    try:
        if os.path.exists(dataset_file):
            with open(dataset_file) as f:
                dataset = json.load(f)
                samples = dataset.get("samples", [])
                existing_questions = [
                    s.get("question", "") for s in samples if "question" in s
                ]
                print(
                    f"Loaded {len(existing_questions)} existing questions from dataset"
                )
    except Exception as e:
        print(f"[WARNING] Error loading existing dataset: {str(e)}")

    # Check each new question against existing ones
    filtered_questions = []
    for q in new_questions:
        question_text = q.get("question", "")
        if not question_text:
            continue

        is_dup, similarity, similar_q = checker.is_duplicate(
            question_text, existing_questions
        )

        if not is_dup:
            filtered_questions.append(q)
            # Add to existing questions to prevent duplicates within new batch
            existing_questions.append(question_text)

    print(
        f"Filtered out {len(new_questions) - len(filtered_questions)} duplicates against existing dataset"
    )
    return filtered_questions
