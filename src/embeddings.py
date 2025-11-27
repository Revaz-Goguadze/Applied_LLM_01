"""generates embeddings using google gemini api"""

import google.generativeai as genai
from typing import List
import numpy as np
from .config import GEMINI_API_KEY, EMBEDDING_MODEL
import time


class EmbeddingGenerator:
    """handles embedding generation with gemini's text-embedding-004 model"""

    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = EMBEDDING_MODEL

    def embed_text(self, text: str, retry_count: int = 3) -> np.ndarray:
        """generates a single embedding vector for given text"""
        for attempt in range(retry_count):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                return np.array(result['embedding'])
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"embedding failed (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """processes multiple texts with progress bar and rate limiting"""
        embeddings = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="embedding")
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.embed_text(text) for text in batch]
            embeddings.extend(batch_embeddings)

            # slight pause to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """generates embedding optimized for query matching"""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return np.array(result['embedding'])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """computes cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def cosine_similarity_batch(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """efficiently computes similarity between one query and many vectors"""
    query_norm = query_vec / np.linalg.norm(query_vec)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms

    return np.dot(vectors_norm, query_norm)
