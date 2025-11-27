"""retrieval systems: dense embeddings, bm25, and hybrid fusion"""

import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import pickle
from .chunking import CodeChunk
from .embeddings import EmbeddingGenerator, cosine_similarity_batch


class DenseRetriever:
    """finds similar code using embedding cosine similarity"""

    def __init__(self, chunks: List[CodeChunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_generator = EmbeddingGenerator()

    def retrieve(self, query_code: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """returns top-k most similar chunks by embedding distance"""
        query_embedding = self.embedding_generator.embed_query(query_code)
        similarities = cosine_similarity_batch(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]

    def save(self, filepath: str):
        data = {
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'embeddings': self.embeddings
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'DenseRetriever':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        chunks = [CodeChunk.from_dict(d) for d in data['chunks']]
        return cls(chunks, data['embeddings'])


class BM25Retriever:
    """lexical search using bm25 algorithm"""

    def __init__(self, chunks: List[CodeChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize_code(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize_code(self, code: str) -> List[str]:
        """simple tokenizer - splits on whitespace and punctuation"""
        import re
        return re.findall(r'\w+', code.lower())

    def retrieve(self, query_code: str, top_k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """returns top-k matches by bm25 score"""
        query_tokens = self._tokenize_code(query_code)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]

    def save(self, filepath: str):
        data = {
            'chunks': [chunk.to_dict() for chunk in self.chunks],
            'tokenized_corpus': self.tokenized_corpus
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'BM25Retriever':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        chunks = [CodeChunk.from_dict(d) for d in data['chunks']]

        retriever = cls.__new__(cls)
        retriever.chunks = chunks
        retriever.tokenized_corpus = data['tokenized_corpus']
        retriever.bm25 = BM25Okapi(retriever.tokenized_corpus)

        return retriever


class HybridRetriever:
    """combines dense and bm25 using reciprocal rank fusion"""

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever

    def retrieve(
        self,
        query_code: str,
        top_k: int = 10,
        alpha: float = 0.5,
        fusion_method: str = 'rrf'
    ) -> List[Tuple[CodeChunk, float]]:
        """fuses results from both retrievers - alpha controls dense vs bm25 weight"""
        retrieve_k = max(top_k * 2, 20)
        dense_results = self.dense.retrieve(query_code, top_k=retrieve_k)
        bm25_results = self.bm25.retrieve(query_code, top_k=retrieve_k)

        if fusion_method == 'rrf':
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results, alpha)
        else:
            fused = self._weighted_fusion(dense_results, bm25_results, alpha)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[CodeChunk, float]],
        bm25_results: List[Tuple[CodeChunk, float]],
        alpha: float,
        k: int = 60
    ) -> List[Tuple[CodeChunk, float]]:
        """rrf formula: score = alpha/(k+rank_dense) + (1-alpha)/(k+rank_bm25)"""
        dense_ranks = {chunk: i for i, (chunk, _) in enumerate(dense_results)}
        bm25_ranks = {chunk: i for i, (chunk, _) in enumerate(bm25_results)}

        all_chunks = set(dense_ranks.keys()) | set(bm25_ranks.keys())

        rrf_scores = {}
        for chunk in all_chunks:
            dense_rank = dense_ranks.get(chunk, len(dense_results))
            bm25_rank = bm25_ranks.get(chunk, len(bm25_results))

            score = (
                alpha * (1.0 / (k + dense_rank)) +
                (1 - alpha) * (1.0 / (k + bm25_rank))
            )
            rrf_scores[chunk] = score

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _weighted_fusion(
        self,
        dense_results: List[Tuple[CodeChunk, float]],
        bm25_results: List[Tuple[CodeChunk, float]],
        alpha: float
    ) -> List[Tuple[CodeChunk, float]]:
        """simple weighted average after normalizing scores to 0-1"""
        dense_scores = {chunk: score for chunk, score in dense_results}
        bm25_scores = {chunk: score for chunk, score in bm25_results}

        max_dense = max(dense_scores.values()) if dense_scores else 1
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1

        all_chunks = set(dense_scores.keys()) | set(bm25_scores.keys())

        weighted = {}
        for chunk in all_chunks:
            dense_norm = dense_scores.get(chunk, 0) / max_dense
            bm25_norm = bm25_scores.get(chunk, 0) / max_bm25
            weighted[chunk] = alpha * dense_norm + (1 - alpha) * bm25_norm

        return sorted(weighted.items(), key=lambda x: x[1], reverse=True)
