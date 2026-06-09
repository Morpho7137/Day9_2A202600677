"""
Task 5 - Semantic Search Module.

Dense retrieval is served from the local index generated in Task 4:
    - data/index/chunks.json
    - data/index/embeddings.npy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .task4_chunking_indexing import EMBEDDING_MODEL

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

_MODEL = None
_CHUNKS = None
_EMBEDDINGS = None


def _quality_score(text: str) -> float:
    """
    Penalize low-information chunks such as forms filled with dots/placeholders.
    """
    length = max(len(text), 1)
    alnum_ratio = sum(ch.isalnum() for ch in text) / length
    token_count = len(text.split())
    token_factor = min(1.0, token_count / 40.0)
    return max(0.15, alnum_ratio * 1.5) * max(0.25, token_factor)


def _load_index() -> tuple[list[dict], np.ndarray]:
    """Load persisted chunks and embedding matrix from Task 4."""
    global _CHUNKS, _EMBEDDINGS
    if _CHUNKS is None:
        chunks_path = INDEX_DIR / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                "Task 4 index not found. Run src/task4_chunking_indexing.py first."
            )
        _CHUNKS = json.loads(chunks_path.read_text(encoding="utf-8"))

    if _EMBEDDINGS is None:
        embeddings_path = INDEX_DIR / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(
                "Task 4 embeddings not found. Run src/task4_chunking_indexing.py first."
            )
        _EMBEDDINGS = np.load(embeddings_path)

    return _CHUNKS, _EMBEDDINGS


def _load_model():
    """Lazy-load the same embedding model used in Task 4."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search by vector similarity using the Task 4 local index.

    Args:
        query: Query string
        top_k: Maximum number of results

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    chunks, embeddings = _load_index()
    if not chunks or embeddings.size == 0:
        return []

    model = _load_model()
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        record = chunks[int(idx)]
        raw_score = float(scores[int(idx)])
        final_score = raw_score * _quality_score(record["content"])
        results.append(
            {
                "content": record["content"],
                "score": final_score,
                "metadata": record["metadata"],
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
