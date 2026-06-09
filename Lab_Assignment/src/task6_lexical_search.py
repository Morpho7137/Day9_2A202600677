"""
Task 6 - Lexical Search Module (BM25).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CORPUS: list[dict] = []
_TOKENIZED_CORPUS = None
_BM25 = None


def _load_corpus() -> list[dict]:
    """Load the chunk corpus persisted by Task 4."""
    global CORPUS
    if not CORPUS:
        chunks_path = INDEX_DIR / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                "Task 4 chunks.json not found. Run src/task4_chunking_indexing.py first."
            )
        CORPUS = json.loads(chunks_path.read_text(encoding="utf-8"))
    return CORPUS


def _tokenize(text: str) -> list[str]:
    """
    Simple Unicode-aware tokenizer for Vietnamese text.

    It keeps letters/numbers, lowercases, and treats punctuation as boundaries.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _quality_score(text: str) -> float:
    """Penalize low-information chunks such as placeholder-heavy forms."""
    length = max(len(text), 1)
    alnum_ratio = sum(ch.isalnum() for ch in text) / length
    token_count = len(_tokenize(text))
    token_factor = min(1.0, token_count / 40.0)
    return max(0.15, alnum_ratio * 1.5) * max(0.25, token_factor)


def build_bm25_index(corpus: list[dict]):
    """
    Build a BM25 index from the chunk corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _TOKENIZED_CORPUS, _BM25
    _TOKENIZED_CORPUS = [_tokenize(doc["content"]) for doc in corpus]
    _BM25 = BM25Okapi(_TOKENIZED_CORPUS)
    return _BM25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search by keywords using BM25.

    Args:
        query: Query string
        top_k: Maximum number of results

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    global _BM25
    if _BM25 is None:
        _BM25 = build_bm25_index(corpus)

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = np.asarray(_BM25.get_scores(tokenized_query), dtype=float)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[int(idx)]) * _quality_score(corpus[int(idx)]["content"])
        if score <= 0:
            continue
        record = corpus[int(idx)]
        results.append(
            {
                "content": record["content"],
                "score": score,
                "metadata": record["metadata"],
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma túy", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
