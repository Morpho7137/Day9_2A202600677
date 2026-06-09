"""
Task 7 - Reranking Module.

Primary reranker:
    jinaai/jina-reranker-v2-base-multilingual

The Hugging Face model card describes it as a multilingual cross-encoder
reranker and shows loading with `CrossEncoder(..., trust_remote_code=True)`.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
_CROSS_ENCODER = None


def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _fallback_rerank_score(query: str, content: str, prior_score: float) -> float:
    """
    Backup scorer when the cross-encoder cannot be loaded.

    It uses token overlap plus the prior retrieval score so the module remains
    functional even if model download or runtime support is unavailable.
    """
    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))
    if not query_tokens or not content_tokens:
        return float(prior_score)

    overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    return float(0.7 * overlap + 0.3 * max(prior_score, 0.0))


def _load_cross_encoder():
    """Lazy-load the Jina multilingual reranker."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(
            RERANKER_MODEL,
            trust_remote_code=True,
        )
    return _CROSS_ENCODER


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using a cross-encoder model.

    Args:
        query: Query string
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Number of results after reranking

    Returns:
        List of top_k candidates, rescored and sorted descending.
    """
    if top_k <= 0 or not candidates:
        return []

    limited_candidates = list(candidates)
    try:
        model = _load_cross_encoder()
        pairs = [(query, candidate["content"]) for candidate in limited_candidates]
        scores = model.predict(pairs)
    except Exception:
        scores = [
            _fallback_rerank_score(query, candidate["content"], candidate.get("score", 0.0))
            for candidate in limited_candidates
        ]

    reranked = []
    for candidate, score in zip(limited_candidates, scores):
        item = dict(candidate)
        item["score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance - choose candidates that are both relevant and diverse.
    """
    if top_k <= 0 or not candidates:
        return []

    selected_indices = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected_indices) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding")
            if embedding is None:
                relevance = float(candidates[idx].get("score", 0.0))
            else:
                relevance = _cosine_sim(query_embedding, embedding)

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sel_embedding = candidates[sel_idx].get("embedding")
                if embedding is None or sel_embedding is None:
                    continue
                max_sim_to_selected = max(
                    max_sim_to_selected,
                    _cosine_sim(embedding, sel_embedding),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected_indices:
        item = dict(candidates[idx])
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion - merge multiple ranked result lists.
    """
    if top_k <= 0:
        return []

    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        item["score"] = float(score)
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    if method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma túy", dummy_candidates, top_k=2)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content']}")
