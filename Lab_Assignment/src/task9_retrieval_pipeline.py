"""
Task 9 - Unified retrieval pipeline.

This stage does not require an LLM API. It only orchestrates retrieval:
semantic search + lexical search + merge + rerank + PageIndex fallback.
"""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _mark_source(items: list[dict], source: str) -> list[dict]:
    marked = []
    for item in items:
        enriched = dict(item)
        enriched["source"] = source
        marked.append(enriched)
    return marked


def _normalize_scores(items: list[dict]) -> list[dict]:
    """
    Normalize scores into [0, 1] so threshold comparisons are less arbitrary
    across BM25, cosine similarity, and cross-encoder outputs.
    """
    if not items:
        return []

    scores = [float(item.get("score", 0.0)) for item in items]
    min_score = min(scores)
    max_score = max(scores)

    normalized = []
    for item, score in zip(items, scores):
        enriched = dict(item)
        if max_score == min_score:
            enriched["score"] = 1.0 if max_score > 0 else 0.0
        else:
            enriched["score"] = (score - min_score) / (max_score - min_score)
        normalized.append(enriched)
    return normalized


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Unified retrieval pipeline with fallback logic.

    Steps:
        1. Run semantic_search + lexical_search
        2. Merge results with RRF
        3. Rerank merged candidates
        4. If best score < threshold -> fallback to PageIndex
        5. Return top_k results
    """
    if top_k <= 0:
        return []

    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = _mark_source(merged, "hybrid")
    merged = _normalize_scores(merged)

    if use_reranking and merged:
        reranked = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _mark_source(reranked, "hybrid")
        final_results = _normalize_scores(final_results)
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy năm 2024",
        "Luật phòng chống ma túy 2021 quy định gì về cai nghiện",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['score']:.3f}] [{result['source']}] {result['content'][:80]}...")
