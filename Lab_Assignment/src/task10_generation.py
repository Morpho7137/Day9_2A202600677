"""
Task 10 - Generation with citations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.example", override=False)

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma túy 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Example:
        input:  [0, 1, 2, 3, 4]
        output: [0, 2, 4, 3, 1]
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


def _source_label(chunk: dict, fallback_index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", f"Source {fallback_index}")
    source = Path(source).stem if "." in source else source

    article_year = metadata.get("published_at") or metadata.get("year")
    if isinstance(article_year, str):
        year_match = re.search(r"(20\d{2}|19\d{2})", article_year)
        if year_match:
            return f"{source}, {year_match.group(1)}"

    if metadata.get("type") == "legal":
        return source
    return source


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a prompt-ready context string with source labels.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        citation_label = _source_label(chunk, i)
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Citation: {citation_label} | Score: {score:.3f}]\n"
            f"{chunk.get('content', '').strip()}\n"
        )
    return "\n---\n".join(context_parts)


def _has_real_openai_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return False
    placeholder_markers = ("xxx", "your_", "sk-xxx")
    lower_key = api_key.lower()
    return not any(marker in lower_key for marker in placeholder_markers)


def _extract_sentences(text: str, max_sentences: int = 2) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    picked = [sentence.strip() for sentence in sentences if sentence.strip()]
    return picked[:max_sentences] if picked else [normalized[:240].strip()]


def _fallback_generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Deterministic citation-based fallback when no LLM API is configured.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    lines = []
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in ("hình phạt", "điều", "quy định", "luật")):
        intro = "Theo các nguồn truy xuất được, thông tin liên quan có thể tóm tắt như sau:"
        lines.append(intro)

    for i, chunk in enumerate(chunks[:3], 1):
        citation = _source_label(chunk, i)
        for sentence in _extract_sentences(chunk.get("content", ""), max_sentences=1):
            lines.append(f"{sentence} [{citation}]")

    if not lines:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return "\n\n".join(lines)


def _format_history(history: list[dict] | None, max_turns: int = 6) -> str:
    """Format recent conversation turns for follow-up questions."""
    if not history:
        return ""

    lines = []
    for item in history[-max_turns:]:
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _generate_with_openai(query: str, context: str, history: list[dict] | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history_text = _format_history(history)
    history_block = f"Conversation history:\n{history_text}\n\n---\n\n" if history_text else ""
    user_message = f"{history_block}Context:\n{context}\n\n---\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def generate_with_citation(query: str, top_k: int = TOP_K, history: list[dict] | None = None) -> dict:
    """
    End-to-end RAG generation with citations.
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if _has_real_openai_key():
        try:
            answer = _generate_with_openai(query, context, history=history)
        except Exception:
            answer = _fallback_generate_answer(query, reordered)
    else:
        answer = _fallback_generate_answer(query, reordered)

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "none") if reordered else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma túy?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy 2021?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {query}")
        print("=" * 70)
        result = generate_with_citation(query)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
