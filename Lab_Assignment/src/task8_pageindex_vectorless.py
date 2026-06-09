"""
Task 8 - PageIndex vectorless retrieval.

This implementation uses the official Python SDK (`PageIndexClient`) and the
two document IDs the user already uploaded to PageIndex.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.example", override=False)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_STATE_PATH = Path(__file__).parent.parent / "data" / "index" / "pageindex_docs.json"


def _get_client():
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is missing. Set it in .env before using Task 8.")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_configured_doc_ids() -> list[str]:
    """
    Load document IDs from env, persisted state, or user-provided defaults.
    """
    doc_ids = []

    env_list = os.getenv("PAGEINDEX_DOC_IDS", "")
    if env_list:
        doc_ids.extend([item.strip() for item in env_list.split(",") if item.strip()])

    for env_key in ("PAGEINDEX_DOC_ID_1", "PAGEINDEX_DOC_ID_2", "PAGEINDEX_DOC_ID"):
        value = os.getenv(env_key, "").strip()
        if value:
            doc_ids.append(value)

    if PAGEINDEX_STATE_PATH.exists():
        try:
            saved = json.loads(PAGEINDEX_STATE_PATH.read_text(encoding="utf-8"))
            doc_ids.extend(saved.get("doc_ids", []))
        except Exception:
            pass

    deduped = []
    seen = set()
    for doc_id in doc_ids:
        if doc_id and doc_id not in seen:
            deduped.append(doc_id)
            seen.add(doc_id)
    return deduped


def _save_doc_ids(doc_ids: list[str]) -> None:
    PAGEINDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_STATE_PATH.write_text(
        json.dumps({"doc_ids": doc_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upload_documents() -> list[str]:
    """
    Reuse already-uploaded PageIndex documents or upload local files if needed.

    Returns:
        List[str]: PageIndex document IDs ready for retrieval.
    """
    client = _get_client()
    configured_ids = _load_configured_doc_ids()

    valid_ids = []
    for doc_id in configured_ids:
        try:
            client.get_document(doc_id)
            valid_ids.append(doc_id)
        except Exception:
            continue

    if valid_ids:
        _save_doc_ids(valid_ids)
        return valid_ids

    uploaded_ids = []
    source_files = []
    landing_legal = Path(__file__).parent.parent / "data" / "landing" / "legal"
    if landing_legal.exists():
        for filepath in sorted(landing_legal.iterdir()):
            if filepath.suffix.lower() in {".pdf", ".docx", ".doc"}:
                source_files.append(filepath)

    if not source_files:
        raise RuntimeError("No uploaded PageIndex docs found and no local legal files available to upload.")

    for filepath in source_files:
        result = client.submit_document(str(filepath))
        doc_id = result.get("doc_id")
        if doc_id:
            uploaded_ids.append(doc_id)
            print(f"  Uploaded: {filepath.name} -> {doc_id}")

    if not uploaded_ids:
        raise RuntimeError("PageIndex upload did not return any document IDs.")

    _save_doc_ids(uploaded_ids)
    return uploaded_ids


def _wait_for_retrieval(client, retrieval_id: str, timeout_seconds: int = 90) -> dict:
    deadline = time.time() + timeout_seconds
    last_response = None

    while time.time() < deadline:
        last_response = client.get_retrieval(retrieval_id)
        status = last_response.get("status")
        if status == "completed":
            return last_response
        if status in {"failed", "error"}:
            raise RuntimeError(f"PageIndex retrieval failed: {last_response}")
        time.sleep(2)

    raise TimeoutError(f"PageIndex retrieval timed out: {last_response}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.

    Args:
        query: Query string
        top_k: Maximum number of results

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    client = _get_client()
    doc_ids = upload_documents()

    raw_hits = []
    for doc_rank, doc_id in enumerate(doc_ids, start=1):
        try:
            if not client.is_retrieval_ready(doc_id):
                continue

            retrieval = client.submit_query(doc_id=doc_id, query=query, thinking=False)
            retrieval_id = retrieval["retrieval_id"]
            result = _wait_for_retrieval(client, retrieval_id)

            for node_rank, node in enumerate(result.get("retrieved_nodes", []), start=1):
                for content_rank, content_item in enumerate(node.get("relevant_contents", []), start=1):
                    text = (content_item.get("relevant_content") or "").strip()
                    if not text:
                        continue

                    rank_score = 1.0 / (doc_rank + node_rank + content_rank - 2)
                    raw_hits.append(
                        {
                            "content": text,
                            "score": float(rank_score),
                            "metadata": {
                                "doc_id": doc_id,
                                "title": node.get("title", ""),
                                "node_id": node.get("node_id", ""),
                                "page_index": content_item.get("page_index"),
                            },
                            "source": "pageindex",
                        }
                    )
        except Exception:
            continue

    # Deduplicate repeated excerpts while keeping the best score.
    deduped = {}
    for hit in raw_hits:
        key = hit["content"]
        if key not in deduped or hit["score"] > deduped[key]["score"]:
            deduped[key] = hit

    return sorted(deduped.values(), key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY in .env before running Task 8.")
    else:
        doc_ids = upload_documents()
        print(f"Using PageIndex docs: {doc_ids}")

        results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)
        for result in results:
            print(f"[{result['score']:.3f}] {result['content'][:100]}...")
