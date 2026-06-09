"""
Task 4 - Custom chunking and indexing.

Chunking choice:
    Legal documents in this repo already follow a strong structure
    (Chuong -> Muc -> Dieu), so a custom rule-based chunker is more appropriate
    than a generic splitter. It preserves legal boundaries first, then applies a
    hard size cap only when one section is still too long.

Embedding choice:
    Quockhanh05/Vietnam_legal_embeddings is a Vietnamese legal-domain sentence
    transformer. The model card states it is intended for legal semantic search
    and retrieval, which matches this assignment better than a general-purpose
    multilingual encoder.

Index choice:
    The repo does not ship a running vector database. To keep the pipeline
    runnable inside `.venv`, embeddings are persisted locally as JSON + NumPy
    arrays. Later tasks can reuse the same files for dense retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Custom rule-based chunking for legal text. Size 1200 keeps one or a few legal
# articles together, while 120 overlap preserves references that spill into the
# next chunk after the hard split fallback.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "custom_legal_sections"

# Hugging Face legal embedding model requested by the user.
EMBEDDING_MODEL = "Quockhanh05/Vietnam_legal_embeddings"
EMBEDDING_DIM = 768

# Local persisted vector index, reusable by later tasks without external infra.
VECTOR_STORE = "local_disk"


LEGAL_HEADING_RE = re.compile(
    r"(?im)^(Chương\s+[IVXLCĐA-Z0-9]+.*|Mục\s+[IVXLCĐA-Z0-9]+.*|Điều\s+\d+[A-Za-z0-9./-]*\.?.*)$"
)


def load_documents() -> list[dict]:
    """
    Read all markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if relative_path.parts[0] == "legal" else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(relative_path).replace("\\", "/"),
                },
            }
        )
    return documents


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sliding_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard cap fallback for sections that exceed CHUNK_SIZE."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        split_at = text.rfind("\n\n", start, end)
        if split_at <= start + int(chunk_size * 0.5):
            split_at = text.rfind("\n", start, end)
        if split_at <= start + int(chunk_size * 0.5):
            split_at = text.rfind(" ", start, end)
        if split_at <= start:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            pieces.append(chunk)
        if split_at >= len(text):
            break
        next_start = split_at - overlap
        if next_start <= start:
            next_start = start + step
        start = min(len(text), next_start)
    return pieces


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge adjacent units up to CHUNK_SIZE, then hard-split oversize blocks."""
    chunks = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        candidate = unit if not current else f"{current}\n\n{unit}"
        if current and len(candidate) > chunk_size:
            chunks.extend(_sliding_split(current, chunk_size, overlap))
            current = unit
        else:
            current = candidate

    if current:
        chunks.extend(_sliding_split(current, chunk_size, overlap))

    return chunks


def _split_legal_document(content: str) -> list[str]:
    """
    Split legal markdown by explicit headings first.

    Preferred boundaries are Dieu/Muc/Chuong headings, which map more naturally
    to legal retrieval than generic paragraph windows.
    """
    content = _normalize_whitespace(content)
    matches = list(LEGAL_HEADING_RE.finditer(content))
    if not matches:
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return _merge_units(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)

    units = []
    first_start = matches[0].start()
    if first_start > 0:
        preamble = content[:first_start].strip()
        if preamble:
            units.append(preamble)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end].strip()
        if section:
            units.append(section)

    return _merge_units(units, CHUNK_SIZE, CHUNK_OVERLAP)


def _split_news_document(content: str) -> list[str]:
    """Paragraph-aware merging for short news articles."""
    content = _normalize_whitespace(content)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return _merge_units(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents according to their type.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks = []
    for doc in documents:
        doc_type = doc.get("metadata", {}).get("type", "unknown")
        content = doc.get("content", "")
        if not content:
            continue

        if doc_type == "legal":
            split_chunks = _split_legal_document(content)
        else:
            split_chunks = _split_news_document(content)

        for i, chunk_text in enumerate(split_chunks):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunking_method": CHUNKING_METHOD,
                    },
                }
            )
    return chunks


def _load_embedding_model():
    """Lazy-load the sentence-transformer to keep import-time test-safe."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    if dimension:
        global EMBEDDING_DIM
        EMBEDDING_DIM = int(dimension)
    return model


def embed_chunks(chunks: list[dict], batch_size: int = 16) -> list[dict]:
    """
    Embed all chunks using the chosen legal-domain model.

    Returns:
        Each chunk dict gets an 'embedding': list[float]
    """
    if not chunks:
        return []

    model = _load_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched = dict(chunk)
        enriched["embedding"] = embedding.tolist()
        embedded_chunks.append(enriched)
    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Persist chunks and embeddings locally for later semantic retrieval.
    """
    import numpy as np

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunk_records = []
    embedding_rows = []

    for chunk in chunks:
        embedding = chunk.get("embedding")
        if embedding is None:
            raise ValueError("All chunks must include embeddings before indexing.")
        embedding_rows.append(embedding)
        chunk_records.append(
            {
                "content": chunk["content"],
                "metadata": chunk["metadata"],
            }
        )

    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(INDEX_DIR / "embeddings.npy", np.asarray(embedding_rows, dtype="float32"))

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_method": CHUNKING_METHOD,
        "vector_store": VECTOR_STORE,
        "num_chunks": len(chunk_records),
    }
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_pipeline():
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to local vector store")


if __name__ == "__main__":
    run_pipeline()
