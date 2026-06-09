"""
Task 3 - Convert files in data/landing/ to Markdown.

The assignment suggests Microsoft's MarkItDown. This implementation uses it
when the correct package is available in the environment and falls back to a
small DOCX extractor so the task remains runnable in a clean venv.
"""

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from markitdown import MarkItDown  # type: ignore
except ImportError:
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _get_markitdown():
    """Return a usable MarkItDown instance if the real package is installed."""
    if MarkItDown is None:
        return None
    if not hasattr(MarkItDown, "__call__") and not isinstance(MarkItDown, type):
        return None
    try:
        return MarkItDown()
    except Exception:
        return None


def _extract_docx_text(filepath: Path) -> str:
    """Read plain text from a DOCX file using only the stdlib."""
    paragraphs = []
    with zipfile.ZipFile(filepath) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        parts = []
        for text_node in paragraph.findall(".//w:t", WORD_NAMESPACE):
            if text_node.text:
                parts.append(text_node.text)
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)

    return "\n\n".join(paragraphs)


def _convert_legal_file(filepath: Path, md_converter) -> str:
    """Convert one legal source file to markdown-like text."""
    if md_converter is not None:
        result = md_converter.convert(str(filepath))
        text_content = getattr(result, "text_content", "") or getattr(result, "markdown", "")
        if text_content:
            return text_content

    if filepath.suffix.lower() in {".docx", ".doc"}:
        return _extract_docx_text(filepath)

    raise RuntimeError(f"Cannot convert file without a valid converter: {filepath.name}")


def convert_legal_docs():
    """Convert PDF/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_converter = _get_markitdown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        print(f"Converting: {filepath.name}")
        content = _convert_legal_file(filepath, md_converter)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert crawled article JSON files in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Published:** {data.get('published_at', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  Saved: {output_path}")


def convert_all():
    """Convert all supported files under data/landing/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\nDone. Output at: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
