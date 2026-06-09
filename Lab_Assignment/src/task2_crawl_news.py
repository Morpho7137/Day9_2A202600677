"""
Task 2 - Crawl news articles about Vietnamese artists related to drugs.

Requirements:
    1. Crawl at least 5 articles from Vietnamese news websites.
    2. Save one JSON file per article in data/landing/news/.
    3. Include metadata: url, title, date_crawled, content_markdown.

This implementation prefers Crawl4AI when available and falls back to
requests + stdlib HTML parsing so the task can still run without extra deps.
"""

import asyncio
import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def setup_directory():
    """Create data/landing/news if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://vnexpress.net/dien-vien-hai-huu-tin-su-dung-ma-tuy-vi-to-mo-4599355.html",
    "https://thanhnien.vn/bat-qua-tang-danh-hai-hiep-ga-mua-ma-tuy-185257742.htm",
    "https://thanhnien.vn/dj-thai-hoang-vua-bi-bat-vi-tang-tru-ma-tuy-la-ai-185230425153220627.htm",
    "https://thanhnien.vn/so-vh-tt-tphcm-noi-ve-vu-ntk-nguyen-cong-tri-bi-bat-vi-ma-tuy-185250724203756295.htm",
]


class ArticleHTMLParser(HTMLParser):
    """Extract paragraph text from known article content containers."""

    def __init__(self):
        super().__init__()
        self.container_depth = 0
        self.capture_paragraph = False
        self.current_paragraph = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class", "")
        is_article_body = (
            tag == "article" and "fck_detail" in class_attr
        ) or (
            tag == "div" and attrs_dict.get("data-role") == "content"
        )

        if is_article_body:
            self.container_depth += 1
            return

        if self.container_depth and tag == "p":
            self.capture_paragraph = True
            self.current_paragraph = []

    def handle_endtag(self, tag):
        if self.container_depth and tag == "p" and self.capture_paragraph:
            text = " ".join("".join(self.current_paragraph).split())
            if text:
                self.paragraphs.append(text)
            self.capture_paragraph = False
            self.current_paragraph = []
            return

        if self.container_depth and tag in {"article", "div"}:
            self.container_depth -= 1

    def handle_data(self, data):
        if self.capture_paragraph:
            self.current_paragraph.append(data)


def _extract_meta(html: str, patterns: list[str], default: str = "") -> str:
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return unescape(match.group(1)).strip()
    return default


def _fetch_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def _parse_article_from_html(url: str, html: str) -> dict:
    parser = ArticleHTMLParser()
    parser.feed(html)

    title = _extract_meta(
        html,
        [
            r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"',
            r"<h1[^>]*>(.*?)</h1>",
            r"<title>\s*(.*?)\s*</title>",
        ],
        default="Unknown title",
    )
    description = _extract_meta(
        html,
        [
            r'<meta[^>]+name="description"[^>]+content="([^"]+)"',
            r'<p[^>]+class="description"[^>]*>(.*?)</p>',
        ],
    )
    published_at = _extract_meta(
        html,
        [
            r'<meta[^>]+itemprop="datePublished"[^>]+content="([^"]+)"',
            r'<meta[^>]+content="([^"]+)"[^>]+itemprop="datePublished"',
            r'<meta[^>]+property="article:published_time"[^>]+content="([^"]+)"',
            r'<meta[^>]+content="([^"]+)"[^>]+property="article:published_time"',
        ],
    )

    paragraphs = []
    if description:
        paragraphs.append(re.sub(r"<[^>]+>", " ", description))

    for paragraph in parser.paragraphs:
        normalized = re.sub(r"<[^>]+>", " ", paragraph)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if len(normalized) < 20:
            continue
        if normalized.startswith("Ảnh:"):
            continue
        paragraphs.append(normalized)

    content_markdown = "\n\n".join(paragraphs)
    if not content_markdown:
        raise ValueError(f"Could not extract article body from {url}")

    return {
        "url": url,
        "title": title,
        "published_at": published_at,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus markdown-like content.

    Returns:
        {
            "url": str,
            "title": str,
            "published_at": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and getattr(result, "markdown", None):
                metadata = getattr(result, "metadata", {}) or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                return {
                    "url": url,
                    "title": metadata.get("title", "Unknown title"),
                    "published_at": metadata.get("published_time", ""),
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except ImportError:
        pass
    except Exception:
        pass

    html = await asyncio.to_thread(_fetch_html, url)
    return _parse_article_from_html(url, html)


async def crawl_all():
    """Crawl all articles in ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filepath = DATA_DIR / f"article_{i:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Please fill ARTICLE_URLS before running.")
    else:
        asyncio.run(crawl_all())
