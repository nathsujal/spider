from pathlib import Path
from typing import Any

from python.loader.base import BaseLoader
from python.models import Document, DocumentMetadata, Page, Section
from python.models.result import ParseResult
from .format import TextFormat, sniff_format
from .parsers import *

_PARSERS = {
    TextFormat.PLAIN: PlainParser(),
    TextFormat.MARKDOWN: MarkdownParser(),
    TextFormat.JSON: JSONParser(),
    TextFormat.JSONL: JSONLParser(),
    TextFormat.XML: XMLParser(),
}


class TextLoader(BaseLoader):
    """
    Loads text-based documents into Spider's normalised `Document` format.

    Supports: plain text (.txt), Markdown (.md / .mdx), HTML (.html / .htm),
    CSV (.csv), JSON (.json), JSONL / NDJSON (.jsonl / .ndjson), XML (.xml).

    Format is detected from the file extension first, then from content
    heuristics when the extension is absent or unrecognised.

    Usage
    -----
    >>> loader = TextLoader()
    >>> doc = loader.load("readme.md")
    >>> for s in doc.sections:
    ...     print(s.title, "—", len(s.text), "chars")
    """

    def __init__(self, *, encoding: str = "utf-8", errors: str = "replace"):
        self.encoding = encoding
        self.errors = errors

    def _load(self, source: str) -> Document:
        raw, path = self._read_source(source)
        fmt = sniff_format(path, raw)
        parser = _PARSERS[fmt]

        result = parser.parse(raw)
        clean = self.clean_text(result.text)

        metadata = self._build_metadata(source, path, fmt, result, raw)
        pages = self._paginate(clean)

        return Document(
            metadata=metadata,
            pages=pages,
            sections=result.sections,
            raw_text=clean,
        )

    def _read_source(self, source: str) -> tuple[str, Path]:
        if source.startswith(("http://", "https://")):
            return self._fetch_url(source)

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = path.read_text(encoding=self.encoding, errors=self.errors)
        return raw, path

    def _fetch_url(self, url: str) -> tuple[str, Path]:
        import httpx
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        # synthesise a fake path so _sniff_format can inspect the extension
        fake_path = Path(url.split("?")[0])
        return resp.text, fake_path

    def _build_metadata(
        self,
        source: str,
        path: Path,
        fmt: TextFormat,
        result: ParseResult,
        raw: str,
    ) -> DocumentMetadata:
        title = self._infer_title(result, path)
        return DocumentMetadata(
            source=source,
            source_type=fmt.name.lower(),
            title=title,
            author=None,
            created_at=None,
            page_count=None,
            extra={
                "format": fmt.name,
                "char_count": len(raw),
                **result.extra_meta,
            },
        )

    @staticmethod
    def _infer_title(result: ParseResult, path: Path) -> str | None:
        """Use first L1 section title, or the filename stem."""
        for s in result.sections:
            if s.level == 1 and s.title not in ("Full Document", "Preamble", "Part 1"):
                return s.title
        return path.stem or None

    @staticmethod
    def _paginate(text: str, chars_per_page: int = 3_000) -> list[Page]:
        """
        Splits the full text into virtual 'pages' for a consistent
        Document interface, even though text files have no real pages.
        """
        pages: list[Page] = []
        for i in range(0, max(len(text), 1), chars_per_page):
            chunk = text[i : i + chars_per_page]
            pages.append(Page(number=len(pages) + 1, text=chunk))
        return pages

if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else None
    if not source:
        print("Usage: python -m python.loader.text.loader <path_or_url>")
        sys.exit(1)

    loader = TextLoader()
    doc = loader.load(source)

    print(f"\n{'=' * 60}")
    print(f"  {doc}")
    print(f"  Title:  {doc.metadata.title}")
    print(f"  Format: {doc.metadata.extra.get('format')}")
    print(f"  Hash:   {doc.content_hash[:16]}...")
    print(f"{'=' * 60}")

    print(f"\n  Pages (virtual): {len(doc.pages)}")
    print(f"  Sections detected: {len(doc.sections)}")
    for s in doc.sections:
        indent = "  " * s.level
        print(f"  {indent}L{s.level}: {s.title!r}  ({len(s.text)} chars)")
    print()