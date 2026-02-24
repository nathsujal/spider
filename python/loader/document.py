import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .base import BaseLoader
from python.models import Document, DocumentMetadata, Page, Section


@dataclass
class _Span:
    """Internal: a text span with font metadata from PyMuPDF."""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    page_num: int


class DocumentLoader(BaseLoader):
    """
    Loads documents into Spider's normalised `Document` format.

    Uses font-size heuristics to detect section headings and build
    a hierarchical section structure for downstream chunking.

    Usage
    -----
    >>> loader = DocumentLoader()
    >>> doc = loader.load("paper.pdf")
    >>> for s in doc.sections:
    ...     print(s)
    """

    def __init__(self, *, min_paragraph_length: int = 20):
        self.min_paragraph_length = min_paragraph_length


    def _load(self, source: str) -> Document:
        pdf_doc = self._open(source)

        try:
            metadata = self._extract_metadata(pdf_doc, source)
            pages = self._extract_pages(pdf_doc)
            spans = self._extract_spans(pdf_doc)
        finally:
            pdf_doc.close()

        joined = "\n\n".join(p.text for p in pages)
        raw_text = self.clean_text(joined)

        sections = self._detect_sections(spans, raw_text)

        return Document(
            metadata=metadata,
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )


    def _open(self, source: str) -> fitz.Document:
        if source.startswith(("http://", "https://")):
            return self._open_from_url(source)

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {path}")

        return fitz.open(str(path))

    @staticmethod
    def _open_from_url(url: str) -> fitz.Document:
        import httpx
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        stream = io.BytesIO(resp.content)
        return fitz.open(stream=stream, filetype="pdf")


    @staticmethod
    def _extract_metadata(pdf_doc: fitz.Document, source: str) -> DocumentMetadata:
        info: dict[str, Any] = pdf_doc.metadata or {}
        return DocumentMetadata(
            source=source,
            source_type="pdf",
            title=info.get("title") or None,
            author=info.get("author") or None,
            created_at=info.get("creationDate") or None,
            page_count=len(pdf_doc),
            extra={
                k: v for k, v in info.items()
                if k not in ("title", "author", "creationDate") and v
            },
        )


    def _extract_pages(self, pdf_doc: fitz.Document) -> list[Page]:
        pages: list[Page] = []
        for page_num, page in enumerate(pdf_doc, start=1):
            text = page.get_text("text")
            pages.append(Page(number=page_num, text=text))
        return pages


    @staticmethod
    def _extract_spans(pdf_doc: fitz.Document) -> list[_Span]:
        """
        Extract text spans with font metadata, merging spans within
        the same line that share a similar font size.

        PyMuPDF often splits a single heading like "1. Introduction"
        into separate spans ("1." and "Introduction"). Merging them
        gives us complete heading text for section detection.
        """
        spans: list[_Span] = []

        for page_num, page in enumerate(pdf_doc, start=1):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    # collect all spans in this line, then merge
                    line_spans: list[_Span] = []
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue

                        font_name = span.get("font", "")
                        is_bold = (
                            "Bold" in font_name
                            or "bold" in font_name
                            or (span.get("flags", 0) & 2 ** 4) != 0
                        )

                        line_spans.append(_Span(
                            text=text,
                            font_size=round(span["size"], 1),
                            font_name=font_name,
                            is_bold=is_bold,
                            page_num=page_num,
                        ))

                    # merge adjacent spans with similar font size (±1pt)
                    merged = DocumentLoader._merge_line_spans(line_spans)
                    spans.extend(merged)

        return spans

    @staticmethod
    def _merge_line_spans(line_spans: list[_Span]) -> list[_Span]:
        """Merge adjacent spans on the same line with similar font size."""
        if not line_spans:
            return []

        merged: list[_Span] = [line_spans[0]]
        for s in line_spans[1:]:
            prev = merged[-1]
            # merge if font sizes are within 1pt of each other
            if abs(prev.font_size - s.font_size) <= 1.0:
                merged[-1] = _Span(
                    text=f"{prev.text} {s.text}",
                    font_size=max(prev.font_size, s.font_size),
                    font_name=prev.font_name,
                    is_bold=prev.is_bold or s.is_bold,
                    page_num=prev.page_num,
                )
            else:
                merged.append(s)

        return merged


    def _detect_sections(self, spans: list[_Span], raw_text: str) -> list[Section]:
        """
        Detect section headings using font-size heuristics.

        Strategy:
        1. Find the body font size (most common size)
        2. Spans larger than body AND (bold or much larger) → heading
        3. Assign 2 levels: L1 for biggest headings, L2 for the rest
        4. Split raw_text at heading boundaries
        """
        if not spans:
            return [Section(title="Full Document", level=1, text=raw_text, page_start=1)]

        # body font size = most common size weighted by char count
        size_chars: Counter[float] = Counter()
        for s in spans:
            size_chars[s.font_size] += len(s.text)
        body_size = size_chars.most_common(1)[0][0]

        # known heading keywords
        KNOWN_HEADINGS = re.compile(
            r"^(Abstract|Introduction|Conclusion|References|"
            r"Acknowledgm?ents?|Appendix|Related Work|Discussion|"
            r"Methodology|Methods|Results|Experiments?|Background|"
            r"Supervised Fine-Tuning|Reinforcement Learning|"
            r"Math Pre-Training|Limitations?|Future Work)\b",
            re.IGNORECASE,
        )

        heading_spans: list[_Span] = []
        for s in spans:
            text = s.text.strip()

            # filter obvious non-headings
            if not text or len(text) > 200 or len(text) < 3:
                continue
            # skip things that look like citations, arxiv IDs, etc.
            if re.match(r"^arXiv:", text):
                continue

            is_heading = False

            # Rule 1: significantly larger font → heading
            if s.font_size > body_size * 1.2:
                is_heading = True

            # Rule 2: bold + larger-than-body + looks like a heading
            elif s.is_bold and s.font_size >= body_size:
                if len(text) < 100:  # headings are short
                    # numbered heading: "1. Title" or "2.1 Title"
                    if re.match(r"^\d+(\.\d+)*\.?\s+\S", text):
                        is_heading = True
                    # known keyword
                    elif KNOWN_HEADINGS.match(text):
                        is_heading = True

            if is_heading:
                heading_spans.append(s)

        if not heading_spans:
            return [Section(title="Full Document", level=1, text=raw_text, page_start=1)]

        # assign heading levels: just 2 tiers
        # largest font size → L1, everything else → L2
        heading_sizes = sorted(set(h.font_size for h in heading_spans), reverse=True)
        max_heading_size = heading_sizes[0]

        def _level(font_size: float) -> int:
            return 1 if font_size >= max_heading_size - 0.5 else 2

        # find heading positions in raw_text
        sections: list[Section] = []
        heading_positions: list[tuple[int, str, int, int]] = []  # (pos, title, level, page)

        for h in heading_spans:
            clean_title = self.clean_text(h.text).strip()
            if not clean_title:
                continue

            pos = raw_text.find(clean_title)
            if pos == -1:
                # fuzzy match on first 40 chars
                pos = raw_text.find(clean_title[:40])

            if pos >= 0:
                level = _level(h.font_size)
                heading_positions.append((pos, clean_title, level, h.page_num))

        if not heading_positions:
            return [Section(title="Full Document", level=1, text=raw_text, page_start=1)]

        # deduplicate by position, keep first occurrence
        seen: set[int] = set()
        unique: list[tuple[int, str, int, int]] = []
        for pos, title, level, page in sorted(heading_positions):
            # also skip if this position is very close to a previous one
            if any(abs(pos - p) < 5 for p in seen):
                continue
            seen.add(pos)
            unique.append((pos, title, level, page))
        heading_positions = unique

        # preamble: text before first heading
        first_pos = heading_positions[0][0]
        preamble = raw_text[:first_pos].strip()
        if preamble and len(preamble) > 50:
            sections.append(Section(
                title="Preamble",
                level=1,
                text=preamble,
                page_start=1,
            ))

        # split text at each heading boundary
        for i, (pos, title, level, page) in enumerate(heading_positions):
            if i + 1 < len(heading_positions):
                next_pos = heading_positions[i + 1][0]
                section_text = raw_text[pos:next_pos].strip()
            else:
                section_text = raw_text[pos:].strip()

            # remove the heading title from the section body
            if section_text.startswith(title):
                section_text = section_text[len(title):].strip()

            if section_text:
                sections.append(Section(
                    title=title,
                    level=level,
                    text=section_text,
                    page_start=page,
                ))

        return sections


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else None
    if not source:
        print("Usage: python -m python.ingest.pdf <path_or_url>")
        sys.exit(1)

    loader = DocumentLoader()
    doc = loader.load(source)

    print(f"\n{'=' * 60}")
    print(f"  {doc}")
    print(f"  Title:  {doc.metadata.title}")
    print(f"  Author: {doc.metadata.author}")
    print(f"  Hash:   {doc.content_hash[:16]}...")
    print(f"{'=' * 60}")

    print(f"\n  Sections detected: {len(doc.sections)}")
    for s in doc.sections:
        indent = "  " * s.level
        print(f"  {indent}L{s.level}: {s.title!r}  ({len(s.text)} chars)")
    print()