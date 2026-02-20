import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    """Source-level metadata attached to every ingested document."""

    source: str                       # file path or URL
    source_type: str                  # "pdf", "web", "text", "csv", "image"
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    page_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    """A single page (or logical section) inside a document."""

    number: int
    text: str


@dataclass
class Section:
    """A detected section/heading in the document."""

    title: str
    level: int              # 1 = top-level, 2 = subsection, etc.
    text: str               # full text content under this heading
    page_start: int = 1     # which page it begins on

    def __repr__(self) -> str:
        chars = len(self.text)
        return f"Section(L{self.level}: {self.title!r}, chars={chars})"


@dataclass
class Document:
    """Normalised output of every loader."""

    metadata: DocumentMetadata
    pages: list[Page] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""

    @property
    def content_hash(self) -> str:
        """SHA-256 of the full text — useful for dedup."""
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    @property
    def is_empty(self) -> bool:
        return len(self.raw_text.strip()) == 0

    def __repr__(self) -> str:
        src = self.metadata.source
        pages = len(self.pages)
        secs = len(self.sections)
        chars = len(self.raw_text)
        return f"Document(source={src!r}, pages={pages}, sections={secs}, chars={chars})"