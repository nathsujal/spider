"""
Abstract base loader for the Spider ingestion pipeline.

Every loader (PDF, web, text, CSV, image) subclasses BaseLoader and
implements `_load()` to produce a `Document`.
"""

import re
from abc import ABC, abstractmethod

from python.models.document import Document


class BaseLoader(ABC):
    """
    Every loader subclasses this.

    Subclasses MUST implement `_load` which returns a `Document`.
    The public `load()` method wraps `_load` and adds common post-processing.
    """

    LIGATURES: dict[str, str] = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\ufb06": "st",
        "\u2013": "-",  "\u2014": "-",
        "\u2018": "'",  "\u2019": "'",
        "\u201c": '"',  "\u201d": '"',
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Common text normalisation shared across all loaders."""

        # 1. Replace ligatures & unicode quotes
        for src, dst in cls.LIGATURES.items():
            text = text.replace(src, dst)

        # 2. Re-join hyphenated line breaks  ("knowl-\nedge" → "knowledge")
        text = re.sub(r"-\n(\w)", r"\1", text)

        # 3. Remove standalone page numbers
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 4. Collapse 3+ newlines → paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Single newline inside paragraph → space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # 6. Collapse runs of spaces
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    @classmethod
    def to_paragraphs(cls, text: str, min_length: int = 20) -> list[str]:
        """Split cleaned text into meaningful paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if len(p) >= min_length]

    def load(self, source: str) -> Document:
        """Load and normalise a source into a `Document`."""
        doc = self._load(source)

        # auto-fill raw_text from pages if the subclass didn't set it
        if not doc.raw_text and doc.pages:
            joined = "\n\n".join(p.text for p in doc.pages)
            doc.raw_text = self.clean_text(joined)

        return doc

    @abstractmethod
    def _load(self, source: str) -> Document:
        """Subclasses implement this to do the actual parsing."""
        ...
