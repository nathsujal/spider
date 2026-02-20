import re
from enum import Enum, auto
from pathlib import Path


class TextFormat(Enum):
    PLAIN = auto()
    MARKDOWN = auto()
    JSON = auto()
    JSONL = auto()
    XML = auto()


_EXT_MAP: dict[str, TextFormat] = {
    ".txt":    TextFormat.PLAIN,
    ".text":   TextFormat.PLAIN,
    ".md":     TextFormat.MARKDOWN,
    ".mdx":    TextFormat.MARKDOWN,
    ".json":   TextFormat.JSON,
    ".jsonl":  TextFormat.JSONL,
    ".ndjson": TextFormat.JSONL,
    ".xml":    TextFormat.XML,
}


def sniff_format(path: Path, raw: str) -> TextFormat:
    """Detect format from extension first, then content heuristics."""
    ext = path.suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    stripped = raw.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<"):
        return TextFormat.XML
    if stripped.startswith("{") or stripped.startswith("["):
        return TextFormat.JSON
    if re.match(r"^#+ ", stripped):
        return TextFormat.MARKDOWN

    return TextFormat.PLAIN
