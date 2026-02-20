"""
Source Router — maps any source string to the correct Loader.

Handles file extensions, URLs, and ambiguous inputs with content sniffing.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

from python.loader import BaseLoader, TextLoader, WebLoader, TableLoader, DocumentLoader, ImageLoader

logger = logging.getLogger(__name__)


# Extension → Loader class mapping

_DOCUMENT_EXTS = {".pdf"}

_TABLE_EXTS = {".csv", ".tsv", ".tab", ".xlsx", ".xls", ".xlsm", ".parquet", ".pq", ".ods"}

_TEXT_EXTS = {
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".log", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".sh", ".bash", ".zsh",
    ".sql", ".r", ".m", ".swift", ".kt",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".ico", ".svg"}


def _ext_to_loader(ext: str) -> type[BaseLoader] | None:
    """Map a file extension to its loader class. Returns None if unknown."""
    if ext in _DOCUMENT_EXTS:
        return DocumentLoader
    if ext in _TABLE_EXTS:
        return TableLoader
    if ext in _TEXT_EXTS:
        return TextLoader
    if ext in _IMAGE_EXTS:
        return ImageLoader
    return None


# URL detection

def _is_url(source: str) -> bool:
    """Check if the source is a URL."""
    return source.startswith(("http://", "https://"))


def _url_to_loader(url: str) -> type[BaseLoader]:
    """
    Route a URL to the right loader.
    
    URLs pointing to known file types (e.g., .pdf, .csv) use the
    file-specific loader. Everything else goes to WebLoader.
    """
    parsed = urlparse(url)
    path = Path(parsed.path)
    ext = path.suffix.lower()

    # URL pointing to a direct file download?
    loader_cls = _ext_to_loader(ext)
    if loader_cls and loader_cls is not TextLoader:
        # Don't send .html/.htm URLs to TextLoader — use WebLoader
        return loader_cls

    return WebLoader


# Public API

def route(source: str) -> BaseLoader:
    """
    Return an instantiated loader for the given source.

    Routing logic:
        1. URLs → check extension, fall back to WebLoader
        2. Files → map extension to loader
        3. Unknown → raise ValueError

    Examples::

        route("paper.pdf")              # → DocumentLoader()
        route("data.csv")               # → TableLoader()
        route("notes.md")               # → TextLoader()
        route("photo.jpg")              # → ImageLoader()
        route("https://example.com")    # → WebLoader()
        route("https://x.com/data.csv") # → TableLoader()
    """
    source = source.strip()

    if not source:
        raise ValueError("Empty source string")

    # URLs
    if _is_url(source):
        loader_cls = _url_to_loader(source)
        logger.debug(f"Routed URL to {loader_cls.__name__}: {source}")
        return loader_cls()

    # Local files
    ext = Path(source).suffix.lower()

    if not ext:
        raise ValueError(
            f"Cannot determine file type (no extension): {source!r}"
        )

    loader_cls = _ext_to_loader(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file extension: {ext!r}. "
            f"Supported: {sorted(_DOCUMENT_EXTS | _TABLE_EXTS | _TEXT_EXTS | _IMAGE_EXTS)}"
        )

    logger.debug(f"Routed {ext} to {loader_cls.__name__}: {source}")
    return loader_cls()


def supported_extensions() -> list[str]:
    """Return all supported file extensions, sorted."""
    return sorted(_DOCUMENT_EXTS | _TABLE_EXTS | _TEXT_EXTS | _IMAGE_EXTS)
