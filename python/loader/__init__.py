from python.models import Document, DocumentMetadata, Page, Section, Chunk, Proposition
from .base import BaseLoader
from .text import TextLoader
from .web import WebLoader
from .table import TableLoader
from .document import DocumentLoader
from .image import ImageLoader

__all__ = [
    "BaseLoader",
    "TextLoader",
    "WebLoader",
    "TableLoader",
    "DocumentLoader",
    "ImageLoader",
    "Document",
    "DocumentMetadata",
    "Page",
    "Section",
    "Chunk",
    "Proposition",
]
