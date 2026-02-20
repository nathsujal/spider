import io
import logging
import warnings
from pathlib import Path
from typing import Callable

import httpx
from PIL import Image, UnidentifiedImageError

from .base import BaseLoader
from python.models import Document, DocumentMetadata, Page, Section
from python.engines.tesseract import TesseractOCREngine
from python.engines.blip import BLIPEngine

logger = logging.getLogger(__name__)


_FORMAT_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
    "ICO": "image/x-icon",
    "SVG": "image/svg+xml",
}

def _mime_type(pil_format: str | None) -> str:
    return _FORMAT_TO_MIME.get(pil_format or "", "image/octet-stream")


class ImageLoader(BaseLoader):
    """
    Loads image files into Spider's normalized Document format.

    """

    def __init__(self, ocr_engine=None, vision_engine=None):
        self.ocr_engine = ocr_engine or TesseractOCREngine()
        self.vision_engine = vision_engine or BLIPEngine()

    def _load(self, source: str) -> Document:
        image, pil_format = self._resolve_source(source)

        ocr_blocks = self.ocr_engine.run(image)
        description = self.vision_engine.describe(image)

        metadata = self._build_metadata(source, image, pil_format, description)
        sections = self._build_sections(ocr_blocks)
        raw_text = "\n\n".join(s.text for s in sections)
        pages = [Page(number=1, text=raw_text)]

        return Document(
            metadata=metadata,
            pages=pages,
            sections=sections,
            raw_text=raw_text,
        )

    def _resolve_source(self, source: str) -> tuple[Image.Image, str | None]:
        if source.startswith(("http://", "https://")):
            return self._load_from_url(source)
        return self._load_from_path(source)

    @staticmethod
    def _load_from_path(source: str) -> tuple[Image.Image, str | None]:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            image = Image.open(path)
            image.load()
            return image, image.format
        except UnidentifiedImageError:
            raise ValueError(f"File is not a recognized image format: {path}")

    @staticmethod
    def _load_from_url(url: str) -> tuple[Image.Image, str | None]:
        resp = httpx.get(url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        try:
            image = Image.open(io.BytesIO(resp.content))
            image.load()
            return image, image.format
        except UnidentifiedImageError:
            raise ValueError(f"URL did not return a recognised image: {url}")

    @staticmethod
    def _build_metadata(
        source: str,
        image: Image.Image,
        pil_format: str | None,
        description: str | None,
    ) -> DocumentMetadata:
        width, height = image.size
        return DocumentMetadata(
            source=source,
            source_type="image",
            title=Path(source).stem if not source.startswith("http") else None,
            author=None,
            created_at=None,
            page_count=1,
            extra={
                "format": pil_format or "unknown",
                "mime_type": _mime_type(pil_format),
                "width": width,
                "height": height,
                "mode": image.mode,
                "description": description,
            },
        )

    @staticmethod
    def _build_sections(ocr_blocks: list[str]) -> list[Section]:
        """
        Each OCR paragraph block -> one Section.
        Falls back to a single empty section if OCR yielded nothing.
        """
        if not ocr_blocks:
            return [
                Section(
                    title="Image Content",
                    level=1,
                    text="",
                    page_start=1,
                )
            ]
        
        return [
            Section(
                title=f"Block {i + 1}",
                level=1,
                text=block,
                page_start=1,
            )
            for i, block in enumerate(ocr_blocks)
        ]


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else None
    if not source:
        print("Usage: python -m python.loader.image <image_path>")
        sys.exit(1)

    loader = ImageLoader()
    doc    = loader.load(source)

    print(f"\n{'=' * 60}")
    print(f"  {doc}")
    print(f"  Title:       {doc.metadata.title}")
    print(f"  Format:      {doc.metadata.extra['format']}")
    print(f"  Dimensions:  {doc.metadata.extra['width']} x {doc.metadata.extra['height']}")
    print(f"  Mode:        {doc.metadata.extra['mode']}")
    print(f"  Description: {doc.metadata.extra['description'] or '(no vision callable provided)'}")
    print(f"  Hash:        {doc.content_hash[:16]}...")
    print(f"{'=' * 60}")

    print(f"\n  OCR Sections detected: {len(doc.sections)}")
    for s in doc.sections:
        preview = s.text[:80].replace("\n", " ")
        print(f"    {s.title}: {preview!r}")
    print()