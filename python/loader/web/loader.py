from .parsers import GithubHandler, WebParser, YoutubeHandler
from .utils import detect_url_type, get
from ..base import BaseLoader
from ..text import TextLoader, TextFormat
from ..text.format import sniff_format
from ..text.loader import _PARSERS
from python.models import Document, DocumentMetadata, Page, Section

class WebLoader(BaseLoader):
    """
    Loads web content into Spider's normalised Document format.

    Supports:
      - Regular web pages  (semantic extraction, h1–h6 sections)
      - GitHub URLs        (raw file content or repo README)
      - YouTube URLs       (page metadata + timed transcript sections)

    Usage
    -----
    >>> loader = WebLoader()
    >>> doc = loader.load("https://github.com/anthropics/anthropic-sdk-python")
    >>> doc = loader.load("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    >>> doc = loader.load("https://docs.python.org/3/library/pathlib.html")
    """

    def __init__(self, *, timeout: int = 30):
        self.timeout = timeout

    
    def _load(self, source: str) -> Document:
        url_type = detect_url_type(source)

        if url_type == "github":
            return self._load_github(source)
        if url_type == "youtube":
            return self._load_youtube(source)
        return self._load_web(source)

    
    def _load_github(self, url: str) -> Document:
        handler = GithubHandler()
        raw_content, ext, resolved_url = handler.fetch(url)

        # Delegate to TextLoader's existing parsers by extension
        fake_path = Path(f"file{ext}")
        fmt = sniff_format(fake_path, raw_content)
        result = _PARSERS[fmt].parse(raw_content)
        clean = self.clean_text(result.text)

        metadata = DocumentMetadata(
            source=url,
            source_type="github",
            title=result.sections[0].title if result.sections else None,
            author=None,
            created_at=None,
            page_count=None,
            extra={
                "resolved_url": resolved_url,
                "format": fmt.name,
                "char_count": len(raw_content),
                **result.extra_meta,
            },
        )
        return Document(
            metadata=metadata,
            pages=self._paginate(clean),
            sections=result.sections,
            raw_text=clean,
        )

    
    def _load_youtube(self, url: str) -> Document:
        handler = YoutubeHandler()
        yt_meta, sections = handler.fetch(url)

        full_text = "\n\n".join(s.text for s in sections)
        clean = self.clean_text(full_text)

        metadata = DocumentMetadata(
            source=url,
            source_type="youtube",
            title=yt_meta.title,
            author=yt_meta.channel,
            created_at=yt_meta.upload_date,
            page_count=None,
            extra={
                "video_id":    yt_meta.video_id,
                "description": yt_meta.description,
                "duration":    yt_meta.duration,
            },
        )
        return Document(
            metadata=metadata,
            pages=self._paginate(clean),
            sections=sections,
            raw_text=clean,
        )

    
    def _load_web(self, url: str) -> Document:
        resp = get(url, timeout=self.timeout)
        parser = WebParser()
        full_text, sections, page_title = parser.parse(resp.text, base_url=url)
        clean = self.clean_text(full_text)

        metadata = DocumentMetadata(
            source=url,
            source_type="web",
            title=page_title,
            author=None,
            created_at=None,
            page_count=None,
            extra={"char_count": len(resp.text)},
        )
        return Document(
            metadata=metadata,
            pages=self._paginate(clean),
            sections=sections,
            raw_text=clean,
        )

    
    @staticmethod
    def _paginate(text: str, chars_per_page: int = 3_000) -> list[Page]:
        pages: list[Page] = []
        for i in range(0, max(len(text), 1), chars_per_page):
            pages.append(Page(number=len(pages) + 1, text=text[i : i + chars_per_page]))
        return pages


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        print("Usage: python -m python.loader.web.loader <url>")
        sys.exit(1)

    loader = WebLoader()
    doc    = loader.load(url)

    print(f"\n{'=' * 60}")
    print(f"  {doc}")
    print(f"  Title:   {doc.metadata.title}")
    print(f"  Type:    {doc.metadata.source_type}")
    print(f"  Author:  {doc.metadata.author}")
    print(f"  Hash:    {doc.content_hash[:16]}...")
    print(f"{'=' * 60}")

    print(f"\n  Pages (virtual): {len(doc.pages)}")
    print(f"  Sections detected: {len(doc.sections)}")
    for s in doc.sections:
        indent = "  " * s.level
        print(f"  {indent}L{s.level}: {s.title!r}  ({len(s.text)} chars)")
    print()