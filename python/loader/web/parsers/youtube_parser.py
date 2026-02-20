import re
import httpx
from dataclasses import dataclass
from urllib.parse import urlparse

from python.models import Section
from ..utils import get


@dataclass
class YoutubeMetadata:
    video_id: str
    title:    str | None
    channel:  str | None
    description: str | None
    upload_date: str | None
    duration:    str | None


class YoutubeHandler:
    """
    Extracts metadata + transcript from a YouTube URL.

    Metadata  — scraped from static HTML <meta> tags (no JS needed).
    Transcript — fetched via youtube-transcript-api, grouped into
                 2-minute windows to form sections.
    """

    _WINDOW_SECONDS = 120   # group captions into 2-minute sections

    # Video ID patterns
    _ID_FROM_QUERY = re.compile(r"(?:^|&)v=([A-Za-z0-9_-]{11})")
    _ID_FROM_PATH  = re.compile(r"^/([A-Za-z0-9_-]{11})$")   # youtu.be/<id>

    def fetch(self, url: str) -> tuple[YoutubeMetadata, list[Section]]:
        video_id = self._extract_video_id(url)
        metadata = self._scrape_metadata(url, video_id)
        sections = self._build_transcript_sections(video_id)
        return metadata, sections

    
    def _extract_video_id(self, url: str) -> str:
        parsed = urlparse(url)

        # youtu.be/<id>
        if m := self._ID_FROM_PATH.match(parsed.path):
            return m.group(1)

        # youtube.com/watch?v=<id>  or  /embed/<id>  or  /v/<id>
        if m := self._ID_FROM_QUERY.search(parsed.query):
            return m.group(1)

        # /embed/<id> or /v/<id>
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[-2] in ("embed", "v", "shorts"):
            return parts[-1]

        raise ValueError(f"Cannot extract video ID from URL: {url}")

    
    def _scrape_metadata(self, url: str, video_id: str) -> YoutubeMetadata:
        resp = get(f"https://www.youtube.com/watch?v={video_id}")
        html = resp.text

        def _meta(name: str) -> str | None:
            # covers both property= and name= variants
            m = re.search(
                rf'<meta[^>]+(?:property|name)="{re.escape(name)}"[^>]+content="([^"]*)"',
                html,
            )
            return m.group(1) if m else None

        return YoutubeMetadata(
            video_id=video_id,
            title=_meta("og:title"),
            channel=_meta("og:site_name") or _meta("twitter:creator"),
            description=_meta("og:description"),
            upload_date=_meta("datePublished"),
            duration=_meta("duration"),
        )

    
    def _build_transcript_sections(self, video_id: str) -> list[Section]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "youtube-transcript-api is required for YouTube loading. "
                "Install it with: pip install youtube-transcript-api"
            )

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        snippets = list(transcript)

        if not snippets:
            return [Section(title="Transcript", level=1, text="", page_start=1)]

        # Group caption entries into _WINDOW_SECONDS-wide time buckets
        sections: list[Section] = []
        bucket_start = 0.0
        bucket_texts: list[str] = []

        def _flush(start: float, end: float, texts: list[str]) -> None:
            if not texts:
                return
            def _fmt(secs: float) -> str:
                m, s = divmod(int(secs), 60)
                return f"{m}:{s:02d}"
            title = f"{_fmt(start)} – {_fmt(end)}"
            sections.append(Section(
                title=title,
                level=1,
                text=" ".join(texts),
                page_start=1,
            ))

        for snippet in snippets:
            start: float = snippet.start
            text: str    = snippet.text.strip()

            if start - bucket_start >= self._WINDOW_SECONDS and bucket_texts:
                _flush(bucket_start, start, bucket_texts)
                bucket_start = start
                bucket_texts = []

            bucket_texts.append(text)

        # flush final bucket
        if snippets:
            last_start = snippets[-1].start
            _flush(bucket_start, last_start, bucket_texts)

        return sections