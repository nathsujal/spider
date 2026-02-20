import re
from html.parser import HTMLParser

from python.models import Section


class WebParser:
    """
    Extracts clean readable text from an arbitrary web page.

    Strategy
    --------
    1. Prefer semantic containers: <main>, <article>, <div role="main">
       over the raw <body> — avoids navbars, sidebars, footers.
    2. Strip noise tags: <nav>, <header>, <footer>, <aside>,
       <script>, <style>, and elements whose class/id contains
       ad-related keywords.
    3. Collect text and heading positions in a single pass.
    4. Split into sections at heading boundaries.
    5. Normalise levels to a gapless 1, 2, 3… sequence (same logic
       as _MarkdownParser._normalize_levels).
    """

    _NOISE_TAGS  = {"nav", "header", "footer", "aside", "script", "style",
                    "noscript", "iframe", "form", "button", "svg"}
    _NOISE_ATTRS = re.compile(
        r"\b(ad|ads|advert|banner|cookie|popup|modal|sidebar|nav|menu"
        r"|footer|header|promo|sponsor)\b",
        re.IGNORECASE,
    )
    _SEMANTIC    = ("main", "article")
    _HEADING_RE  = re.compile(r"^h([1-6])$")

    def parse(self, html: str, base_url: str = "") -> tuple[str, list[Section], str | None]:
        """
        Returns (full_text, sections, page_title).
        """
        collector = self._Collector(
            noise_tags=self._NOISE_TAGS,
            noise_attrs=self._NOISE_ATTRS,
            semantic=self._SEMANTIC,
            heading_re=self._HEADING_RE,
        )
        collector.feed(html)

        page_title = collector.page_title
        chunks     = collector.chunks   # list of (text, depth | None)

        full_text = "\n\n".join(t for t, _ in chunks).strip()

        if not any(d is not None for _, d in chunks):
            return full_text, [Section(title="Full Document", level=1,
                                        text=full_text, page_start=1)], page_title

        # Build sections at heading boundaries
        sections: list[Section] = []
        pending_heading: str | None = None
        pending_depth:   int        = 1
        pending_body:    list[str]  = []

        def _flush() -> None:
            body = "\n\n".join(pending_body).strip()
            if body or pending_heading:
                sections.append(Section(
                    title=pending_heading or "Preamble",
                    level=pending_depth,
                    text=body,
                    page_start=1,
                ))

        for text, depth in chunks:
            if depth is not None:          # it's a heading
                _flush()
                pending_body    = []
                pending_heading = text
                pending_depth   = depth
            else:
                pending_body.append(text)

        _flush()
        self._normalize_levels(sections)
        return full_text, sections, page_title

    
    @staticmethod
    def _normalize_levels(sections: list[Section]) -> None:
        heading_depths = sorted(set(
            s.level for s in sections if s.title != "Preamble"
        ))
        if not heading_depths:
            return
        depth_to_level = {d: r for r, d in enumerate(heading_depths, start=1)}
        for s in sections:
            if s.title != "Preamble":
                s.level = depth_to_level[s.level]

    
    class _Collector(HTMLParser):
        """
        Single-pass HTML parser that:
          - locates the best semantic container
          - strips noise tags
          - emits (text, depth | None) chunks
        """

        def __init__(self, *, noise_tags, noise_attrs, semantic, heading_re):
            super().__init__()
            self._noise_tags  = noise_tags
            self._noise_attrs = noise_attrs
            self._semantic    = semantic
            self._heading_re  = heading_re

            # state
            self._depth_stack: list[str] = []   # open tag names
            self._skip_depth:  int | None = None # depth at which we entered a noise tag
            self._in_semantic: bool = False
            self._semantic_depth: int = 0
            self._in_heading:  str | None = None # current h1-h6 tag
            self._buf:         list[str] = []

            self.chunks:     list[tuple[str, int | None]] = []
            self.page_title: str | None = None
            self._in_title:  bool = False
            self._title_buf: list[str] = []

        
        def handle_starttag(self, tag: str, attrs: list) -> None:
            self._depth_stack.append(tag)
            attr_dict = dict(attrs)
            combined  = " ".join(filter(None, [
                attr_dict.get("class", ""),
                attr_dict.get("id", ""),
                attr_dict.get("role", ""),
            ]))

            # page <title>
            if tag == "title":
                self._in_title = True
                return

            # enter semantic container (only outermost counts)
            if not self._in_semantic and tag in self._semantic:
                self._in_semantic    = True
                self._semantic_depth = len(self._depth_stack)
                return

            # noise suppression — skip entire subtree
            if self._skip_depth is None:
                is_noise_tag  = tag in self._noise_tags
                is_noise_attr = bool(self._noise_attrs.search(combined))
                if is_noise_tag or is_noise_attr:
                    self._skip_depth = len(self._depth_stack)
                    return

            if self._skip_depth is not None:
                return

            # heading open
            if m := self._heading_re.match(tag):
                self._flush_buf(depth=None)
                self._in_heading = tag

        def handle_endtag(self, tag: str) -> None:
            current_depth = len(self._depth_stack)

            # close title
            if tag == "title" and self._in_title:
                self.page_title = "".join(self._title_buf).strip() or None
                self._title_buf = []
                self._in_title  = False

            # exit noise subtree
            if self._skip_depth is not None:
                if current_depth < self._skip_depth:
                    self._skip_depth = None
                if self._depth_stack:
                    self._depth_stack.pop()
                return

            # close heading
            if self._in_heading == tag:
                if m := self._heading_re.match(tag):
                    self._flush_buf(depth=int(m.group(1)))
                self._in_heading = None

            # exit semantic container
            if self._in_semantic and current_depth < self._semantic_depth:
                self._in_semantic = False

            if self._depth_stack:
                self._depth_stack.pop()

        def handle_data(self, data: str) -> None:
            text = data.strip()
            if not text:
                return

            if self._in_title:
                self._title_buf.append(data)
                return

            if self._skip_depth is not None:
                return

            # only collect if inside semantic container (or no semantic found yet)
            self._buf.append(text)

        
        def _flush_buf(self, depth: int | None) -> None:
            text = " ".join(self._buf).strip()
            self._buf = []
            if text:
                self.chunks.append((text, depth))