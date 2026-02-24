import re

from python.models.document import Section
from python.models.result import ParseResult

class MarkdownParser:
    """
    Parse Markdown headings (# through ######) into sections, preserving the
    full depth as the section level (1–6). Code fences are masked before
    heading detection so that # lines inside code blocks are never mistaken
    for headings.
    """

    _CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

    # Safe inline patterns — no backreferences, no catastrophic backtracking
    _BOLD_STAR = re.compile(r"\*\*([^*]+)\*\*")
    _BOLD_UNDER = re.compile(r"__([^_]+)__")
    _ITALIC_STAR = re.compile(r"\*([^*]+)\*")
    _ITALIC_UNDER = re.compile(r"(?<!\w)_([^_]+)_(?!\w)")
    _INLINE_CODE = re.compile(r"`([^`]+)`")
    _LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    _IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")

    def parse(self, raw: str) -> ParseResult:
        # Mask code-fence content with spaces so heading regex never fires
        # inside a fenced block, while keeping character positions intact.
        def _mask_fence(m: re.Match) -> str:
            return " " * len(m.group(0))

        masked = self._CODE_FENCE.sub(_mask_fence, raw)
        headings = list(self._HEADING.finditer(masked))

        clean_full = self._strip_markdown(raw).strip()

        if not headings:
            return ParseResult(
                text=clean_full,
                sections=[Section(title="Full Document", level=1, text=clean_full, page_start=1)],
                extra_meta={},
            )

        # Preserve the true heading depth (1–6) directly as the section level.
        sections: list[Section] = []
        positions = [
            (m.start(), m.end(), len(m.group(1)), m.group(2).strip())
            for m in headings
        ]

        # preamble: text before the very first heading
        if positions[0][0] > 0:
            preamble = self._strip_markdown(raw[: positions[0][0]]).strip()
            if preamble:
                sections.append(Section(title="Preamble", level=1, text=preamble, page_start=1))

        for i, (start, end, depth, title) in enumerate(positions):
            body_end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
            body = self._strip_markdown(raw[end:body_end]).strip()
            sections.append(Section(
                title=title,
                level=depth,
                text=body,
                page_start=1,
            ))

        self._normalize_levels(sections)
        return ParseResult(text=clean_full, sections=sections, extra_meta={})

    @staticmethod
    def _normalize_levels(sections: list[Section]) -> None:
        """Remap section levels to a gapless 1, 2, 3... sequence in-place."""
        heading_depths = sorted(set(
            s.level for s in sections if s.title != "Preamble"
        ))
        if not heading_depths:
            return
        depth_to_level = {depth: rank for rank, depth in enumerate(heading_depths, start=1)}
        for s in sections:
            if s.title != "Preamble":
                s.level = depth_to_level[s.level]

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting, returning plain text."""
        text = self._CODE_FENCE.sub("", text)
        text = self._HEADING.sub(r"\2", text)
        text = self._IMAGE.sub(r"\1", text)
        text = self._LINK.sub(r"\1", text)
        text = self._BOLD_STAR.sub(r"\1", text)
        text = self._BOLD_UNDER.sub(r"\1", text)
        text = self._ITALIC_STAR.sub(r"\1", text)
        text = self._ITALIC_UNDER.sub(r"\1", text)
        text = self._INLINE_CODE.sub(r"\1", text)
        text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^>+\s?", "", text, flags=re.MULTILINE)
        return text.strip()