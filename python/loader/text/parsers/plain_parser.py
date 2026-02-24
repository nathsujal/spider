import re

from python.models.document import Section
from python.models.result import ParseResult


class PlainParser:
    """
    Split plain text into sections on blank-line-separated blocks.
    Falls back to a single section when the text is short.
    """

    MIN_SECTION_LENGTH = 200

    def parse(self, raw: str) -> ParseResult:
        text = raw.strip()
        paragraphs = re.split(r"\n{2,}", text)

        if len(paragraphs) <= 3 or len(text) < self.MIN_SECTION_LENGTH:
            return ParseResult(
                text=text,
                sections=[Section(title="Full Document", level=1, text=text, page_start=1)],
                extra_meta={},
            )

        # group into logical chunks (~5 paragraphs each) as L1 sections
        chunk_size = 5
        sections: list[Section] = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk = "\n\n".join(paragraphs[i : i + chunk_size]).strip()
            if chunk:
                sections.append(Section(
                    title=f"Part {i // chunk_size + 1}",
                    level=1,
                    text=chunk,
                    page_start=1,
                ))

        return ParseResult(text=text, sections=sections, extra_meta={})
