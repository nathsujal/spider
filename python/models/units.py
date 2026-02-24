from dataclasses import dataclass


@dataclass
class Chunk:
    """Atomic text unit for storage in Spider."""

    text: str
    index: int              # global position in document (0-based)
    section_title: str      # parent section title
    page_num: int
    token_count: int = 0
    content_hash: str = ""  # SHA-256 for dedup

    def __repr__(self) -> str:
        return f"Chunk(idx={self.index}, section={self.section_title!r}, tokens={self.token_count})"


@dataclass
class Proposition:
    """Atomic fact extracted from a chunk by the SLM."""

    text: str
    chunk_index: int        # which chunk it was extracted from
    section_title: str      # inherited from parent chunk
    page_num: int           # inherited from parent chunk
    content_hash: str = ""  # SHA-256 for dedup

    def __repr__(self) -> str:
        return f"Prop({self.text[:60]!r})"
