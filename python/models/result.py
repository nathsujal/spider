from dataclasses import dataclass
from typing import Any

from .document import Section


@dataclass
class ParseResult:
    """Intermediate parse result before Document assembly."""
    text: str
    sections: list[Section]
    extra_meta: dict[str, Any]