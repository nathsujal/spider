import json

from python.models.document import Section
from python.models.result import ParseResult

class JSONParser:
    """
    Parses a JSON object or array into sections.
    - Object: each top-level key becomes a section.
    - Array: chunked into sections of N items each.
    """

    ITEMS_PER_SECTION = 50

    def parse(self, raw: str) -> ParseResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        sections: list[Section] = []

        if isinstance(data, dict):
            for key, value in data.items():
                text = json.dumps(value, indent=2, ensure_ascii=False)
                sections.append(Section(title=str(key), level=1, text=text, page_start=1))
            full_text = "\n\n".join(s.text for s in sections)

        elif isinstance(data, list):
            for i in range(0, len(data), self.ITEMS_PER_SECTION):
                chunk = data[i : i + self.ITEMS_PER_SECTION]
                text = json.dumps(chunk, indent=2, ensure_ascii=False)
                sections.append(Section(
                    title=f"Items {i + 1}–{i + len(chunk)}",
                    level=1,
                    text=text,
                    page_start=1,
                ))
            full_text = json.dumps(data, indent=2, ensure_ascii=False)

        else:
            full_text = str(data)
            sections = [Section(title="Full Document", level=1, text=full_text, page_start=1)]

        return ParseResult(text=full_text, sections=sections, extra_meta={"json_type": type(data).__name__})


class JSONLParser:
    """
    Parses newline-delimited JSON (.jsonl / .ndjson).
    Each line is a JSON object; lines are batched into sections.
    """

    ITEMS_PER_SECTION = 50

    def parse(self, raw: str) -> ParseResult:
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        items: list[dict] = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines

        if not items:
            return ParseResult(
                text=raw,
                sections=[Section(title="Full Document", level=1, text=raw, page_start=1)],
                extra_meta={},
            )

        sections: list[Section] = []
        for i in range(0, len(items), self.ITEMS_PER_SECTION):
            chunk = items[i : i + self.ITEMS_PER_SECTION]
            text = json.dumps(chunk, indent=2, ensure_ascii=False)
            sections.append(Section(
                title=f"Items {i + 1}–{i + len(chunk)}",
                level=1,
                text=text,
                page_start=1,
            ))

        full_text = json.dumps(items, indent=2, ensure_ascii=False)
        return ParseResult(text=full_text, sections=sections, extra_meta={"line_count": len(items)})
