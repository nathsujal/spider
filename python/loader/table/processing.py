import re
import math
from typing import Any


class HeaderDetector:
    """
    Auto-detects which row should be treated as the header.

    Strategy (in order):
    1. Row 0 is all strings AND row 1 contains at least one numeric  → row 0
    2. Row 0 values look like column names (short, identifier-like)   → row 0
    3. Any row before row 5 that is all-string while next is numeric  → that row
    4. Fall back to row 0 regardless.

    If no rows exist, synthesises Col_1, Col_2, … headers.
    """

    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]{0,40}$")

    def detect(self, raw_rows: list[list[Any]]) -> tuple[list[str], list[list[Any]]]:
        """
        Returns (headers, data_rows) after promoting the header row.
        """
        if not raw_rows:
            return [], []

        # Try to find header in first 5 rows
        header_idx = self._find_header_row(raw_rows[:6])
        headers    = [str(v).strip() for v in raw_rows[header_idx]]
        data_rows  = raw_rows[header_idx + 1:]

        # Deduplicate blank / duplicate column names
        headers = self._sanitise_headers(headers)
        return headers, data_rows

    def _find_header_row(self, rows: list[list[Any]]) -> int:
        for i, row in enumerate(rows[:-1]):
            next_row = rows[i + 1]
            if self._all_strings(row) and self._has_numeric(next_row):
                return i
            if self._looks_like_identifiers(row):
                return i
        return 0   # safe fallback

    @staticmethod
    def _all_strings(row: list[Any]) -> bool:
        return all(isinstance(v, str) for v in row if v is not None)

    @staticmethod
    def _has_numeric(row: list[Any]) -> bool:
        return any(isinstance(v, (int, float)) for v in row)

    def _looks_like_identifiers(self, row: list[Any]) -> bool:
        if not row:
            return False
        matches = sum(
            1 for v in row
            if isinstance(v, str) and self._IDENTIFIER_RE.match(v.strip())
        )
        return matches / len(row) >= 0.6   # 60% of cells look like identifiers

    @staticmethod
    def _sanitise_headers(headers: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result: list[str] = []
        for i, h in enumerate(headers):
            h = h.strip() or f"Col_{i + 1}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            result.append(h)
        return result


class NaNHandler:
    """Replaces missing / NaN values with a configurable placeholder."""

    def __init__(self, placeholder: str = "N/A"):
        self.placeholder = placeholder

    def clean(self, rows: list[list[Any]]) -> list[list[str]]:
        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append([
                self.placeholder if self._is_missing(v) else str(v).strip()
                for v in row
            ])
        return cleaned

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            if isinstance(value, float) and math.isnan(value):
                return True
        except Exception:
            pass
        return str(value).strip().lower() in ("nan", "none", "null", "na", "n/a", "")


class MarkdownSerializer:
    """
    Converts a list of cleaned string rows into a GitHub-flavoured
    markdown table, including the header and separator rows.

    Output example:
        | Name  | Age | City |
        |-------|-----|------|
        | Alice | 30  | NYC  |
        | Bob   | 25  | LA   |
    """

    def serialize(self, headers: list[str], rows: list[list[str]]) -> str:
        if not headers:
            return ""

        # Column widths — max of header and all cell widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))

        def _row_str(cells: list[str]) -> str:
            padded = [
                cells[i].ljust(widths[i]) if i < len(widths) else cells[i]
                for i in range(len(headers))
            ]
            return "| " + " | ".join(padded) + " |"

        separator = "| " + " | ".join("-" * w for w in widths) + " |"

        lines = [_row_str(headers), separator]
        for row in rows:
            # pad short rows with placeholder
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append(_row_str(padded_row[: len(headers)]))

        return "\n".join(lines)