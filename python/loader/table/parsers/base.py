import httpx
import pandas as pd
from pathlib import Path

from ..format import RawTable

class BaseTableReader:
    """
    Abstract base for all format-specific readers.

    Subclasses implement read_bytes() which receives raw file bytes
    and returns a list of _RawTable objects (one per sheet/table).
    """

    def read(self, source: str) -> list[RawTable]:
        raw_bytes = self._fetch(source)
        return self.read_bytes(raw_bytes, Path(source).stem)

    def read_bytes(self, raw_bytes: bytes, stem: str) -> list[RawTable]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement read_bytes()")

    @staticmethod
    def _fetch(source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            resp = httpx.get(source, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            return resp.content

        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Table file not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _df_to_raw_table(df: pd.DataFrame, name: str) -> RawTable:
        """Convert a pandas DataFrame (with headers already set) to RawTable."""
        headers = [str(c) for c in df.columns]
        rows    = [list(row) for row in df.itertuples(index=False, name=None)]
        return RawTable(name=name, headers=headers, rows=rows)

