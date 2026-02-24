import io
import pandas as pd

from .base import BaseTableReader
from ..format import RawTable


class ParquetReader(BaseTableReader):
    """
    Reads Parquet files. Column names from schema are used directly
    as headers (Parquet always has a schema, so header detection is skipped).

    Requires: pip install pyarrow
    """

    def read_bytes(self, raw_bytes: bytes, stem: str) -> list[RawTable]:
        try:
            df = pd.read_parquet(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to parse Parquet file: {exc}") from exc

        # Parquet has a reliable schema — use it directly, skip header detection
        headers = [str(c) for c in df.columns]
        rows    = [
            [str(v) if v is not None else "" for v in row]
            for row in df.itertuples(index=False, name=None)
        ]
        return [RawTable(name=stem, headers=headers, rows=rows)]