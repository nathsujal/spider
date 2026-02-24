import io
import pandas as pd

from .base import BaseTableReader
from ..format import RawTable

class TSVReader(BaseTableReader):
    """Reads TSV (tab-separated) files."""

    def read_bytes(self, raw_bytes: bytes, stem: str) -> list[RawTable]:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    sep="\t",
                    header=None,
                    dtype=str,
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                return [self._df_to_raw_table(df, stem)]
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ValueError(f"Failed to parse TSV: {exc}") from exc

        raise ValueError(f"Could not decode TSV: {stem}")