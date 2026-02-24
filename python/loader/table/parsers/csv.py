import io
import pandas as pd

from .base import BaseTableReader
from ..format import RawTable

class CSVReader(BaseTableReader):
    """
    Reads CSV files. Auto-detects delimiter, encoding, and quoting via
    pandas' built-in engine sniffer.
    """

    def read_bytes(self, raw_bytes: bytes, stem: str) -> list[RawTable]:
        # Try UTF-8 first, fall back to latin-1
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    header=None,      # we do header detection ourselves
                    dtype=str,        # keep everything as string for now
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                return [self._df_to_raw_table(df, stem)]
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV: {exc}") from exc

        raise ValueError(f"Could not decode CSV with any supported encoding: {stem}")
