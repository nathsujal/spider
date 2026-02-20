import io
import zipfile
from typing import Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RawTable:
    """
    Common intermediate representation returned by every reader.
    All downstream processing works against this — never raw pandas or dicts.
    """
    name:    str               # sheet name, filename stem, or "Table N"
    headers: list[str]
    rows:    list[list[Any]]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers)


EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".ods": "ods",
}


def detect_format(path: Path, raw_bytes: bytes | None = None) -> str:
    """
    Detect tabular format from extension first, then content sniffing.
    Falls back to 'csv' for unknown plain-text files.
    """
    ext = path.suffix.lower()
    if ext in EXT_TO_FORMAT:
        return EXT_TO_FORMAT[ext]

    # Content sniffing for extension-less or misnamed files
    if raw_bytes:
        # Parquet magic bytes: PAR1
        if raw_bytes[:4] == b"PAR1":
            return "parquet"
        # ODS is a ZIP with a mimetype entry
        if raw_bytes[:2] == b"PK":
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    if "mimetype" in zf.namelist():
                        mt = zf.read("mimetype").decode()
                        if "opendocument.spreadsheet" in mt:
                            return "ods"
                        if "spreadsheetml" in mt:
                            return "excel"
            except Exception:
                pass
        # TSV heuristic: more tabs than commas in first line
        try:
            first_line = raw_bytes.split(b"\n")[0].decode(errors="replace")
            if first_line.count("\t") > first_line.count(","):
                return "tsv"
        except Exception:
            pass

    return "csv"