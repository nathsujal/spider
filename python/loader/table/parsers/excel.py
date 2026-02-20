import io
import pandas as pd

from .base import BaseTableReader
from ..format import RawTable

class ExcelReader(BaseTableReader):
    """
    Reads Excel files (.xlsx / .xls / .xlsm).
    Each sheet becomes a separate _RawTable.

    Requires: pip install openpyxl xlrd
    """

    def read_bytes(self, raw_bytes: bytes, stem: str) -> list[RawTable]:
        try:
            xl = pd.ExcelFile(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ValueError(f"Failed to open Excel file: {exc}") from exc

        tables: list[RawTable] = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, header=None, dtype=str)
            if df.empty:
                continue
            tables.append(self._df_to_raw_table(df, str(sheet_name)))

        return tables or [RawTable(name=stem, headers=[], rows=[])]