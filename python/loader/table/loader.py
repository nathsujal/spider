from pathlib import Path

from .processing import HeaderDetector, NaNHandler, MarkdownSerializer
from .format import RawTable, detect_format
from .parsers import BaseTableReader, CSVReader, TSVReader, ExcelReader, ParquetReader, ODSReader
from ..base import BaseLoader
from python.models import Document, DocumentMetadata, Page, Section


_READERS: dict[str, BaseTableReader] = {
    "csv": CSVReader(),
    "tsv": TSVReader(),
    "excel": ExcelReader(),
    "parquet": ParquetReader(),
    "ods": ODSReader(),
}


class TableLoader(BaseLoader):
    """
    Loads tabular data files into Spider's normalised Document format.

    Supported formats: CSV, TSV, Excel (.xlsx/.xls/.xlsm), Parquet, ODS.

    Section structure
    -----------------
    L1 per table / sheet, L2 per row chunk (default 500 rows):

        L1: "Sales Data"
          L2: "Rows 1–500"
          L2: "Rows 501–1000"
        L1: "Inventory"
          L2: "Rows 1–300"

    Parameters
    ----------
    rows_per_chunk  : int
        Number of data rows per L2 section. Default: 500.
    nan_placeholder : str
        String to fill missing / NaN cells with. Default: "N/A".

    Usage
    -----
    >>> loader = TableLoader()
    >>> doc = loader.load("sales.csv")
    >>> doc = loader.load("https://example.com/data.tsv")
    >>> doc = loader.load("report.xlsx")
    """

    def __init__(
        self,
        *,
        rows_per_chunk: int = 500,
        nan_placeholder: str = "N/A",
    ):
        self.rows_per_chunk = rows_per_chunk
        self._header_detector = HeaderDetector()
        self._nan_handler = NaNHandler(placeholder=nan_placeholder)
        self._serializer = MarkdownSerializer()


    def _load(self, source: str) -> Document:
        path = Path(source.split("?")[0])   # strip query params for ext detection
        raw_bytes = BaseTableReader._fetch(source)
        fmt = detect_format(path, raw_bytes)
        reader = _READERS[fmt]

        raw_tables = reader.read_bytes(raw_bytes, path.stem)
        sections, full_text = self._assemble_sections(raw_tables)
        metadata = self._build_metadata(source, fmt, raw_tables, full_text)
        pages = self._paginate(full_text)

        return Document(
            metadata=metadata,
            pages=pages,
            sections=sections,
            raw_text=full_text,
        )

    
    def _assemble_sections(
        self, raw_tables: list[RawTable]
    ) -> tuple[list[Section], str]:
        """
        Builds the two-level section hierarchy:
            L1 per table/sheet → L2 per row chunk.
        """
        sections:   list[Section] = []
        text_parts: list[str]     = []

        for table in raw_tables:
            # --- header detection (skip for Parquet — already has schema) ---
            if table.headers:
                headers = table.headers
                data_rows = table.rows
            else:
                headers, data_rows = self._header_detector.detect(table.rows)

            # --- fill missing values ---
            clean_rows = self._nan_handler.clean(data_rows)

            # --- chunk rows into L2 sections ---
            l2_sections: list[Section] = []
            for chunk_start in range(0, max(len(clean_rows), 1), self.rows_per_chunk):
                chunk = clean_rows[chunk_start : chunk_start + self.rows_per_chunk]
                chunk_end = chunk_start + len(chunk)
                title = f"Rows {chunk_start + 1}–{chunk_end}"
                text = self._serializer.serialize(headers, chunk)

                if text:
                    l2_sections.append(Section(
                        title=title,
                        level=2,
                        text=text,
                        page_start=1,
                    ))
                    text_parts.append(text)

            # --- L1 section for the whole table/sheet ---
            # body of L1 = summary line (not the full text, that lives in L2)
            l1_summary = (
                f"{table.row_count} rows × {table.col_count} columns. "
                f"Columns: {', '.join(headers[:10])}"
                + (" …" if len(headers) > 10 else "")
            )
            sections.append(Section(
                title=table.name,
                level=1,
                text=l1_summary,
                page_start=1,
            ))
            sections.extend(l2_sections)

        full_text = "\n\n".join(text_parts)
        return sections, full_text

    
    @staticmethod
    def _build_metadata(
        source: str,
        fmt: str,
        raw_tables: list[RawTable],
        full_text: str,
    ) -> DocumentMetadata:
        total_rows = sum(t.row_count for t in raw_tables)
        total_cols = max((t.col_count for t in raw_tables), default=0)
        sheet_names = [t.name for t in raw_tables]

        return DocumentMetadata(
            source=source,
            source_type="table",
            title=raw_tables[0].name if raw_tables else None,
            author=None,
            created_at=None,
            page_count=None,
            extra={
                "format": fmt,
                "sheet_count": len(raw_tables),
                "sheet_names": sheet_names,
                "total_rows": total_rows,
                "total_cols": total_cols,
                "char_count": len(full_text),
            },
        )

    
    @staticmethod
    def _paginate(text: str, chars_per_page: int = 3_000) -> list[Page]:
        pages: list[Page] = []
        for i in range(0, max(len(text), 1), chars_per_page):
            pages.append(Page(number=len(pages) + 1, text=text[i : i + chars_per_page]))
        return pages


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else None
    if not source:
        print("Usage: python -m python.loader.table.loader <path_or_url>")
        sys.exit(1)

    loader = TableLoader()
    doc    = loader.load(source)

    print(f"\n{'=' * 60}")
    print(f"  {doc}")
    print(f"  Title:   {doc.metadata.title}")
    print(f"  Format:  {doc.metadata.extra['format']}")
    print(f"  Sheets:  {doc.metadata.extra['sheet_names']}")
    print(f"  Rows:    {doc.metadata.extra['total_rows']}")
    print(f"  Cols:    {doc.metadata.extra['total_cols']}")
    print(f"  Hash:    {doc.content_hash[:16]}...")
    print(f"{'=' * 60}")

    print(f"\n  Sections ({len(doc.sections)}):")
    for s in doc.sections:
        indent = "  " * s.level
        print(f"  {indent}L{s.level}: {s.title!r}  ({len(s.text)} chars)")
    print()
