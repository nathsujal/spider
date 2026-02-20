import logging
from pathlib import Path
from typing import Any

from spider import PySpiderDB
from . import router
from python.core.graph import GraphBuilder
from python.intelligence.chunker import Chunker
from python.intelligence.extractors import PropositionExtractor
from python.intelligence.slm import OllamaClient
from python.models import Document, Proposition

logger = logging.getLogger(__name__)


class Spider:
    """
    The main interface for the Spider Knowledge Graph.

    Usage:
        with Spider("./my_graph") as db:
            db.add("paper.pdf")
            db.add("image.png")
            print(db.stats())
    """

    def __init__(
        self,
        db_path: str = "./spider.db",
        *,
        chunk_size: int = 512,
        extract: bool = True,
        slm: OllamaClient | None = None,
    ):
        """
        Initialize the Spider knowledge graph.

        Args:
            db_path: Path to the RocksDB/SpiderDB folder.
            chunk_size: Token size for chunking documents.
            extract: If True, uses SLM to extract propositions.
            slm: Optional custom SLM client (defaults to Ollama).
        """
        self.db_path = db_path
        self._db = PySpiderDB(db_path)
        self._graph = GraphBuilder(self._db)
        
        self._chunker = Chunker(chunk_size=chunk_size)
        
        if extract:
            self._slm = slm or OllamaClient()
            self._extractor = PropositionExtractor(client=self._slm)
        else:
            self._slm = None
            self._extractor = None

    def add(self, source: str) -> dict[str, Any]:
        """
        Ingest a source into the knowledge graph.

        Supports:
        - PDF (.pdf)
        - Text/Code (.md, .txt, .py, etc.)
        - Tables (.csv, .xlsx, .parquet)
        - Images (.jpg, .png) -> Caption + OCR
        - URLs (http://...)
        """
        logger.info(f"Adding source: {source}")

        # 1. Route & Load
        loader = router.route(source)
        doc = loader.load(source)

        if doc.is_empty:
            logger.warning(f"Source yield no content: {source}")
            return {"status": "empty"}

        # 2. Check Dedup (Fast path)
        existing_nodes = self._db.find_nodes_by_property("content_hash", doc.content_hash)
        if existing_nodes:
            logger.info(f"Document already ingested: {source}")
            return {"status": "skipped_duplicate", "doc_id": existing_nodes[0]}

        # 3. Chunk
        chunks = self._chunker.chunk(doc)
        logger.debug(f"Created {len(chunks)} chunks")

        # 4. Extract
        propositions: list[Proposition] = []
        if self._extractor and self._slm and self._slm.is_available():
            logger.info(f"Extracting propositions from {len(chunks)} chunks...")
            propositions = self._extractor.extract(chunks)
        elif self._extractor:
            logger.warning("SLM not available. Skipping extraction.")

        # 5. Build Graph
        stats = self._graph.build(doc, propositions)
        return stats

    def status(self) -> dict[str, Any]:
        """Return basic database statistics."""
        return {
            "node_count": self._db.node_count(),
            # "content_store": self._db.content_stats(), # TODO: expose in PySpiderDB if needed
            "path": self.db_path,
        }

    def close(self):
        """Close the database connection."""
        if self._db and self._db.is_open():
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
