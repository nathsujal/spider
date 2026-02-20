# """
# Ingester: orchestrates the full pipeline from source file to SpiderDB graph.

# Flow: Source → Loader → Document → Chunker → Chunks → SLM → Propositions → SpiderDB

# Chunks are intermediate (not stored). Only DOCUMENT, SECTION, and
# PROPOSITION nodes are written to the graph.
# """

# import hashlib

# try:
#     from python._spider import PySpiderDB
# except ImportError:
#     PySpiderDB = None  # Rust bindings not built yet
# from python.ingest.base import Chunk, Document, Proposition, Section
# from python.ingest.pdf import PDFLoader
# from python.intelligence.chunker import Chunker
# from python.intelligence.extractors import PropositionExtractor
# from python.intelligence.slm import OllamaClient


# class Ingester:
#     """
#     End-to-end ingestion pipeline for Spider.

#     Usage:
#         db = PySpiderDB("./memory.db")
#         ingester = Ingester(db)
#         ingester.ingest("paper.pdf")
#     """

#     def __init__(
#         self,
#         db: PySpiderDB,
#         *,
#         slm: OllamaClient | None = None,
#         chunk_size: int = 512,
#     ):
#         self.db = db
#         self.slm = slm or OllamaClient()
#         self.chunker = Chunker(chunk_size=chunk_size)
#         self.extractor = PropositionExtractor(client=self.slm)

#     def ingest(self, source: str) -> dict:
#         """
#         Ingest a source file into SpiderDB.

#         Returns a summary dict with node counts and IDs.
#         """
#         # ── Step 1: Load ──────────────────────────────────────────────
#         print(f"[1/4] Loading: {source}")
#         doc = PDFLoader().load(source)
#         print(f"       {doc}")

#         # ── Step 2: Chunk (intermediate) ──────────────────────────────
#         print(f"[2/4] Chunking...")
#         chunks = self.chunker.chunk(doc)
#         print(f"       {len(chunks)} chunks")

#         # ── Step 3: Extract propositions (optional SLM step) ─────────
#         props: list[Proposition] = []
#         if self.slm.is_available():
#             print(f"[3/4] Extracting propositions via {self.slm.model}...")
#             props = self.extractor.extract(chunks)
#             print(f"       {len(props)} propositions extracted")
#         else:
#             print(f"[3/4] SLM not available — skipping proposition extraction")

#         # ── Step 4: Write to graph ────────────────────────────────────
#         print(f"[4/4] Writing to SpiderDB...")
#         result = self._build_graph(doc, chunks, props)
#         print(f"       Done. {result['total_nodes']} nodes, {result['total_rels']} relationships")

#         return result

#     def _build_graph(self, doc: Document, chunks: list[Chunk], props: list[Proposition]) -> dict:
#         """Map the processed document into SpiderDB nodes and relationships."""

#         stats = {"doc_id": 0, "section_ids": [], "prop_ids": [], "total_nodes": 0, "total_rels": 0}

#         # ── DOCUMENT node ─────────────────────────────────────────────
#         doc_id = self.db.create_node(["DOCUMENT"])
#         self.db.set_node_property(doc_id, "source", doc.metadata.source)
#         if doc.metadata.title:
#             self.db.set_node_property(doc_id, "title", doc.metadata.title)
#         self.db.set_node_property(doc_id, "page_count", doc.metadata.page_count)
#         self.db.set_significance(doc_id, 128)
#         stats["doc_id"] = doc_id
#         stats["total_nodes"] += 1

#         # ── SECTION nodes ─────────────────────────────────────────────
#         section_ids: dict[str, int] = {}  # title → node_id
#         prev_sec_id = None

#         for section in doc.sections:
#             sec_id = self.db.create_node(["SECTION"])
#             self.db.set_node_property(sec_id, "title", section.title)
#             self.db.set_node_property(sec_id, "level", section.level)
#             self.db.set_node_property(sec_id, "page_start", section.page_start)
#             self.db.set_significance(sec_id, 96)

#             # DOCUMENT → CONTAINS → SECTION
#             self.db.create_rel(doc_id, sec_id, "CONTAINS")
#             stats["total_rels"] += 1

#             # SECTION → NEXT → SECTION (reading order)
#             if prev_sec_id is not None:
#                 self.db.create_rel(prev_sec_id, sec_id, "NEXT")
#                 stats["total_rels"] += 1
#             prev_sec_id = sec_id

#             section_ids[section.title] = sec_id
#             stats["section_ids"].append(sec_id)
#             stats["total_nodes"] += 1

#         # ── PROPOSITION nodes ─────────────────────────────────────────
#         prev_prop_id = None
#         prev_section_title = None

#         for prop in props:
#             p_id = self.db.create_node(["PROPOSITION"])
#             self.db.set_node_property(p_id, "text", prop.text)
#             self.db.set_node_property(p_id, "page_num", prop.page_num)
#             self.db.set_node_property(p_id, "content_hash", prop.content_hash)
#             self.db.set_significance(p_id, 64)

#             # SECTION → CONTAINS → PROPOSITION
#             sec_id = section_ids.get(prop.section_title)
#             if sec_id:
#                 self.db.create_rel(sec_id, p_id, "CONTAINS")
#                 stats["total_rels"] += 1

#             # PROPOSITION → NEXT → PROPOSITION (within same section)
#             if prev_prop_id is not None and prop.section_title == prev_section_title:
#                 self.db.create_rel(prev_prop_id, p_id, "NEXT")
#                 stats["total_rels"] += 1

#             prev_prop_id = p_id
#             prev_section_title = prop.section_title
#             stats["prop_ids"].append(p_id)
#             stats["total_nodes"] += 1

#         return stats


# # ── CLI ──────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     import tempfile

#     source = sys.argv[1] if len(sys.argv) > 1 else None
#     if not source:
#         print("Usage: python -m python.core.ingester <path_or_url>")
#         sys.exit(1)

#     # create a temp DB for testing
#     db_path = tempfile.mkdtemp(prefix="spider_test_")
#     print(f"Test DB: {db_path}\n")

#     db = PySpiderDB(db_path)
#     ingester = Ingester(db)
#     result = ingester.ingest(source)

#     print(f"\n{'='*60}")
#     print(f"Document node:  {result['doc_id']}")
#     print(f"Section nodes:  {len(result['section_ids'])}")
#     print(f"Proposition nodes: {len(result['prop_ids'])}")
#     print(f"Total nodes:    {result['total_nodes']}")
#     print(f"Total rels:     {result['total_rels']}")
