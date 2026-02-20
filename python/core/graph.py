import logging
from typing import Any

from spider import PySpiderDB
from python.models import Document, Proposition

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Persists document structure and extracted facts into the graph.

    Schema:
      DOCUMENT
       │
       ├─[CONTAINS]→ SECTION (with [NEXT] links for reading order)
       │              │
       │              └─[CONTAINS]→ PROPOSITION (with [NEXT] links)
       │
       └─ Properties: source, title, content_hash, etc.
    """

    def __init__(self, db: PySpiderDB):
        self.db = db

    def has_document(self, content_hash: str) -> bool:
        """Check if a document with this hash already exists."""
        nodes = self.db.find_nodes_by_property("content_hash", content_hash)
        return len(nodes) > 0

    def build(self, doc: Document, propositions: list[Proposition]) -> dict[str, Any]:
        """
        Build graph nodes/edges for a document and its propositions.

        Returns stats dict with counts and root node ID.
        """
        stats = {
            "doc_id": 0,
            "sections_created": 0,
            "props_created": 0,
            "total_nodes": 0,
            "total_rels": 0,
        }

        # 1. Idempotency check
        if self.has_document(doc.content_hash):
            logger.info(f"Document already exists (hash={doc.content_hash[:8]}). Skipping build.")
            # Find the existing ID to return
            nodes = self.db.find_nodes_by_property("content_hash", doc.content_hash)
            stats["doc_id"] = nodes[0]
            stats["status"] = "skipped_duplicate"
            return stats

        logger.info(f"Building graph for: {doc.metadata.source}")

        # 2. Create DOCUMENT node
        doc_id = self.db.create_node(["DOCUMENT"])
        self.db.set_node_property(doc_id, "source", doc.metadata.source)
        self.db.set_node_property(doc_id, "source_type", doc.metadata.source_type)
        self.db.set_node_property(doc_id, "content_hash", doc.content_hash)
        
        if doc.metadata.title:
            self.db.set_node_property(doc_id, "title", doc.metadata.title)
        if doc.metadata.author:
            self.db.set_node_property(doc_id, "author", doc.metadata.author)
        if doc.metadata.page_count:
            self.db.set_node_property(doc_id, "page_count", doc.metadata.page_count)
            
        # Store extra metadata (like URL, specific format info)
        for k, v in doc.metadata.extra.items():
            if isinstance(v, (str, int, float, bool)):
                self.db.set_node_property(doc_id, k, v)

        # High significance for root nodes
        self.db.set_significance(doc_id, 200)
        
        stats["doc_id"] = doc_id
        stats["total_nodes"] += 1

        # 3. Create SECTION nodes
        section_map: dict[str, int] = {}  # title -> node_id
        prev_sec_id = None

        for section in doc.sections:
            sec_id = self.db.create_node(["SECTION"])
            self.db.set_node_property(sec_id, "title", section.title)
            self.db.set_node_property(sec_id, "level", section.level)
            self.db.set_node_property(sec_id, "page_start", section.page_start)
            
            self.db.set_significance(sec_id, 150)  # Medium importance

            # Rel: DOC -> SECTION
            self.db.create_rel(doc_id, sec_id, "CONTAINS")
            stats["total_rels"] += 1

            # Rel: SECTION -> NEXT -> SECTION (Reading order)
            if prev_sec_id is not None:
                self.db.create_rel(prev_sec_id, sec_id, "NEXT")
                stats["total_rels"] += 1
            
            prev_sec_id = sec_id
            section_map[section.title] = sec_id
            stats["sections_created"] += 1
            stats["total_nodes"] += 1

        # 4. Create PROPOSITION nodes
        prev_prop_id = None
        prev_parent_title = None

        for prop in propositions:
            prop_id = self.db.create_node(["PROPOSITION"])
            self.db.set_node_property(prop_id, "text", prop.text)
            self.db.set_node_property(prop_id, "page_num", prop.page_num)
            
            if prop.content_hash:
                self.db.set_node_property(prop_id, "content_hash", prop.content_hash)

            self.db.set_significance(prop_id, 100)  # Granular info

            # Rel: SECTION -> PROPOSITION
            # Fallback to document if section not found (shouldn't happen with correct extraction)
            parent_id = section_map.get(prop.section_title)
            if parent_id:
                self.db.create_rel(parent_id, prop_id, "CONTAINS")
                stats["total_rels"] += 1
            else:
                self.db.create_rel(doc_id, prop_id, "CONTAINS")
                stats["total_rels"] += 1

            # Rel: PROP -> NEXT -> PROP (Reading order within same parent)
            # Only link sequential props if they belong to the same logical section
            if prev_prop_id is not None and prop.section_title == prev_parent_title:
                self.db.create_rel(prev_prop_id, prop_id, "NEXT")
                stats["total_rels"] += 1

            prev_prop_id = prop_id
            prev_parent_title = prop.section_title
            stats["props_created"] += 1
            stats["total_nodes"] += 1

        stats["status"] = "created"
        logger.info(f"Graph build complete. Nodes: {stats['total_nodes']}, Rels: {stats['total_rels']}")
        return stats
