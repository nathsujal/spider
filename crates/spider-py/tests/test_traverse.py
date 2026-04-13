"""
Tests for graph traversal: get_neighbors, get_relationships, count_relationships.
"""

import pytest

import spider


class TestGetNeighbors:
    """Tests for Spider.get_neighbors()."""

    def _ingest_sample(self, tmp_db):
        """Ingest a sample document and return document/proposition IDs."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "Mumbai is in India",
                    entities=[
                        spider.Entity("Mumbai", "LOCATION"),
                        spider.Entity("India", "LOCATION"),
                    ],
                ),
            ],
        )
        result = tmp_db.index(request)

        # Find the proposition node
        prop_nodes = tmp_db.find_by_label("PROPOSITION")
        assert len(prop_nodes) >= 1
        prop_id = prop_nodes[0]

        return result.document_id, prop_id

    def test_get_neighbors_outgoing_from_document(self, tmp_db):
        """get_neighbors with OUTGOING from document returns propositions."""
        doc_id, prop_id = self._ingest_sample(tmp_db)

        neighbors = tmp_db.get_neighbors(doc_id, spider.Direction.OUTGOING)
        assert len(neighbors) >= 1
        assert all(isinstance(n, spider.Neighbor) for n in neighbors)
        assert all(isinstance(n.node_id, spider.NodeId) for n in neighbors)
        assert all(isinstance(n.edge_id, spider.EdgeId) for n in neighbors)

    def test_get_neighbors_incoming_to_proposition(self, tmp_db):
        """get_neighbors with INCOMING to proposition returns document."""
        doc_id, prop_id = self._ingest_sample(tmp_db)

        neighbors = tmp_db.get_neighbors(prop_id, spider.Direction.INCOMING)
        # Should have at least the document node
        assert len(neighbors) >= 1

    def test_get_neighbors_both_directions(self, tmp_db):
        """get_neighbors with BOTH returns all connected nodes."""
        doc_id, prop_id = self._ingest_sample(tmp_db)

        neighbors = tmp_db.get_neighbors(doc_id, spider.Direction.BOTH)
        assert len(neighbors) >= 1

    def test_get_neighbors_with_string_direction(self, tmp_db):
        """get_neighbors accepts string direction."""
        doc_id, prop_id = self._ingest_sample(tmp_db)

        neighbors = tmp_db.get_neighbors(doc_id, "outgoing")
        assert len(neighbors) >= 1

    def test_get_neighbors_case_insensitive(self, tmp_db):
        """get_neighbors direction string is case-insensitive."""
        doc_id, prop_id = self._ingest_sample(tmp_db)

        neighbors = tmp_db.get_neighbors(doc_id, "OUTGOING")
        assert len(neighbors) >= 1

    def test_get_neighbors_nonexistent_node(self, tmp_db):
        """get_neighbors raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_neighbors(fake_id, spider.Direction.BOTH)

    def test_get_neighbors_no_edges(self, tmp_db):
        """get_neighbors returns empty list for node with no edges."""
        # Create a document with no propositions
        request = spider.IngestRequest(
            title="Empty",
            propositions=[],
        )
        result = tmp_db.index(request)

        # Document should have no outgoing edges
        neighbors = tmp_db.get_neighbors(result.document_id, spider.Direction.OUTGOING)
        assert neighbors == []


class TestGetRelationships:
    """Tests for Spider.get_relationships()."""

    def _ingest_sample(self, tmp_db):
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "Mumbai is in India",
                    entities=[
                        spider.Entity("Mumbai", "LOCATION"),
                        spider.Entity("India", "LOCATION"),
                    ],
                ),
            ],
        )
        return tmp_db.index(request)

    def test_get_relationships_returns_dicts(self, tmp_db):
        """get_relationships returns list of dicts with source_id, target_id."""
        self._ingest_sample(tmp_db)

        doc_nodes = tmp_db.find_by_label("DOCUMENT")
        assert len(doc_nodes) >= 1
        doc_id = doc_nodes[0]

        rels = tmp_db.get_relationships(doc_id, spider.Direction.OUTGOING)
        assert isinstance(rels, list)
        if len(rels) > 0:
            rel = rels[0]
            assert isinstance(rel, dict)
            assert "source_id" in rel
            assert "target_id" in rel

    def test_get_relationships_nonexistent_node(self, tmp_db):
        """get_relationships raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_relationships(fake_id, spider.Direction.BOTH)


class TestCountRelationships:
    """Tests for Spider.count_relationships()."""

    def _ingest_sample(self, tmp_db):
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "Mumbai is in India",
                    entities=[
                        spider.Entity("Mumbai", "LOCATION"),
                        spider.Entity("India", "LOCATION"),
                    ],
                ),
            ],
        )
        return tmp_db.index(request)

    def test_count_matches_relationships_length(self, tmp_db):
        """count_relationships matches len(get_relationships())."""
        self._ingest_sample(tmp_db)

        doc_nodes = tmp_db.find_by_label("DOCUMENT")
        assert len(doc_nodes) >= 1
        doc_id = doc_nodes[0]

        count = tmp_db.count_relationships(doc_id, spider.Direction.OUTGOING)
        rels = tmp_db.get_relationships(doc_id, spider.Direction.OUTGOING)
        assert count == len(rels)

    def test_count_nonexistent_node(self, tmp_db):
        """count_relationships raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.count_relationships(fake_id, spider.Direction.BOTH)


class TestNeighborClass:
    """Tests for the Neighbor type."""

    def test_neighbor_properties(self, tmp_db):
        """Neighbor has node_id and edge_id properties."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("name", "TYPE")],
                ),
            ],
        )
        tmp_db.index(request)

        doc_nodes = tmp_db.find_by_label("DOCUMENT")
        if doc_nodes:
            neighbors = tmp_db.get_neighbors(doc_nodes[0], spider.Direction.OUTGOING)
            if neighbors:
                n = neighbors[0]
                assert isinstance(n.node_id, spider.NodeId)
                assert isinstance(n.edge_id, spider.EdgeId)

    def test_neighbor_repr(self, tmp_db):
        """Neighbor.__repr__ returns a string."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("name", "TYPE")],
                ),
            ],
        )
        tmp_db.index(request)

        doc_nodes = tmp_db.find_by_label("DOCUMENT")
        if doc_nodes:
            neighbors = tmp_db.get_neighbors(doc_nodes[0], spider.Direction.OUTGOING)
            if neighbors:
                assert isinstance(repr(neighbors[0]), str)
