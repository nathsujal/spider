"""
Tests for find queries: find_by_label, find_by_property, find_one_by_property.
"""

import pytest

import spider


class TestFindByLabel:
    """Tests for Spider.find_by_label()."""

    def test_find_by_label_after_ingest(self, tmp_db):
        """find_by_label returns nodes with the given label after ingestion."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("name", "LOCATION")],
                ),
            ],
        )
        tmp_db.index(request)

        # Should find DOCUMENT nodes
        docs = tmp_db.find_by_label("DOCUMENT")
        assert len(docs) >= 1
        assert all(isinstance(nid, spider.NodeId) for nid in docs)

    def test_find_by_label_nonexistent(self, tmp_db):
        """find_by_label returns empty list for non-existent label."""
        results = tmp_db.find_by_label("NONEXISTENT_LABEL")
        assert results == []

    def test_find_by_label_empty_database(self, tmp_db):
        """find_by_label returns empty list on empty database."""
        results = tmp_db.find_by_label("DOCUMENT")
        assert results == []


class TestFindByProperty:
    """Tests for Spider.find_by_property()."""

    def test_find_by_property_after_ingest(self, tmp_db):
        """find_by_property finds ingested entities."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("TestEntity", "LOCATION")],
                ),
            ],
        )
        tmp_db.index(request)

        # Find by entity name
        results = tmp_db.find_by_property("name", "TestEntity")
        assert len(results) >= 1
        assert all(isinstance(nid, spider.NodeId) for nid in results)

    def test_find_by_property_unknown_key(self, tmp_db):
        """find_by_property returns empty list for unknown key."""
        results = tmp_db.find_by_property("nonexistent_key", "value")
        assert results == []

    def test_find_by_property_empty_database(self, tmp_db):
        """find_by_property returns empty list on empty database."""
        results = tmp_db.find_by_property("name", "anything")
        assert results == []

    def test_find_by_property_multiple_matches(self, tmp_db):
        """find_by_property finds all matching nodes."""
        # Ingest two documents with entities having the same name
        request1 = spider.IngestRequest(
            title="Doc1",
            propositions=[
                spider.Proposition(
                    "text1",
                    entities=[spider.Entity("SharedName", "LOCATION")],
                ),
            ],
        )
        request2 = spider.IngestRequest(
            title="Doc2",
            propositions=[
                spider.Proposition(
                    "text2",
                    entities=[spider.Entity("SharedName", "LOCATION")],
                ),
            ],
        )
        tmp_db.index(request1)
        tmp_db.index(request2)

        # Entity should be deduplicated, but documents are separate
        results = tmp_db.find_by_property("name", "SharedName")
        # At least one result (may be deduplicated to one)
        assert len(results) >= 1


class TestFindOneByProperty:
    """Tests for Spider.find_one_by_property()."""

    def test_find_one_by_property_returns_first_match(self, tmp_db):
        """find_one_by_property returns first matching node."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("UniqueName", "PERSON")],
                ),
            ],
        )
        tmp_db.index(request)

        result = tmp_db.find_one_by_property("name", "UniqueName")
        assert result is not None
        assert isinstance(result, spider.NodeId)

    def test_find_one_by_property_no_match(self, tmp_db):
        """find_one_by_property returns None for no match."""
        result = tmp_db.find_one_by_property("name", "nonexistent")
        assert result is None

    def test_find_one_by_property_empty_database(self, tmp_db):
        """find_one_by_property returns None on empty database."""
        result = tmp_db.find_one_by_property("name", "anything")
        assert result is None
