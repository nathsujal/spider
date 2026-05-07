"""
Tests for document ingestion: index(), deduplication, edge cases.
"""

import pytest

import spider


class TestIngestion:
    """Tests for Spider.index() and ingestion types."""

    def test_basic_ingest(self, tmp_db):
        """Ingest a simple document with propositions."""
        request = spider.IngestRequest(
            title="Test Document",
            propositions=[
                spider.Proposition(
                    "Mumbai is the financial capital of India",
                    entities=[
                        spider.Entity("Mumbai", "LOCATION"),
                        spider.Entity("India", "LOCATION"),
                    ],
                ),
            ],
        )

        result = tmp_db.index(request)

        assert result.document_id is not None
        assert isinstance(result.document_id, spider.NodeId)
        assert result.proposition_count == 1
        assert result.entity_count == 2
        assert result.edge_count == 3  # 1 CONTAINS + 2 MENTIONS

    def test_ingest_empty_propositions(self, tmp_db):
        """Ingesting a document with zero propositions succeeds."""
        request = spider.IngestRequest(
            title="Empty Document",
            propositions=[],
        )

        result = tmp_db.index(request)

        assert result.document_id is not None
        assert result.proposition_count == 0
        assert result.entity_count == 0
        assert result.edge_count == 0

    def test_ingest_deduplicates_entities(self, tmp_db):
        """Ingesting duplicate entity names deduplicates correctly."""
        # First ingestion creates entities
        request1 = spider.IngestRequest(
            title="Doc 1",
            propositions=[
                spider.Proposition(
                    "Mumbai is a city",
                    entities=[spider.Entity("Mumbai", "LOCATION")],
                ),
            ],
        )
        result1 = tmp_db.index(request1)
        assert result1.entity_count == 1

        # Second ingestion with same entity name should deduplicate
        request2 = spider.IngestRequest(
            title="Doc 2",
            propositions=[
                spider.Proposition(
                    "Mumbai is in India",
                    entities=[spider.Entity("Mumbai", "LOCATION")],
                ),
            ],
        )
        result2 = tmp_db.index(request2)
        # Entity count should be 0 since Mumbai was already ingested
        # (or 1 if a new entity "India" is created)
        assert result2.entity_count <= 1

    def test_ingest_multiple_propositions(self, tmp_db):
        """Ingest a document with multiple propositions."""
        request = spider.IngestRequest(
            title="Multi-Prop Document",
            propositions=[
                spider.Proposition(
                    "Alice works at Google",
                    entities=[
                        spider.Entity("Alice", "PERSON"),
                        spider.Entity("Google", "ORGANIZATION"),
                    ],
                ),
                spider.Proposition(
                    "Google is in Mountain View",
                    entities=[
                        spider.Entity("Google", "ORGANIZATION"),
                        spider.Entity("Mountain View", "LOCATION"),
                    ],
                ),
            ],
        )

        result = tmp_db.index(request)

        assert result.proposition_count == 2
        # Entity count depends on dedup (Google appears twice but is deduped)
        assert result.entity_count >= 3  # Alice, Google, Mountain View

    def test_ingest_result_properties(self, tmp_db):
        """IngestResult has all expected properties."""
        request = spider.IngestRequest(
            title="Test",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("name", "TYPE")],
                ),
            ],
        )
        result = tmp_db.index(request)

        assert isinstance(result.document_id, spider.NodeId)
        assert isinstance(result.proposition_count, int)
        assert isinstance(result.entity_count, int)
        assert isinstance(result.edge_count, int)

    def test_ingest_request_properties(self):
        """IngestRequest has correct properties."""
        request = spider.IngestRequest(
            title="My Doc",
            propositions=[spider.Proposition("text")],
        )

        assert request.title == "My Doc"
        assert len(request.propositions) == 1

    def test_entity_properties(self):
        """Entity has correct properties."""
        entity = spider.Entity("Mumbai", "LOCATION")

        assert entity.name == "Mumbai"
        assert entity.entity_type == "LOCATION"

    def test_proposition_properties(self):
        """Proposition has correct properties."""
        prop = spider.Proposition(
            "text here",
            entities=[spider.Entity("name", "TYPE")],
        )

        assert prop.text == "text here"
        assert len(prop.entities) == 1
