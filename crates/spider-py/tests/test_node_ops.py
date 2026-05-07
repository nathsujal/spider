"""
Tests for node operations: node_count, node_touch, set_significance.
"""

import pytest

import spider


class TestNodeCount:
    """Tests for Spider.node_count()."""

    def test_node_count_empty_database(self, tmp_db):
        """node_count returns 0 for an empty database."""
        count = tmp_db.node_count()
        assert count == 0

    def test_node_count_after_ingest(self, tmp_db):
        """node_count reflects nodes created by ingestion."""
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

        count = tmp_db.node_count()
        # Should have at least: 1 document + 1 proposition + 1 entity
        assert count >= 3

    def test_node_count_increases_with_ingestion(self, tmp_db):
        """node_count increases after each ingestion."""
        count_before = tmp_db.node_count()

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

        count_after = tmp_db.node_count()
        assert count_after > count_before


class TestNodeTouch:
    """Tests for Spider.node_touch()."""

    def _ingest_and_get_node_id(self, tmp_db):
        """Ingest a document and return its node ID."""
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
        return result.document_id

    def test_touch_increments_access_count(self, tmp_db):
        """node_touch increments the access count."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        count_before = tmp_db.node_touch(node_id)
        count_after = tmp_db.node_touch(node_id)

        assert count_after == count_before + 1

    def test_touch_nonexistent_node(self, tmp_db):
        """node_touch raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.node_touch(fake_id)

    def test_touch_returns_new_count(self, tmp_db):
        """node_touch returns the new access count."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        # First touch
        count1 = tmp_db.node_touch(node_id)
        assert isinstance(count1, int)
        assert count1 >= 1


class TestSetSignificance:
    """Tests for Spider.set_significance()."""

    def _ingest_and_get_node_id(self, tmp_db):
        """Ingest a document and return its node ID."""
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
        return result.document_id

    def test_set_significance_valid_range(self, tmp_db):
        """set_significance accepts values 0-255."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        # Should not raise
        tmp_db.set_significance(node_id, 0)
        tmp_db.set_significance(node_id, 128)
        tmp_db.set_significance(node_id, 255)

    def test_set_significance_nonexistent_node(self, tmp_db):
        """set_significance raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.set_significance(fake_id, 100)

    def test_set_significance_affects_bio_score(self, tmp_db):
        """Higher significance increases bio score."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        tmp_db.set_significance(node_id, 10)
        score_low = tmp_db.get_bio_score(node_id)

        tmp_db.set_significance(node_id, 200)
        score_high = tmp_db.get_bio_score(node_id)

        assert score_high > score_low
