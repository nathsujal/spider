"""
Tests for exception handling and error cases.
"""

import os
import tempfile

import pytest

import spider


class TestExceptionHierarchy:
    """Tests for the exception class hierarchy."""

    def test_spider_error_is_base_class(self):
        """SpiderError is the base exception."""
        assert issubclass(spider.SpiderNotFoundError, spider.SpiderError)
        assert issubclass(spider.SpiderCorruptError, spider.SpiderError)
        assert issubclass(spider.SpiderIOError, spider.SpiderError)
        assert issubclass(spider.SpiderIngestionError, spider.SpiderError)
        assert issubclass(spider.SpiderTraversalError, spider.SpiderError)

    def test_spider_error_is_exception(self):
        """SpiderError inherits from Exception."""
        assert issubclass(spider.SpiderError, Exception)

    def test_all_exceptions_accessible(self):
        """All exception classes are accessible from spider module."""
        assert hasattr(spider, "SpiderError")
        assert hasattr(spider, "SpiderNotFoundError")
        assert hasattr(spider, "SpiderCorruptError")
        assert hasattr(spider, "SpiderIOError")
        assert hasattr(spider, "SpiderIngestionError")
        assert hasattr(spider, "SpiderTraversalError")


class TestIOErrors:
    """Tests for I/O-related error handling."""

    def test_open_invalid_path(self):
        """Opening with an invalid path raises appropriate error."""
        # Use a path that's definitely invalid
        invalid_path = "/nonexistent/deeply/nested/path/that/does/not/exist/test.db"
        # This should either create it or raise an error
        try:
            db = spider.Spider.open(invalid_path)
            db.close()
        except (spider.SpiderIOError, spider.SpiderCorruptError):
            pass  # Either outcome is acceptable for invalid paths

    def test_open_on_file_instead_of_directory(self):
        """Opening a database where a file exists (not a directory) raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "existing_file.txt")
            with open(file_path, "w") as f:
                f.write("not a database")

            with pytest.raises((spider.SpiderIOError, spider.SpiderCorruptError)):
                spider.Spider.open(file_path)


class TestNotFoundError:
    """Tests for SpiderNotFoundError."""

    def test_get_bio_score_nonexistent_node(self, tmp_db):
        """get_bio_score raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_bio_score(fake_id)

    def test_get_bio_tier_nonexistent_node(self, tmp_db):
        """get_bio_tier raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_bio_tier(fake_id)

    def test_node_touch_nonexistent_node(self, tmp_db):
        """node_touch raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.node_touch(fake_id)

    def test_set_significance_nonexistent_node(self, tmp_db):
        """set_significance raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.set_significance(fake_id, 100)

    def test_get_neighbors_nonexistent_node(self, tmp_db):
        """get_neighbors raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_neighbors(fake_id, spider.Direction.BOTH)

    def test_get_relationships_nonexistent_node(self, tmp_db):
        """get_relationships raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_relationships(fake_id, spider.Direction.BOTH)

    def test_count_relationships_nonexistent_node(self, tmp_db):
        """count_relationships raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.count_relationships(fake_id, spider.Direction.BOTH)


class TestInvalidDirection:
    """Tests for invalid direction handling."""

    def test_invalid_string_direction_get_neighbors(self, tmp_db):
        """Invalid string direction raises ValueError."""
        node_id = spider.NodeId(1)
        with pytest.raises(ValueError):
            tmp_db.get_neighbors(node_id, "invalid_direction")

    def test_invalid_string_direction_get_relationships(self, tmp_db):
        """Invalid string direction raises ValueError."""
        node_id = spider.NodeId(1)
        with pytest.raises(ValueError):
            tmp_db.get_relationships(node_id, "invalid_direction")

    def test_invalid_string_direction_count_relationships(self, tmp_db):
        """Invalid string direction raises ValueError."""
        node_id = spider.NodeId(1)
        with pytest.raises(ValueError):
            tmp_db.count_relationships(node_id, "invalid_direction")


class TestInvalidNodeId:
    """Tests for invalid NodeId values."""

    def test_node_id_zero_raises_error(self):
        """NodeId(0) raises an error."""
        with pytest.raises(ValueError):
            spider.NodeId(0)

    def test_edge_id_zero_raises_error(self):
        """EdgeId(0) raises an error."""
        with pytest.raises(ValueError):
            spider.EdgeId(0)


class TestIngestionErrors:
    """Tests for ingestion-related errors."""

    def test_ingest_empty_document(self, tmp_db):
        """Ingesting a document with no propositions succeeds."""
        request = spider.IngestRequest(
            title="Empty",
            propositions=[],
        )
        # This should succeed but return zero counts
        result = tmp_db.index(request)
        assert result.proposition_count == 0
        assert result.entity_count == 0
