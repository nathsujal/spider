"""
Tests for bio scoring: get_bio_score, get_bio_tier.
"""

import time

import pytest

import spider


class TestBioScore:
    """Tests for Spider.get_bio_score()."""

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

    def test_bio_score_returns_float(self, tmp_db):
        """get_bio_score returns a positive float for a live node."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        score = tmp_db.get_bio_score(node_id)

        assert isinstance(score, float)
        assert score > 0

    def test_bio_score_nonexistent_node(self, tmp_db):
        """get_bio_score raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_bio_score(fake_id)

    def test_bio_score_increases_with_significance(self, tmp_db):
        """Higher significance increases bio score."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        score_low = tmp_db.get_bio_score(node_id)
        tmp_db.set_significance(node_id, 200)
        score_high = tmp_db.get_bio_score(node_id)

        assert score_high > score_low

    def test_bio_score_increases_after_touch(self, tmp_db):
        """Touching a node increases its bio score."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        score_before = tmp_db.get_bio_score(node_id)
        tmp_db.node_touch(node_id)
        score_after = tmp_db.get_bio_score(node_id)

        # Touching updates last_accessed_at to now, which should increase score
        assert score_after >= score_before


class TestBioTier:
    """Tests for Spider.get_bio_tier() and BioTier enum."""

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

    def test_bio_tier_returns_bio_tier(self, tmp_db):
        """get_bio_tier returns a BioTier enum value."""
        node_id = self._ingest_and_get_node_id(tmp_db)

        tier = tmp_db.get_bio_tier(node_id)

        assert isinstance(tier, spider.BioTier)

    def test_bio_tier_nonexistent_node(self, tmp_db):
        """get_bio_tier raises SpiderNotFoundError for non-existent node."""
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_bio_tier(fake_id)

    def test_bio_tier_classification(self):
        """BioTier.from_score classifies scores correctly."""
        # High score should be Hot
        hot = spider.BioTier.from_score(100.0)
        assert hot == spider.BioTier.HOT
        assert hot.is_active()
        assert not hot.is_prunable()

        # Medium score should be Warm
        warm = spider.BioTier.from_score(10.0)
        assert warm == spider.BioTier.WARM
        assert warm.is_active()
        assert not warm.is_prunable()

        # Low positive score should be Cold
        cold = spider.BioTier.from_score(1.0)
        assert cold == spider.BioTier.COLD
        assert not cold.is_active()
        assert not cold.is_prunable()

        # Zero or negative should be Pruned
        pruned = spider.BioTier.from_score(0.0)
        assert pruned == spider.BioTier.PRUNED
        assert not pruned.is_active()
        assert pruned.is_prunable()

        pruned_neg = spider.BioTier.from_score(-5.0)
        assert pruned_neg == spider.BioTier.PRUNED
        assert pruned_neg.is_prunable()

    def test_bio_tier_string_representation(self):
        """BioTier has meaningful string representation."""
        assert str(spider.BioTier.HOT) == "Hot"
        assert str(spider.BioTier.WARM) == "Warm"
        assert str(spider.BioTier.COLD) == "Cold"
        assert str(spider.BioTier.PRUNED) == "Pruned"

    def test_bio_tier_ordering(self):
        """BioTier supports ordering: Pruned < Cold < Warm < Hot."""
        assert spider.BioTier.PRUNED < spider.BioTier.COLD
        assert spider.BioTier.COLD < spider.BioTier.WARM
        assert spider.BioTier.WARM < spider.BioTier.HOT

    def test_bio_tier_equality(self):
        """BioTier equality works correctly."""
        assert spider.BioTier.HOT == spider.BioTier.HOT
        assert spider.BioTier.HOT != spider.BioTier.COLD

    def test_bio_tier_class_attributes(self):
        """BioTier class attributes are accessible."""
        assert isinstance(spider.BioTier.HOT, spider.BioTier)
        assert isinstance(spider.BioTier.WARM, spider.BioTier)
        assert isinstance(spider.BioTier.COLD, spider.BioTier)
        assert isinstance(spider.BioTier.PRUNED, spider.BioTier)
