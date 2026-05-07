"""
Tests for memory leak detection: Arc cycles, unclosed handles, object retention.

These tests verify that the Python bindings don't leak memory through:
- Arc cycles that Python's GC can't collect
- Unclosed database handles
- Python object retention
"""

import gc
import os
import tempfile
import tracemalloc

import pytest

import spider


class TestArcCycleCollection:
    """Tests for Arc cycle detection and cleanup."""

    def test_spider_gc_cleanup(self):
        """100 open/close cycles don't grow object count significantly."""
        # Get baseline object count
        gc.collect()
        before = len(gc.get_objects())

        for _ in range(100):
            with tempfile.TemporaryDirectory(prefix="spider_gc_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                db = spider.Spider.open(db_path)
                db.close()

        gc.collect()
        after = len(gc.get_objects())

        # Allow some growth for temporary objects, but not excessive
        growth = after - before
        assert growth < 50, f"Possible Arc leak: {growth} objects retained after 100 cycles"

    def test_spider_gc_cleanup_with_ingestion(self):
        """Open/close cycles with ingestion don't grow object count significantly."""
        gc.collect()
        before = len(gc.get_objects())

        for i in range(50):
            with tempfile.TemporaryDirectory(prefix="spider_gc_ingest_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                db = spider.Spider.open(db_path)
                request = spider.IngestRequest(
                    title=f"Doc {i}",
                    propositions=[
                        spider.Proposition(
                            f"text {i}",
                            entities=[spider.Entity(f"name{i}", "TYPE")],
                        ),
                    ],
                )
                db.index(request)
                db.close()

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 100, f"Possible Arc leak with ingestion: {growth} objects retained"


class TestContextManagerCleanup:
    """Tests for context manager cleanup and resource release."""

    def test_context_manager_no_leak(self):
        """Using 'with Spider.open(...) as db:' doesn't leak."""
        gc.collect()
        before = len(gc.get_objects())

        for _ in range(100):
            with tempfile.TemporaryDirectory(prefix="spider_ctx_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                with spider.Spider.open(db_path) as db:
                    pass  # Just open and close via context manager

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 50, f"Context manager leak: {growth} objects retained"

    def test_context_manager_with_ingestion(self):
        """Context manager with ingestion doesn't leak."""
        gc.collect()
        before = len(gc.get_objects())

        for i in range(50):
            with tempfile.TemporaryDirectory(prefix="spider_ctx_ingest_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                with spider.Spider.open(db_path) as db:
                    request = spider.IngestRequest(
                        title=f"Doc {i}",
                        propositions=[
                            spider.Proposition(
                                f"text {i}",
                                entities=[spider.Entity(f"name{i}", "TYPE")],
                            ),
                        ],
                    )
                    db.index(request)

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 100, f"Context manager ingestion leak: {growth} objects retained"


class TestUnclosedSpiderCleanup:
    """Tests for unclosed Spider handle cleanup via Drop."""

    def test_unclosed_spider_no_leak(self):
        """Forgetting to call close() doesn't leak (Drop handles it)."""
        gc.collect()
        before = len(gc.get_objects())

        for _ in range(100):
            with tempfile.TemporaryDirectory(prefix="spider_unclosed_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                db = spider.Spider.open(db_path)
                # Deliberately NOT calling db.close()
                del db
                gc.collect()

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 50, f"Unclosed Spider leak: {growth} objects retained"

    def test_unclosed_spider_with_exception(self):
        """Spider is cleaned up even if an exception occurs."""
        gc.collect()
        before = len(gc.get_objects())

        for _ in range(50):
            with tempfile.TemporaryDirectory(prefix="spider_exc_test_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                try:
                    db = spider.Spider.open(db_path)
                    raise ValueError("test exception")
                except ValueError:
                    pass
                gc.collect()

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 50, f"Exception cleanup leak: {growth} objects retained"


class TestTracemallocTracking:
    """Tests using tracemalloc for Python-side memory tracking."""

    def test_large_ingestion_memory_cleanup(self):
        """Large ingestion (1000+ propositions) doesn't grow memory disproportionately."""
        tracemalloc.start()

        with tempfile.TemporaryDirectory(prefix="spider_tracemalloc_test_") as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)

            # Get initial memory snapshot
            gc.collect()
            snapshot1 = tracemalloc.take_snapshot()

            # Ingest a large document
            propositions = []
            for i in range(1000):
                propositions.append(
                    spider.Proposition(
                        f"Proposition text {i}",
                        entities=[
                            spider.Entity(f"Entity{i}", "TYPE"),
                        ],
                    )
                )

            request = spider.IngestRequest(
                title="Large Document",
                propositions=propositions,
            )
            db.index(request)

            gc.collect()
            snapshot2 = tracemalloc.take_snapshot()

            db.close()

        # Check that memory growth is reasonable
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Allow up to 10MB growth for 1000 propositions
        assert total_growth < 10 * 1024 * 1024, f"Memory growth too large: {total_growth / 1024 / 1024:.2f}MB"

        tracemalloc.stop()

    def test_repeated_open_close_memory(self):
        """Repeated open/close doesn't grow memory over time."""
        tracemalloc.start()

        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(200):
            with tempfile.TemporaryDirectory(prefix="spider_tracemalloc_repeat_") as tmpdir:
                db_path = os.path.join(tmpdir, "test.db")
                db = spider.Spider.open(db_path)
                db.close()

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        # Allow up to 5MB growth for 200 cycles
        assert total_growth < 5 * 1024 * 1024, f"Repeated open/close memory growth: {total_growth / 1024 / 1024:.2f}MB"

        tracemalloc.stop()


class TestNodeValueTypeCleanup:
    """Tests for NodeId, EdgeId, and other value type cleanup."""

    def test_nodeid_creation_cleanup(self):
        """Creating many NodeId objects doesn't leak."""
        gc.collect()
        before = len(gc.get_objects())

        for i in range(10000):
            nid = spider.NodeId((i % 100000) + 1)
            del nid

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 100, f"NodeId leak: {growth} objects retained"

    def test_edgeid_creation_cleanup(self):
        """Creating many EdgeId objects doesn't leak."""
        gc.collect()
        before = len(gc.get_objects())

        for i in range(10000):
            eid = spider.EdgeId((i % 100000) + 1)
            del eid

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 100, f"EdgeId leak: {growth} objects retained"

    def test_ingestion_type_cleanup(self):
        """Creating many ingestion types doesn't leak."""
        gc.collect()
        before = len(gc.get_objects())

        for i in range(5000):
            entity = spider.Entity(f"name{i}", "TYPE")
            prop = spider.Proposition(f"text {i}", [entity])
            request = spider.IngestRequest(f"title {i}", [prop])
            del entity, prop, request

        gc.collect()
        after = len(gc.get_objects())

        growth = after - before
        assert growth < 200, f"Ingestion type leak: {growth} objects retained"
