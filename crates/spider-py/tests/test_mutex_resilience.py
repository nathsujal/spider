"""
Tests for mutex poisoning resilience and concurrent access safety.

Verifies that:
- parking_lot::Mutex doesn't poison (unlike std::sync::Mutex)
- Spider can be used after error-returning operations
- Concurrent access from multiple threads doesn't deadlock
- close() is idempotent and safe to call multiple times
- Operations after errors still work correctly
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import spider


class TestMutexNoPoisoning:
    """Tests verifying parking_lot::Mutex doesn't poison."""

    def test_operations_after_error(self, tmp_db):
        """Spider can be used after an error-returning operation."""
        # Cause an error (non-existent node)
        fake_id = spider.NodeId(999999)
        with pytest.raises(spider.SpiderNotFoundError):
            tmp_db.get_bio_score(fake_id)

        # Spider should still be usable
        count = tmp_db.node_count()
        assert isinstance(count, int)

        # Should be able to ingest after error
        request = spider.IngestRequest(
            title="After Error",
            propositions=[
                spider.Proposition(
                    "text",
                    entities=[spider.Entity("name", "TYPE")],
                ),
            ],
        )
        result = tmp_db.index(request)
        assert result.document_id is not None

    def test_multiple_errors_same_instance(self, tmp_db):
        """Multiple errors on same instance don't corrupt state."""
        fake_id = spider.NodeId(999999)

        # Cause multiple errors
        for _ in range(10):
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.get_bio_score(fake_id)
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.get_bio_tier(fake_id)
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.node_touch(fake_id)

        # Spider should still be usable
        count = tmp_db.node_count()
        assert isinstance(count, int)

    def test_find_on_empty_database(self, tmp_db):
        """Find operations on empty database work without error."""
        # These should return empty results, not errors
        assert tmp_db.find_by_label("NONEXISTENT") == []
        assert tmp_db.find_by_property("key", "value") == []
        assert tmp_db.find_one_by_property("key", "value") is None


class TestConcurrentAccess:
    """Tests for concurrent access without deadlocks."""

    def _create_db_with_data(self):
        """Create a database with some data for concurrent testing."""
        tmpdir = tempfile.mkdtemp(prefix="spider_concurrent_test_")
        db_path = os.path.join(tmpdir, "test.db")
        db = spider.Spider.open(db_path)

        for i in range(10):
            request = spider.IngestRequest(
                title=f"Doc {i}",
                propositions=[
                    spider.Proposition(
                        f"Entity{i} is related to Entity{i+1}",
                        entities=[
                            spider.Entity(f"Entity{i}", "TYPE"),
                            spider.Entity(f"Entity{i+1}", "TYPE"),
                        ],
                    ),
                ],
            )
            db.index(request)

        return db, tmpdir

    def _cleanup(self, db, tmpdir):
        """Close database and remove temp directory."""
        try:
            db.close()
        except Exception:
            pass
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    def test_concurrent_reads_no_deadlock(self):
        """Multiple concurrent read operations don't deadlock."""
        db, tmpdir = self._create_db_with_data()
        try:
            errors = []
            results = []

            def do_read(task_id):
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        score = db.get_bio_score(docs[0])
                        tier = db.get_bio_tier(docs[0])
                        neighbors = db.get_neighbors(docs[0], spider.Direction.BOTH)
                        results.append((task_id, len(docs), score, tier, len(neighbors)))
                except Exception as e:
                    errors.append((task_id, e))

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(do_read, i) for i in range(20)]
                for f in as_completed(futures):
                    f.result(timeout=10)

            assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
            assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        finally:
            self._cleanup(db, tmpdir)

    def test_concurrent_mixed_operations_no_deadlock(self):
        """Multiple concurrent mixed operations (reads + writes) don't deadlock."""
        db, tmpdir = self._create_db_with_data()
        try:
            errors = []
            results = []

            def do_read(task_id):
                try:
                    docs = db.find_by_label("DOCUMENT")
                    results.append(("read", task_id, len(docs)))
                except Exception as e:
                    errors.append(("read", task_id, e))

            def do_bio(task_id):
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        score = db.get_bio_score(docs[0])
                        results.append(("bio", task_id, score))
                except Exception as e:
                    errors.append(("bio", task_id, e))

            def do_touch(task_id):
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        count = db.node_touch(docs[0])
                        results.append(("touch", task_id, count))
                except Exception as e:
                    errors.append(("touch", task_id, e))

            def do_neighbors(task_id):
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        neighbors = db.get_neighbors(docs[0], spider.Direction.BOTH)
                        results.append(("neighbors", task_id, len(neighbors)))
                except Exception as e:
                    errors.append(("neighbors", task_id, e))

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for i in range(10):
                    futures.append(executor.submit(do_read, i))
                    futures.append(executor.submit(do_bio, i))
                    futures.append(executor.submit(do_touch, i))
                    futures.append(executor.submit(do_neighbors, i))

                for f in as_completed(futures):
                    f.result(timeout=10)

            assert len(errors) == 0, f"Errors during concurrent mixed ops: {errors}"
            assert len(results) == 40, f"Expected 40 results, got {len(results)}"
        finally:
            self._cleanup(db, tmpdir)

    def test_concurrent_close_is_safe(self):
        """Multiple threads calling close() simultaneously is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)

            errors = []

            def close_db():
                try:
                    db.close()
                except Exception as e:
                    errors.append(e)

            # Multiple threads calling close() simultaneously
            threads = [threading.Thread(target=close_db) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            # Should have no errors (close is idempotent)
            assert len(errors) == 0, f"Errors during concurrent close: {errors}"


class TestCloseIdempotency:
    """Tests for close() idempotency and safety."""

    def test_double_close_no_error(self):
        """Calling close() twice doesn't raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)
            db.close()
            db.close()  # Should not raise

    def test_triple_close_no_error(self):
        """Calling close() three times doesn't raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)
            db.close()
            db.close()
            db.close()  # Should not raise

    def test_close_after_context_manager(self):
        """Calling close() after context manager exit is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with spider.Spider.open(db_path) as db:
                pass  # Auto-closed on exit
            db.close()  # Should not raise

    def test_operations_after_close(self, tmp_db):
        """Operations after close() behavior depends on implementation."""
        # Close the database
        tmp_db.close()

        # Note: Behavior after close is implementation-defined
        # This test verifies we don't crash/panic
        try:
            tmp_db.node_count()
        except Exception:
            pass  # Either success or error is acceptable


class TestCorruptDatabaseHandling:
    """Tests for handling corrupt databases."""

    def test_close_after_corrupt_db(self):
        """Opening a corrupt database, then closing it, works without panicking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupt database file
            db_path = os.path.join(tmpdir, "test.db")
            with open(db_path, "wb") as f:
                f.write(b"this is not a valid spider database file")

            try:
                db = spider.Spider.open(db_path)
                # If we got here, the database was opened despite corruption
                # Close should still work
                db.close()
            except (spider.SpiderCorruptError, spider.SpiderIOError):
                # Expected: opening a corrupt file should raise an error
                pass

    def test_error_doesnt_corrupt_mutex(self, tmp_db):
        """Error operations don't corrupt the mutex state."""
        # Cause various errors
        fake_id = spider.NodeId(999999)
        for _ in range(5):
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.get_bio_score(fake_id)
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.get_neighbors(fake_id, spider.Direction.BOTH)
            with pytest.raises(spider.SpiderNotFoundError):
                tmp_db.count_relationships(fake_id, spider.Direction.OUTGOING)

        # Mutex should still be functional
        count = tmp_db.node_count()
        assert isinstance(count, int)
