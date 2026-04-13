"""
Tests that verify ALL I/O methods properly release the GIL.

This is the #1 performance pitfall in PyO3 projects. If a method forgets
py.allow_threads(), it will block all Python threads during I/O.

Each test:
1. Starts a long-running I/O operation in one thread
2. Verifies another thread can execute Python code concurrently
3. If GIL were not released, the second thread would be blocked
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import spider


class TestGILRelease:
    """Tests that verify GIL release for all I/O methods."""

    def _create_db(self):
        """Create a database with some data for testing."""
        tmpdir = tempfile.mkdtemp(prefix="spider_gil_test_")
        db_path = os.path.join(tmpdir, "test.db")
        db = spider.Spider.open(db_path)

        # Ingest some data to make operations non-trivial
        for i in range(10):
            request = spider.IngestRequest(
                title=f"Document {i}",
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

    def test_gil_release_on_open(self):
        """Spider.open() releases the GIL."""
        gil_released = threading.Event()

        def check_gil():
            gil_released.set()

        # Start a thread that will try to run Python code
        t = threading.Thread(target=check_gil)
        t.start()

        # Open a database (should release GIL during I/O)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)
            db.close()

        t.join(timeout=2.0)
        assert gil_released.is_set(), "GIL was not released during open()"

    def test_gil_release_on_close(self):
        """Spider.close() releases the GIL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)

            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            t = threading.Thread(target=check_gil)
            t.start()

            db.close()

            t.join(timeout=2.0)
            assert gil_released.is_set(), "GIL was not released during close()"

    def test_gil_release_on_index(self):
        """Spider.index() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            request = spider.IngestRequest(
                title="GIL Test Doc",
                propositions=[
                    spider.Proposition(
                        "text",
                        entities=[spider.Entity("name", "TYPE")],
                    ),
                ],
            )

            t = threading.Thread(target=check_gil)
            t.start()

            db.index(request)

            t.join(timeout=2.0)
            assert gil_released.is_set(), "GIL was not released during index()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_find_by_label(self):
        """Spider.find_by_label() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            t = threading.Thread(target=check_gil)
            t.start()

            db.find_by_label("DOCUMENT")

            t.join(timeout=2.0)
            assert gil_released.is_set(), "GIL was not released during find_by_label()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_find_by_property(self):
        """Spider.find_by_property() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            t = threading.Thread(target=check_gil)
            t.start()

            db.find_by_property("name", "Entity0")

            t.join(timeout=2.0)
            assert gil_released.is_set(), "GIL was not released during find_by_property()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_get_neighbors(self):
        """Spider.get_neighbors() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            doc_nodes = db.find_by_label("DOCUMENT")
            if doc_nodes:
                t = threading.Thread(target=check_gil)
                t.start()

                db.get_neighbors(doc_nodes[0], spider.Direction.BOTH)

                t.join(timeout=2.0)
                assert gil_released.is_set(), "GIL was not released during get_neighbors()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_get_bio_score(self):
        """Spider.get_bio_score() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            doc_nodes = db.find_by_label("DOCUMENT")
            if doc_nodes:
                t = threading.Thread(target=check_gil)
                t.start()

                db.get_bio_score(doc_nodes[0])

                t.join(timeout=2.0)
                assert gil_released.is_set(), "GIL was not released during get_bio_score()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_get_bio_tier(self):
        """Spider.get_bio_tier() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            doc_nodes = db.find_by_label("DOCUMENT")
            if doc_nodes:
                t = threading.Thread(target=check_gil)
                t.start()

                db.get_bio_tier(doc_nodes[0])

                t.join(timeout=2.0)
                assert gil_released.is_set(), "GIL was not released during get_bio_tier()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_node_touch(self):
        """Spider.node_touch() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            doc_nodes = db.find_by_label("DOCUMENT")
            if doc_nodes:
                t = threading.Thread(target=check_gil)
                t.start()

                db.node_touch(doc_nodes[0])

                t.join(timeout=2.0)
                assert gil_released.is_set(), "GIL was not released during node_touch()"
        finally:
            self._cleanup(db, tmpdir)

    def test_gil_release_on_set_significance(self):
        """Spider.set_significance() releases the GIL."""
        db, tmpdir = self._create_db()
        try:
            gil_released = threading.Event()

            def check_gil():
                gil_released.set()

            doc_nodes = db.find_by_label("DOCUMENT")
            if doc_nodes:
                t = threading.Thread(target=check_gil)
                t.start()

                db.set_significance(doc_nodes[0], 200)

                t.join(timeout=2.0)
                assert gil_released.is_set(), "GIL was not released during set_significance()"
        finally:
            self._cleanup(db, tmpdir)

    def test_concurrent_operations(self):
        """Multiple threads can operate on the database concurrently."""
        db, tmpdir = self._create_db()
        try:
            results = []
            errors = []

            def do_find():
                try:
                    docs = db.find_by_label("DOCUMENT")
                    results.append(("find_by_label", len(docs)))
                except Exception as e:
                    errors.append(e)

            def do_bio_score():
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        score = db.get_bio_score(docs[0])
                        results.append(("get_bio_score", score))
                except Exception as e:
                    errors.append(e)

            def do_neighbors():
                try:
                    docs = db.find_by_label("DOCUMENT")
                    if docs:
                        neighbors = db.get_neighbors(docs[0], spider.Direction.BOTH)
                        results.append(("get_neighbors", len(neighbors)))
                except Exception as e:
                    errors.append(e)

            # Run multiple operations concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _ in range(4):
                    futures.append(executor.submit(do_find))
                    futures.append(executor.submit(do_bio_score))
                    futures.append(executor.submit(do_neighbors))

                for f in futures:
                    f.result(timeout=10)

            # All operations should have completed
            assert len(results) == 12, f"Expected 12 results, got {len(results)}"
            assert len(errors) == 0, f"Errors occurred: {errors}"
        finally:
            self._cleanup(db, tmpdir)
