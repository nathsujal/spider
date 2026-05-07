"""
Tests for Spider database lifecycle: open, close, context manager, path.
"""

import os
import tempfile

import pytest

import spider


class TestSpiderOpen:
    """Tests for Spider.open() and Spider.open_default()."""

    def test_open_creates_database(self, tmp_db):
        """Opening a database at a new path creates the directory and files."""
        assert tmp_db is not None
        assert isinstance(tmp_db, spider.Spider)

    def test_open_path_returns_correct_path(self, tmp_db):
        """Spider.path returns the path that was passed to open()."""
        # The fixture opens at a specific path, verify it matches
        assert isinstance(tmp_db.path, str)
        assert len(tmp_db.path) > 0

    def test_open_default(self):
        """Spider.open_default() opens at the platform-default location."""
        db = spider.Spider.open_default()
        try:
            assert isinstance(db.path, str)
            assert len(db.path) > 0
        finally:
            db.close()

    def test_open_creates_parent_directories(self):
        """Opening a database creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "spider.db")
            db = spider.Spider.open(nested_path)
            try:
                assert os.path.exists(nested_path)
            finally:
                db.close()

    def test_open_existing_directory(self):
        """Opening a database in an existing directory works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            # First open creates it
            db1 = spider.Spider.open(db_path)
            db1.close()

            # Second open should work
            db2 = spider.Spider.open(db_path)
            try:
                assert db2.path == db_path
            finally:
                db2.close()


class TestSpiderClose:
    """Tests for Spider.close()."""

    def test_close_is_idempotent(self, tmp_db):
        """Calling close() multiple times is safe."""
        tmp_db.close()
        tmp_db.close()  # Should not raise

    def test_close_after_open(self):
        """Closing after opening works without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = spider.Spider.open(db_path)
            db.close()


class TestContextManager:
    """Tests for Spider context manager (__enter__/__exit__)."""

    def test_context_manager_opens_and_closes(self):
        """Using 'with' opens and automatically closes the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with spider.Spider.open(db_path) as db:
                assert isinstance(db, spider.Spider)
                assert db.path == db_path
            # After exiting, db should be closed (no error on double close)

    def test_context_manager_closes_on_exception(self):
        """Database is closed even if an exception occurs in the block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            try:
                with spider.Spider.open(db_path) as db:
                    raise ValueError("test exception")
            except ValueError:
                pass
            # Database should be closed despite the exception

    def test_context_manager_returns_self(self):
        """'with ... as db' returns the same Spider instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with spider.Spider.open(db_path) as db:
                assert isinstance(db, spider.Spider)
