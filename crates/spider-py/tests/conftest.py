"""
Pytest fixtures for spider-py integration tests.

Provides temporary database fixtures that are automatically cleaned up
after each test.
"""

import os
import tempfile
from typing import Generator

import pytest

# The spider module is built by maturin and importable as a native extension.
# Tests require `maturin develop` to be run first.
import spider


@pytest.fixture
def tmp_db() -> Generator[spider.Spider, None, None]:
    """Open a Spider database in a temporary directory, yield it, and close.

    The database is created in a unique temporary directory for each test.
    The database is closed when the test completes, even if the test fails.
    """
    with tempfile.TemporaryDirectory(prefix="spider_test_") as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = spider.Spider.open(db_path)
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
def tmp_db_default() -> Generator[spider.Spider, None, None]:
    """Open a Spider database at the platform-default location in a temp dir.

    Uses a temporary directory with an override of the default path.
    """
    with tempfile.TemporaryDirectory(prefix="spider_default_test_") as tmpdir:
        # Open at a subdirectory of temp dir to avoid conflicts
        db_path = os.path.join(tmpdir, "default")
        db = spider.Spider.open(db_path)
        try:
            yield db
        finally:
            db.close()
