// PySpider class - Python wrapper for spider_core::db::Spider

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyType;

use spider_core::db::lifecycle::Spider;
use spider_core::db::ingest;

use crate::error::db_error_to_pyerr;
use crate::ingest::{PyIngestRequest, PyIngestResult};

/// Python wrapper for the Spider database handle.
///
/// This class provides a Pythonic interface to the spider-core database engine
/// with automatic GIL release during I/O operations and context manager support.
///
/// Example usage:
/// ```python
/// import spider
///
/// # Open a database
/// db = spider.Spider.open("/tmp/my_db")
///
/// # Use as context manager (auto-closes on exit)
/// with spider.Spider.open("/tmp/my_db") as db:
///     # ... use db ...
///     pass
/// ```
#[pyclass]
pub struct PySpider {
    /// The inner Spider database handle, wrapped in Arc<Mutex<>> for thread-safe
    /// sharing across Python threads.
    inner: Arc<Mutex<Spider>>,
    /// Cached database path for the path() method.
    path: String,
}

#[pymethods]
impl PySpider {
    /// Open or create a Spider database at the given path.
    ///
    /// Args:
    ///     path: Filesystem path to the database directory.
    ///           The directory will be created if it doesn't exist.
    ///
    /// Returns:
    ///     Spider: A new Spider database handle.
    ///
    /// Raises:
    ///     SpiderIOError: If the database cannot be opened or created.
    ///     SpiderCorruptError: If the database metadata is corrupt.
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &str, py: Python<'_>) -> PyResult<Self> {
        let path_buf = Path::new(path);

        // Perform the open operation outside the GIL (I/O operation)
        let spider = py.allow_threads(|| {
            Spider::open(path_buf).map_err(db_error_to_pyerr)
        })?;

        let path_str = spider.path().to_string_lossy().to_string();
        let inner = Arc::new(Mutex::new(spider));

        Ok(PySpider {
            inner,
            path: path_str,
        })
    }

    /// Open or create a Spider database at the platform-default location.
    ///
    /// Default paths:
    ///     Linux: ~/.local/share/spider/default/
    ///     macOS: ~/Library/Application Support/spider/default/
    ///     Windows: %APPDATA%\spider\default\
    ///
    /// Returns:
    ///     Spider: A new Spider database handle.
    ///
    /// Raises:
    ///     SpiderIOError: If the database cannot be opened or created.
    ///     SpiderCorruptError: If the database metadata is corrupt.
    #[classmethod]
    fn open_default(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        // Perform the open operation outside the GIL (I/O operation)
        let spider = py.allow_threads(|| {
            Spider::open_default().map_err(db_error_to_pyerr)
        })?;

        let path_str = spider.path().to_string_lossy().to_string();
        let inner = Arc::new(Mutex::new(spider));

        Ok(PySpider {
            inner,
            path: path_str,
        })
    }

    /// Gracefully close the database, flushing all data to disk.
    ///
    /// This method is idempotent -- safe to call multiple times.
    /// Subsequent calls after the first are no-ops.
    ///
    /// The database is also automatically closed when the Python object
    /// is garbage collected (via Drop), but explicit closing is recommended
    /// to catch any errors.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner: Arc<Mutex<Spider>> = Arc::clone(&self.inner);

        py.allow_threads(move || {
            let mut db = inner.lock();
            db.close().map_err(db_error_to_pyerr)
        })
    }

    /// Return the database directory path.
    ///
    /// Returns:
    ///     str: The filesystem path to the database directory.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// Context manager entry: returns self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit: closes the database and suppresses errors.
    ///
    /// Silently ignores errors during __exit__ to match spider-core's
    /// Drop behavior (errors printed to stderr, not propagated).
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Call close, but suppress errors to match Drop behavior
        let _ = self.close(py);
        Ok(())
    }

    /// Return a debug representation of the Spider handle.
    ///
    /// Returns:
    ///     str: A string like "Spider('/path/to/db')".
    fn __repr__(&self) -> String {
        format!("Spider('{}')", self.path)
    }

    /// Ingest a document with propositions into the database.
    ///
    /// Creates a Document node with the given title, proposition nodes for each
    /// proposition, entity nodes (deduplicated by name), and wires CONTAINS +
    /// MENTIONS edges between them.
    ///
    /// Args:
    ///     request: An IngestRequest containing the document title and propositions.
    ///
    /// Returns:
    ///     IngestResult: Contains the created document ID and counts of nodes/edges.
    ///
    /// Raises:
    ///     SpiderIngestionError: If ingestion produces zero propositions.
    ///     SpiderNotFoundError: If a referenced node is not found.
    ///     SpiderIOError: If a file I/O error occurs.
    fn index(&self, py: Python<'_>, request: &PyIngestRequest) -> PyResult<PyIngestResult> {
        // Step 1: Clone Arc BEFORE GIL release (can't access self inside allow_threads)
        let inner = Arc::clone(&self.inner);

        // Step 2: Convert Python IngestRequest -> Rust IngestRequest<'_> BEFORE GIL release
        // This builds &str slices pointing into Python-owned String fields.
        // The borrow is valid for the duration of the method call.
        let rust_request = request.to_rust();

        // Step 3: Release GIL, lock mutex, call spider-core, convert result
        py.allow_threads(move || {
            let mut db = inner.lock();
            ingest::index(&mut db, &rust_request)
                .map(PyIngestResult::from)
                .map_err(db_error_to_pyerr)
        })
    }
}
