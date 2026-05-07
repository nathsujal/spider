// PySpider class - Python wrapper for spider_core::db::Spider

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyType;

use spider_core::db::lifecycle::Spider;
use spider_core::db::ingest;
use spider_core::db::find;
use spider_core::query::traverse;
use spider_core::bio::score;
use spider_core::bio::tier::BioTier;

use crate::error::db_error_to_pyerr;
use crate::ingest::{PyIngestRequest, PyIngestResult};
use crate::types::{PyBioTier, PyDirection, PyNodeId};

/// Parse a direction from either a PyDirection enum or a string.
///
/// Accepts:
/// - `PyDirection` (e.g. `spider.Direction.OUTGOING`)
/// - `str` (case-insensitive: "outgoing", "incoming", "both")
///
/// Raises ValueError if the string is not a valid direction.
fn parse_direction(value: &Bound<'_, PyAny>) -> PyResult<spider_core::db::rels::Direction> {
    use spider_core::db::rels::Direction as RustDirection;

    // Try PyDirection first
    if let Ok(py_dir) = value.extract::<PyDirection>() {
        return Ok(py_dir.inner());
    }

    // Try string
    if let Ok(s) = value.extract::<&str>() {
        return match s.to_lowercase().as_str() {
            "outgoing" => Ok(RustDirection::Outgoing),
            "incoming" => Ok(RustDirection::Incoming),
            "both" => Ok(RustDirection::Both),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid direction '{}'. Must be 'outgoing', 'incoming', or 'both', or a Direction enum value",
                s
            ))),
        };
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "direction must be a Direction enum or a string ('outgoing', 'incoming', 'both')",
    ))
}

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

    // ========================================================================
    // Find queries
    // ========================================================================

    /// Find all nodes with a given label.
    ///
    /// Performs a sequential scan over all nodes, checking if each live node
    /// has the given label. Returns an empty list if the label has never been
    /// used.
    ///
    /// Args:
    ///     label: The label string to search for (e.g. "DOCUMENT", "ENTITY").
    ///
    /// Returns:
    ///     list[NodeId]: A list of NodeId objects with the given label.
    ///
    /// Raises:
    ///     SpiderIOError: If a file I/O error occurs during the scan.
    ///     SpiderTraversalError: If a property chain exceeds the depth limit.
    fn find_by_label(&self, py: Python<'_>, label: &str) -> PyResult<Vec<PyNodeId>> {
        let inner = Arc::clone(&self.inner);

        py.allow_threads(move || {
            let mut db = inner.lock();
            find::find_by_label(&mut db, label)
                .map(|ids| ids.iter().map(|nid| PyNodeId::from(nid.get())).collect())
                .map_err(db_error_to_pyerr)
        })
    }

    /// Find all nodes with a property matching the given key and value.
    ///
    /// Performs a sequential scan over all nodes, checking each node's property
    /// chain for a matching key/value pair. Only matches inline short strings
    /// (<=6 bytes).
    ///
    /// Args:
    ///     key: The property key (e.g. "name").
    ///     value: The property value to match (e.g. "Mumbai").
    ///
    /// Returns:
    ///     list[NodeId]: A list of NodeId objects with the matching property.
    ///                   Returns an empty list if the key has never been used
    ///                   or no matches are found.
    ///
    /// Raises:
    ///     SpiderIOError: If a file I/O error occurs during the scan.
    ///     SpiderTraversalError: If a property chain exceeds the depth limit.
    fn find_by_property(&self, py: Python<'_>, key: &str, value: &str) -> PyResult<Vec<PyNodeId>> {
        let inner = Arc::clone(&self.inner);

        py.allow_threads(move || {
            let mut db = inner.lock();
            find::find_by_property(&mut db, key, value)
                .map(|ids| ids.iter().map(|nid| PyNodeId::from(nid.get())).collect())
                .map_err(db_error_to_pyerr)
        })
    }

    /// Find the first node with a property matching the given key and value.
    ///
    /// Like `find_by_property`, but short-circuits on the first match.
    /// Performs a sequential scan and returns immediately when a match is found.
    ///
    /// Args:
    ///     key: The property key (e.g. "name").
    ///     value: The property value to match (e.g. "Mumbai").
    ///
    /// Returns:
    ///     NodeId | None: The first matching NodeId, or None if no match found.
    ///
    /// Raises:
    ///     SpiderIOError: If a file I/O error occurs during the scan.
    ///     SpiderTraversalError: If a property chain exceeds the depth limit.
    fn find_one_by_property(
        &self,
        py: Python<'_>,
        key: &str,
        value: &str,
    ) -> PyResult<Option<PyNodeId>> {
        let inner = Arc::clone(&self.inner);

        py.allow_threads(move || {
            let mut db = inner.lock();
            find::find_one_by_property(&mut db, key, value)
                .map(|opt| opt.map(|nid| PyNodeId::from(nid.get())))
                .map_err(db_error_to_pyerr)
        })
    }

    // ========================================================================
    // Graph traversal
    // ========================================================================

    /// Get all neighbor nodes connected to the given node.
    ///
    /// Walks the edge chain from the specified node and returns all neighbors
    /// in the given direction.
    ///
    /// Args:
    ///     node_id: The NodeId to find neighbors for.
    ///     direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
    ///                Can also be a string: "outgoing", "incoming", "both".
    ///
    /// Returns:
    ///     list[Neighbor]: A list of Neighbor objects, each with node_id and edge_id.
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderTraversalError: If the edge chain exceeds the depth limit.
    #[pyo3(signature = (node_id, direction))]
    fn get_neighbors(
        &self,
        py: Python<'_>,
        node_id: &PyNodeId,
        direction: Bound<'_, PyAny>,
    ) -> PyResult<Vec<crate::types::PyNeighbor>> {
        let inner = Arc::clone(&self.inner);
        let direction = parse_direction(&direction)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            traverse::get_neighbors(&mut db, node_id.into(), direction)
                .map(|neighbors| {
                    neighbors
                        .into_iter()
                        .map(crate::types::PyNeighbor::from_rust)
                        .collect()
                })
                .map_err(db_error_to_pyerr)
        })
    }

    /// Get all relationships (edges) connected to the given node.
    ///
    /// Returns a list of dictionaries with edge details: source_id, target_id.
    ///
    /// Args:
    ///     node_id: The NodeId to find relationships for.
    ///     direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
    ///                Can also be a string: "outgoing", "incoming", "both".
    ///
    /// Returns:
    ///     list[dict]: A list of dicts with keys: source_id, target_id.
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderTraversalError: If the edge chain exceeds the depth limit.
    #[pyo3(signature = (node_id, direction))]
    fn get_relationships<'py>(
        &self,
        py: Python<'py>,
        node_id: &PyNodeId,
        direction: Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
        let inner = Arc::clone(&self.inner);
        let direction = parse_direction(&direction)?;

        // Step 1: Get edge data (raw u32 values) OUTSIDE Python object creation
        let edges: Vec<(u32, u32)> = py.allow_threads(move || {
            let mut db = inner.lock();
            traverse::get_relationships(&mut db, node_id.into(), direction)
                .map(|edges| edges.iter().map(|e| (e.source_id, e.target_id)).collect())
                .map_err(db_error_to_pyerr)
        })?;

        // Step 2: Convert raw values to Python dicts AFTER GIL release
        let dicts: Vec<Bound<'py, pyo3::types::PyDict>> = edges
            .iter()
            .map(|(source_id, target_id)| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("source_id", *source_id).unwrap();
                dict.set_item("target_id", *target_id).unwrap();
                dict
            })
            .collect();

        Ok(dicts)
    }

    /// Count the number of relationships connected to the given node.
    ///
    /// More efficient than `get_relationships` when you only need the count,
    /// as it doesn't allocate a Vec of edge records.
    ///
    /// Args:
    ///     node_id: The NodeId to count relationships for.
    ///     direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
    ///                Can also be a string: "outgoing", "incoming", "both".
    ///
    /// Returns:
    ///     int: The number of relationships.
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderTraversalError: If the edge chain exceeds the depth limit.
    #[pyo3(signature = (node_id, direction))]
    fn count_relationships(
        &self,
        py: Python<'_>,
        node_id: &PyNodeId,
        direction: Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let inner = Arc::clone(&self.inner);
        let direction = parse_direction(&direction)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            traverse::count_relationships(&mut db, node_id.into(), direction)
                .map_err(db_error_to_pyerr)
        })
    }

    // ========================================================================
    // Bio scoring
    // ========================================================================

    /// Calculate the bio-inspired vitality score for a node.
    ///
    /// The bio score reflects a node's "memory strength" based on:
    /// - Significance: Higher significance increases the score.
    /// - Access frequency: More accesses increase the score (logarithmic).
    /// - Recency: Older nodes have decaying scores (gravitational decay).
    ///
    /// Args:
    ///     node_id: The NodeId to calculate the score for.
    ///
    /// Returns:
    ///     float: The bio vitality score (positive number for live nodes).
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderIOError: If a file I/O error occurs.
    fn get_bio_score(&self, py: Python<'_>, node_id: &PyNodeId) -> PyResult<f64> {
        let inner = Arc::clone(&self.inner);
        let nid = spider_core::db::nodes::NodeId::new(node_id.inner())
            .map_err(db_error_to_pyerr)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            // Get node: RecordFile uses 0-based index, NodeId is 1-based
            let node = db.nodes.get(nid.get() - 1).map_err(db_error_to_pyerr)?;
            Ok(score::calculate(&node))
        })
    }

    /// Get the bio storage tier for a node.
    ///
    /// Tiers classify nodes by their vitality score:
    ///     HOT: score > 20.0 (in RAM, instant access)
    ///     WARM: score > 5.0 (on SSD, fast I/O)
    ///     COLD: score > 0.0 (archived, slow access)
    ///     PRUNED: score <= 0.0 (eligible for deletion)
    ///
    /// Args:
    ///     node_id: The NodeId to classify.
    ///
    /// Returns:
    ///     BioTier: The storage tier (Hot, Warm, Cold, or Pruned).
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderIOError: If a file I/O error occurs.
    fn get_bio_tier(&self, py: Python<'_>, node_id: &PyNodeId) -> PyResult<PyBioTier> {
        let inner = Arc::clone(&self.inner);
        let nid = spider_core::db::nodes::NodeId::new(node_id.inner())
            .map_err(db_error_to_pyerr)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            // Get node: RecordFile uses 0-based index, NodeId is 1-based
            let node = db.nodes.get(nid.get() - 1).map_err(db_error_to_pyerr)?;
            let score_val = score::calculate(&node);
            Ok(PyBioTier::from(BioTier::from_score(score_val)))
        })
    }

    // ========================================================================
    // Node operations
    // ========================================================================

    /// Get the total number of live nodes in the database.
    ///
    /// Returns `metadata.next_node_id - 1`, which is the count of all node
    /// slots ever created (including deleted ones). For the exact count of
    /// live nodes, a full scan would be needed — this returns the upper bound.
    ///
    /// Returns:
    ///     int: The number of node slots (live + deleted).
    fn node_count(&self, py: Python<'_>) -> PyResult<u32> {
        let inner = Arc::clone(&self.inner);

        py.allow_threads(move || {
            let db = inner.lock();
            // next_node_id starts at 1, so count is next_node_id - 1
            Ok(db.metadata.next_node_id - 1)
        })
    }

    /// Touch a node, incrementing its access count and updating its last
    /// accessed timestamp.
    ///
    /// This increases the node's bio vitality score by refreshing its
    /// `last_accessed_at` to the current time and incrementing `access_count`.
    ///
    /// Args:
    ///     node_id: The NodeId to touch.
    ///
    /// Returns:
    ///     int: The new access count after incrementing.
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderIOError: If a file I/O error occurs.
    fn node_touch(&self, py: Python<'_>, node_id: &PyNodeId) -> PyResult<u32> {
        let inner = Arc::clone(&self.inner);
        let nid = spider_core::db::nodes::NodeId::new(node_id.inner())
            .map_err(db_error_to_pyerr)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            let idx = nid.get() - 1;
            let mut node = db.nodes.get(idx).map_err(db_error_to_pyerr)?;

            // Check if node is deleted (tombstone)
            if node.id == 0 {
                return Err(db_error_to_pyerr(spider_core::error::DbError::NodeNotFound(nid.get())));
            }

            // Increment access_count (saturating) and update timestamp
            node.access_count = node.access_count.saturating_add(1);
            node.last_accessed_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);

            db.nodes.set(idx, &node).map_err(db_error_to_pyerr)?;
            Ok(node.access_count)
        })
    }

    /// Set the significance value for a node.
    ///
    /// Significance affects the bio vitality score — higher significance
    /// means a higher score. Valid range is 0-255.
    ///
    /// Args:
    ///     node_id: The NodeId to update.
    ///     significance: The new significance value (0-255).
    ///
    /// Raises:
    ///     SpiderNotFoundError: If the node does not exist.
    ///     SpiderIOError: If a file I/O error occurs.
    fn set_significance(
        &self,
        py: Python<'_>,
        node_id: &PyNodeId,
        significance: u8,
    ) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        let nid = spider_core::db::nodes::NodeId::new(node_id.inner())
            .map_err(db_error_to_pyerr)?;

        py.allow_threads(move || {
            let mut db = inner.lock();
            let idx = nid.get() - 1;
            let mut node = db.nodes.get(idx).map_err(db_error_to_pyerr)?;

            // Check if node is deleted (tombstone)
            if node.id == 0 {
                return Err(db_error_to_pyerr(spider_core::error::DbError::NodeNotFound(nid.get())));
            }

            node.significance = significance;
            db.nodes.set(idx, &node).map_err(db_error_to_pyerr)
        })
    }
}
