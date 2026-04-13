// Python type wrappers: NodeId, EdgeId, Direction, BioTier

use pyo3::prelude::*;

use spider_core::db::nodes::NodeId;
use spider_core::db::rels::{Direction, EdgeId};
use spider_core::query::traverse;

// ============================================================================
// PyNodeId
// ============================================================================

/// A unique identifier for a node in the Spider graph database.
///
/// NodeIds are positive integers (1-based). The value 0 is reserved
/// as a sentinel and is not a valid NodeId.
///
/// Example:
/// ```python
/// import spider
///
/// node_id = spider.NodeId(42)
/// print(int(node_id))  # 42
/// print(repr(node_id))  # NodeId(42)
/// ```
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyNodeId {
    inner: u32,
}

#[pymethods]
impl PyNodeId {
    /// Create a new NodeId.
    ///
    /// Args:
    ///     raw: A positive integer (>= 1). Zero is not valid.
    ///
    /// Returns:
    ///     NodeId: A new NodeId instance.
    ///
    /// Raises:
    ///     SpiderError: If raw is 0 (invalid ID).
    #[new]
    fn new(raw: u32) -> PyResult<Self> {
        if raw == 0 {
            Err(pyo3::exceptions::PyValueError::new_err(
                "NodeId(0) is not valid — ID must be >= 1",
            ))
        } else {
            Ok(PyNodeId { inner: raw })
        }
    }

    /// Convert to a Python integer.
    fn __int__(&self) -> u32 {
        self.inner
    }

    /// Return a debug representation like "NodeId(42)".
    fn __repr__(&self) -> String {
        format!("NodeId({})", self.inner)
    }

    /// Compare equality with another NodeId.
    fn __eq__(&self, other: &PyNodeId) -> bool {
        self.inner == other.inner
    }

    /// Return a hash value for use as a dictionary key.
    fn __hash__(&self) -> u64 {
        self.inner as u64
    }

    /// Return the underlying u32 value.
    ///
    /// Returns:
    ///     int: The raw integer ID.
    fn as_int(&self) -> u32 {
        self.inner
    }
}

// Conversion: &PyNodeId -> NodeId
impl From<&PyNodeId> for NodeId {
    fn from(py_id: &PyNodeId) -> Self {
        // Safe: PyNodeId construction already validated that inner != 0
        NodeId::new(py_id.inner).expect("PyNodeId inner should never be 0")
    }
}

impl PyNodeId {
    /// Create a PyNodeId from a raw u32 value (internal use only).
    ///
    /// # Safety
    /// The caller must ensure `raw` is non-zero.
    pub(crate) fn from(raw: u32) -> Self {
        debug_assert!(raw != 0, "PyNodeId::from called with 0");
        PyNodeId { inner: raw }
    }

    /// Return the underlying u32 value (for internal use).
    pub(crate) fn inner(&self) -> u32 {
        self.inner
    }
}

// ============================================================================
// PyEdgeId
// ============================================================================

/// A unique identifier for an edge in the Spider graph database.
///
/// EdgeIds are positive integers (1-based). The value 0 is reserved
/// as a sentinel and is not a valid EdgeId.
///
/// Example:
/// ```python
/// import spider
///
/// edge_id = spider.EdgeId(7)
/// print(int(edge_id))  # 7
/// print(repr(edge_id))  # EdgeId(7)
/// ```
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyEdgeId {
    inner: u32,
}

#[pymethods]
impl PyEdgeId {
    /// Create a new EdgeId.
    ///
    /// Args:
    ///     raw: A positive integer (>= 1). Zero is not valid.
    ///
    /// Returns:
    ///     EdgeId: A new EdgeId instance.
    ///
    /// Raises:
    ///     SpiderError: If raw is 0 (invalid ID).
    #[new]
    fn new(raw: u32) -> PyResult<Self> {
        if raw == 0 {
            Err(pyo3::exceptions::PyValueError::new_err(
                "EdgeId(0) is not valid — ID must be >= 1",
            ))
        } else {
            Ok(PyEdgeId { inner: raw })
        }
    }

    /// Convert to a Python integer.
    fn __int__(&self) -> u32 {
        self.inner
    }

    /// Return a debug representation like "EdgeId(7)".
    fn __repr__(&self) -> String {
        format!("EdgeId({})", self.inner)
    }

    /// Compare equality with another EdgeId.
    fn __eq__(&self, other: &PyEdgeId) -> bool {
        self.inner == other.inner
    }

    /// Return a hash value for use as a dictionary key.
    fn __hash__(&self) -> u64 {
        self.inner as u64
    }

    /// Return the underlying u32 value.
    ///
    /// Returns:
    ///     int: The raw integer ID.
    fn as_int(&self) -> u32 {
        self.inner
    }
}

// Conversion: &PyEdgeId -> EdgeId
impl From<&PyEdgeId> for EdgeId {
    fn from(py_id: &PyEdgeId) -> Self {
        // Safe: PyEdgeId construction already validated that inner != 0
        EdgeId::new(py_id.inner).expect("PyEdgeId inner should never be 0")
    }
}

impl PyEdgeId {
    /// Create a PyEdgeId from a raw u32 value (internal use only).
    pub(crate) fn from(raw: u32) -> Self {
        debug_assert!(raw != 0, "PyEdgeId::from called with 0");
        PyEdgeId { inner: raw }
    }

    /// Return the underlying u32 value (for internal use).
    pub(crate) fn inner(&self) -> u32 {
        self.inner
    }
}

// ============================================================================
// PyDirection
// ============================================================================

/// Traversal direction for edge queries in the Spider graph database.
///
/// Variants:
///     OUTGOING: Edges where the queried node is the source.
///     INCOMING: Edges where the queried node is the target.
///     BOTH: All edges connected to the queried node.
///
/// Methods also accept string arguments: "outgoing", "incoming", "both"
/// (case-insensitive).
///
/// Example:
/// ```python
/// import spider
///
/// # Using enum values
/// neighbors = db.get_neighbors(node_id, spider.Direction.OUTGOING)
///
/// # Using strings (case-insensitive)
/// neighbors = db.get_neighbors(node_id, "outgoing")
/// ```
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyDirection {
    inner: Direction,
}

// Re-export for internal use
// (Direction is already imported at the top of the file)

#[pymethods]
impl PyDirection {
    /// Create a Direction from a string.
    ///
    /// Args:
    ///     value: One of "outgoing", "incoming", "both" (case-insensitive).
    ///
    /// Returns:
    ///     Direction: The corresponding Direction enum value.
    ///
    /// Raises:
    ///     ValueError: If the string is not a valid direction.
    #[staticmethod]
    fn from_str(value: &str) -> PyResult<Self> {
        match value.to_lowercase().as_str() {
            "outgoing" => Ok(PyDirection { inner: Direction::Outgoing }),
            "incoming" => Ok(PyDirection { inner: Direction::Incoming }),
            "both" => Ok(PyDirection { inner: Direction::Both }),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid direction '{}'. Must be 'outgoing', 'incoming', or 'both'", value),
            )),
        }
    }

    /// Convert to a human-readable string.
    ///
    /// Returns:
    ///     str: "Outgoing", "Incoming", or "Both".
    fn __str__(&self) -> &str {
        match self.inner {
            Direction::Outgoing => "Outgoing",
            Direction::Incoming => "Incoming",
            Direction::Both => "Both",
        }
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!("Direction.{}", self.__str__())
    }

    /// Compare equality with another Direction.
    fn __eq__(&self, other: &PyDirection) -> bool {
        self.inner == other.inner
    }

    /// Return a hash value.
    fn __hash__(&self) -> u64 {
        match self.inner {
            Direction::Outgoing => 1,
            Direction::Incoming => 2,
            Direction::Both => 3,
        }
    }

    /// Class attribute: OUTGOING
    #[classattr]
    const OUTGOING: PyDirection = PyDirection { inner: Direction::Outgoing };

    /// Class attribute: INCOMING
    #[classattr]
    const INCOMING: PyDirection = PyDirection { inner: Direction::Incoming };

    /// Class attribute: BOTH
    #[classattr]
    const BOTH: PyDirection = PyDirection { inner: Direction::Both };
}

// Conversion: &PyDirection -> Direction
impl From<&PyDirection> for Direction {
    fn from(py_dir: &PyDirection) -> Self {
        py_dir.inner
    }
}

impl PyDirection {
    /// Return the underlying Direction value (for internal use).
    pub(crate) fn inner(&self) -> Direction {
        self.inner
    }
}

// ============================================================================
// PyNeighbor
// ============================================================================

/// A neighbor node returned by graph traversal.
///
/// Contains the neighbor's NodeId and the EdgeId that connects to it.
///
/// Example:
/// ```python
/// neighbor = neighbors[0]
/// print(neighbor.node_id)   # NodeId(42)
/// print(neighbor.edge_id)   # EdgeId(7)
/// ```
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyNeighbor {
    #[pyo3(get)]
    pub node_id: PyNodeId,
    #[pyo3(get)]
    pub edge_id: PyEdgeId,
}

#[pymethods]
impl PyNeighbor {
    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!(
            "Neighbor(node_id={}, edge_id={})",
            self.node_id.inner(),
            self.edge_id.inner()
        )
    }

    /// Compare equality with another Neighbor.
    fn __eq__(&self, other: &PyNeighbor) -> bool {
        self.node_id.inner() == other.node_id.inner()
            && self.edge_id.inner() == other.edge_id.inner()
    }
}

impl PyNeighbor {
    /// Create a PyNeighbor from spider-core Neighbor.
    pub(crate) fn from_rust(neighbor: traverse::Neighbor) -> Self {
        PyNeighbor {
            node_id: PyNodeId::from(neighbor.node_id.get()),
            edge_id: PyEdgeId::from(neighbor.edge_id.get()),
        }
    }
}

// ============================================================================
// PyBioTier
// ============================================================================

/// Bio-inspired storage tier classification for nodes in the Spider graph database.
///
/// Nodes are classified into tiers based on their "vitality" score:
///     HOT: In RAM, embeddings cached, instant access (score > 20.0)
///     WARM: On SSD, metadata in memory, fast I/O path (score > 5.0)
///     COLD: Archived, slow access, future compressed off-SSD storage (score > 0.0)
///     PRUNED: Eligible for deletion (score <= 0.0)
///
/// Example:
/// ```python
/// import spider
///
/// tier = spider.BioTier.from_score(100.0)
/// print(tier)  # Hot
/// print(tier.is_active())  # True
///
/// tier = spider.BioTier.from_score(-5.0)
/// print(tier.is_prunable())  # True
/// ```
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyBioTier {
    inner: spider_core::bio::tier::BioTier,
}

use spider_core::bio::tier::BioTier;

#[pymethods]
impl PyBioTier {
    /// Classify a bio score into a storage tier.
    ///
    /// Args:
    ///     score: The bio vitality score (floating point number).
    ///
    /// Returns:
    ///     BioTier: The corresponding storage tier.
    #[staticmethod]
    fn from_score(score: f64) -> Self {
        PyBioTier {
            inner: BioTier::from_score(score),
        }
    }

    /// Check if this node is eligible for pruning (score <= 0).
    ///
    /// Returns:
    ///     bool: True if this tier is Pruned.
    fn is_prunable(&self) -> bool {
        self.inner.is_prunable()
    }

    /// Check if this node is in active storage (Warm or Hot).
    ///
    /// Returns:
    ///     bool: True if this tier is Warm or Hot.
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Convert to a human-readable string.
    ///
    /// Returns:
    ///     str: "Hot", "Warm", "Cold", or "Pruned".
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!("BioTier.{}", self.__str__())
    }

    /// Compare equality with another BioTier.
    fn __eq__(&self, other: &PyBioTier) -> bool {
        self.inner == other.inner
    }

    /// Compare ordering (for sorting by tier quality).
    fn __lt__(&self, other: &PyBioTier) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &PyBioTier) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &PyBioTier) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &PyBioTier) -> bool {
        self.inner >= other.inner
    }

    /// Return a hash value.
    fn __hash__(&self) -> u64 {
        match self.inner {
            BioTier::Pruned => 0,
            BioTier::Cold => 1,
            BioTier::Warm => 2,
            BioTier::Hot => 3,
        }
    }

    /// Class attribute: HOT
    #[classattr]
    const HOT: PyBioTier = PyBioTier { inner: BioTier::Hot };

    /// Class attribute: WARM
    #[classattr]
    const WARM: PyBioTier = PyBioTier { inner: BioTier::Warm };

    /// Class attribute: COLD
    #[classattr]
    const COLD: PyBioTier = PyBioTier { inner: BioTier::Cold };

    /// Class attribute: PRUNED
    #[classattr]
    const PRUNED: PyBioTier = PyBioTier { inner: BioTier::Pruned };
}

// Conversion: BioTier -> PyBioTier
impl From<BioTier> for PyBioTier {
    fn from(tier: BioTier) -> Self {
        PyBioTier { inner: tier }
    }
}
