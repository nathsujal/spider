//! Python bindings for Spider.
//!
//! Exposes three classes:
//!   - `Memory`  — the database handle (main entry point)
//!   - `Node`    — a node record returned by get_node / find_by_*
//!   - `Edge`    — a relationship record returned by get_edge / edges

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

use crate::db::{DbError, Direction, PropertyValue, Spider};

/// Error Conversion/


fn to_py_err(e: DbError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

macro_rules! db_ref {
    ($self:expr) => {
        $self.inner.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?
    };
}

macro_rules! db_mut {
    ($self:expr) => {
        $self.inner.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?
    };
}

/// Memory (main class) 

/// A Spider memory graph database.
///
/// Example:
///     mem = Memory()                          # OS default path
///     mem = Memory("./agent.db")              # explicit path
///     mem = Memory(gravity=2.0)               # tuned decay
///
///     alice = mem.create_node(["Person"])
///     mem.set_property(alice, "name", "Alice")
///     mem.reinforce(alice)
///     print(mem.vitality(alice))
#[pyclass]
pub struct Memory {
    inner: Option<Spider>,
}

#[pymethods]
impl Memory {
    /// Open a Spider memory graph.
    ///
    /// Args:
    ///     path:    Path to the database directory. Defaults to the OS data directory.
    ///     w_sig:   Significance weight for bio scoring (default: persisted or 1.0).
    ///     w_freq:  Frequency weight for bio scoring (default: persisted or 1.0).
    ///     gravity: Decay exponent (default: persisted or 1.0). Higher = faster forgetting.
    #[new]
    #[pyo3(signature = (path=None, w_sig=None, w_freq=None, gravity=None))]
    fn new(
        path: Option<&str>,
        w_sig: Option<f64>,
        w_freq: Option<f64>,
        gravity: Option<f64>,
    ) -> PyResult<Self> {
        let path_buf = path.map(PathBuf::from);
        let db = Spider::open(path_buf, w_sig, w_freq, gravity)
            .map_err(to_py_err)?;
        Ok(Self { inner: Some(db) })
    }

    /// Node Operations 

    /// Create a new node with the given labels. Returns the node ID.
    fn create_node(&mut self, labels: Vec<String>) -> PyResult<u32> {
        let db = db_mut!(self);
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        db.create_node(&label_refs).map_err(to_py_err)
    }

    /// Get a node by ID. Returns None if not found.
    fn get_node(&self, id: u32) -> PyResult<Option<Node>> {
        let db = db_ref!(self);
        Ok(db.get_node(id).map(|n| Node {
            id: n.id,
            labels: n.get_labels(),
            access_count: n.access_count,
            significance: n.significance,
            created_at: n.created_at,
            last_accessed_at: n.last_accessed_at,
        }))
    }

    /// Delete a node and all its relationships by ID.
    fn remove_node(&mut self, id: u32) -> PyResult<()> {
        let db = db_mut!(self);
        db.delete_node(id).map_err(to_py_err)
    }

    /// Get all live node IDs in the database.
    fn all_node_ids(&self) -> PyResult<Vec<u32>> {
        let db = db_ref!(self);
        Ok(db.get_all_node_ids())
    }

    /// Number of live nodes.
    fn node_count(&self) -> PyResult<usize> {
        let db = db_ref!(self);
        Ok(db.node_count())
    }

    /// Edge Operations 

    /// Create a directed edge between two nodes. Returns the edge ID.
    fn add_edge(&mut self, source_id: u32, target_id: u32, kind: &str) -> PyResult<u32> {
        let db = db_mut!(self);
        db.create_rel(source_id, target_id, kind).map_err(to_py_err)
    }

    /// Get an edge by ID. Returns None if not found.
    fn get_edge(&self, id: u32) -> PyResult<Option<Edge>> {
        let db = db_ref!(self);
        Ok(db.get_rel(id).and_then(|r| {
            db.get_rel_type_name(r.id).ok().flatten().map(|kind| Edge {
                id: r.id,
                source_id: r.source_id,
                target_id: r.target_id,
                kind,
            })
        }))
    }

    /// Delete an edge by ID.
    fn remove_edge(&mut self, id: u32) -> PyResult<()> {
        let db = db_mut!(self);
        db.delete_rel(id).map_err(to_py_err)
    }

    /// Get edges for a node, filtered by direction and optional type.
    ///
    /// direction: "outgoing", "incoming", or "both" (case-insensitive).
    fn edges(
        &self,
        node_id: u32,
        direction: &str,
        kind: Option<&str>,
    ) -> PyResult<Vec<Edge>> {
        let db = db_ref!(self);
        let dir = match direction.to_lowercase().as_str() {
            "outgoing" | "out" => Direction::Outgoing,
            "incoming" | "in"  => Direction::Incoming,
            "both" | "all"     => Direction::Both,
            _ => return Err(PyRuntimeError::new_err(format!(
                "Invalid direction '{}'. Use 'outgoing', 'incoming', or 'both'.", direction
            ))),
        };
        let rels = db.get_relationships(node_id, dir, kind).map_err(to_py_err)?;
        Ok(rels.into_iter().map(|r| Edge {
            id: r.rel_id,
            source_id: r.source_id,
            target_id: r.target_id,
            kind: r.rel_type,
        }).collect())
    }

    /// Get all neighbor node IDs for a given node.
    fn neighbors(&self, node_id: u32) -> PyResult<Vec<u32>> {
        let db = db_ref!(self);
        Ok(db.get_neighbors(node_id))
    }

    /// Property Operations

    /// Set a property on a node. Value may be bool, int, float, or str.
    fn set_property(&mut self, node_id: u32, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let db = db_mut!(self);
        db.set_node_property(node_id, key, py_to_property_value(value)?)
            .map_err(to_py_err)
    }

    /// Get a property from a node. Returns None if the key does not exist.
    fn get_property(&self, py: Python<'_>, node_id: u32, key: &str) -> PyResult<PyObject> {
        let db = db_ref!(self);
        match db.get_node_property(node_id, key).map_err(to_py_err)? {
            Some(val) => Ok(property_value_to_py(py, val)),
            None      => Ok(py.None()),
        }
    }

    /// Remove a property from a node.
    fn remove_property(&mut self, node_id: u32, key: &str) -> PyResult<()> {
        let db = db_mut!(self);
        db.delete_node_property(node_id, key).map_err(to_py_err)
    }

    /// Get all properties of a node as a dict.
    fn get_properties(&self, py: Python<'_>, node_id: u32) -> PyResult<PyObject> {
        let db = db_ref!(self);
        let map = db.get_all_node_properties(node_id).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        for (k, v) in map {
            dict.set_item(k, property_value_to_py(py, v))?;
        }
        Ok(dict.into())
    }

    /// Set a property on an edge.
    fn set_edge_property(&mut self, edge_id: u32, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let db = db_mut!(self);
        db.set_rel_property(edge_id, key, py_to_property_value(value)?)
            .map_err(to_py_err)
    }

    /// Get a property from an edge. Returns None if the key does not exist.
    fn get_edge_property(&self, py: Python<'_>, edge_id: u32, key: &str) -> PyResult<PyObject> {
        let db = db_ref!(self);
        match db.get_rel_property(edge_id, key).map_err(to_py_err)? {
            Some(val) => Ok(property_value_to_py(py, val)),
            None      => Ok(py.None()),
        }
    }

    /// Remove a property from an edge.
    fn remove_edge_property(&mut self, edge_id: u32, key: &str) -> PyResult<()> {
        let db = db_mut!(self);
        db.delete_rel_property(edge_id, key).map_err(to_py_err)
    }

    /// Get all properties of an edge as a dict.
    fn get_edge_properties(&self, py: Python<'_>, edge_id: u32) -> PyResult<PyObject> {
        let db = db_ref!(self);
        let map = db.get_all_rel_properties(edge_id).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        for (k, v) in map {
            dict.set_item(k, property_value_to_py(py, v))?;
        }
        Ok(dict.into())
    }

    /// Search


    /// Find all node IDs that carry a given label.
    fn find_by_label(&self, label: &str) -> PyResult<Vec<u32>> {
        let db = db_ref!(self);
        db.find_nodes_by_label(label).map_err(to_py_err)
    }

    /// Find all node IDs where a property matches a value.
    fn find_by_property(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        let db = db_ref!(self);
        let val = py_to_property_value(value)?;
        db.find_nodes_by_property(key, &val).map_err(to_py_err)
    }

    ///Bio Operations

    /// Reinforce a node — records an access and strengthens its bio score.
    fn reinforce(&mut self, id: u32) -> PyResult<()> {
        let db = db_mut!(self);
        db.touch_node(id).map_err(to_py_err)
    }

    /// Set the importance (significance) of a node. Range: 0–255.
    fn set_importance(&mut self, id: u32, importance: u8) -> PyResult<()> {
        let db = db_mut!(self);
        db.set_significance(id, importance).map_err(to_py_err)
    }

    /// Get the current vitality (bio score) of a node.
    fn vitality(&self, id: u32) -> PyResult<f64> {
        let db = db_ref!(self);
        Ok(db.get_bio_score(id))
    }

    /// Get the current bio scoring parameters as (w_sig, w_freq, gravity).
    fn scoring_params(&self) -> PyResult<(f64, f64, f64)> {
        let db = db_ref!(self);
        db.get_bio_params().map_err(to_py_err)
    }

    /// Update the bio scoring parameters.
    fn set_scoring_params(&mut self, w_sig: f64, w_freq: f64, gravity: f64) -> PyResult<()> {
        let db = db_mut!(self);
        db.set_bio_params(w_sig, w_freq, gravity).map_err(to_py_err)?;
        Ok(())
    }

    /// File Operations

    /// Store binary file and create a node for it. Returns (node_id, blob_hash).
    fn file(&mut self, data: &[u8], name: &str, labels: Vec<String>) -> PyResult<u32> {
        let db = db_mut!(self);
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        db.file(data, name, &label_refs).map_err(to_py_err)
    }

    /// Read binary file for a file node. Returns raw bytes.
    fn read_file(&self, py: Python<'_>, node_id: u32) -> PyResult<PyObject> {
        let db = db_ref!(self);
        let data = db.read_file(node_id).map_err(to_py_err)?;
        Ok(PyBytes::new_bound(py, &data).into())
    }

    /// Remove a file node and decrement the blob reference count.
    fn delete_file(&mut self, node_id: u32) -> PyResult<()> {
        let db = db_mut!(self);
        db.delete_file_node(node_id).map_err(to_py_err)
    }

    /// Content store statistics: (blob_count, total_bytes).
    fn file_stats(&self) -> PyResult<(usize, u64)> {
        let db = db_ref!(self);
        Ok(db.file_stats())
    }

    /// Remove unreferenced blobs from the file store. Returns the count removed.
    fn file_blobs(&mut self) -> PyResult<usize> {
        let db = db_mut!(self);
        db.gc_files().map_err(to_py_err)
    }

    /// Lifecycle

    /// Flush all data to disk and close the database.
    fn close(&mut self) -> PyResult<()> {
        if let Some(mut db) = self.inner.take() {
            db.close().map_err(to_py_err)?;
        }
        Ok(())
    }

    /// Returns True if the database is open.
    fn is_open(&self) -> bool {
        self.inner.is_some()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(_) => "Memory(open)".to_string(),
            None    => "Memory(closed)".to_string(),
        }
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Node/


#[pyclass]
#[derive(Clone)]
pub struct Node {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub labels: Vec<u8>,
    #[pyo3(get)]
    pub access_count: u32,
    #[pyo3(get)]
    pub significance: u8,
    #[pyo3(get)]
    pub created_at: u32,
    #[pyo3(get)]
    pub last_accessed_at: u32,
}

#[pymethods]
impl Node {
    fn __repr__(&self) -> String {
        format!(
            "Node(id={}, labels={:?}, access_count={}, significance={})",
            self.id, self.labels, self.access_count, self.significance,
        )
    }
}

/// Edge/


#[pyclass]
#[derive(Clone)]
pub struct Edge {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub source_id: u32,
    #[pyo3(get)]
    pub target_id: u32,
    #[pyo3(get)]
    pub kind: String,
}

#[pymethods]
impl Edge {
    fn __repr__(&self) -> String {
        format!(
            "Edge(id={}, {} -[{}]-> {})",
            self.id, self.source_id, self.kind, self.target_id,
        )
    }
}

/// Type Conversion Helpers/──



fn py_to_property_value(obj: &Bound<'_, PyAny>) -> PyResult<PropertyValue> {
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(PropertyValue::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(PropertyValue::Int(i));
    }
    if let Ok(f) = obj.extract::<f32>() {
        return Ok(PropertyValue::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(PropertyValue::String(s));
    }
    Err(PyErr::new::<PyTypeError, _>(format!(
        "Unsupported property type: {}. Use bool, int, float, or str.", obj.get_type()
    )))
}

fn property_value_to_py(py: Python<'_>, val: PropertyValue) -> PyObject {
    match val {
        PropertyValue::Bool(v)   => v.into_py(py),
        PropertyValue::Int(v)    => v.into_py(py),
        PropertyValue::Float(v)  => v.into_py(py),
        PropertyValue::String(v) => v.into_py(py),
    }
}

/// Module Registration 

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<Node>()?;
    m.add_class::<Edge>()?;
    Ok(())
}