//! Python bindings for SpiderDB.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use std::path::PathBuf;

use crate::db::{DbError, Direction, PropertyValue, SpiderDB};

/// Convert DbError to PyErr
fn to_py_err(e: DbError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Python wrapper for SpiderDB.
#[pyclass]
pub struct PySpiderDB {
    db: Option<SpiderDB>,
}

#[pymethods]
impl PySpiderDB {
    /// Create and open a new SpiderDB.
    ///
    /// Optionally pass bio scoring parameters to override the defaults.
    /// If not provided, the persisted values from meta.db are used
    #[new]
    #[pyo3(signature = (path, w_sig=None, w_freq=None, gravity=None))]
    fn new(
        path: &str,
        w_sig: Option<f64>,
        w_freq: Option<f64>,
        gravity: Option<f64>,
    ) -> PyResult<Self> {
        let db = SpiderDB::open_with_bio_params(
            PathBuf::from(path), w_sig, w_freq, gravity,
        ).map_err(to_py_err)?;
        Ok(Self { db: Some(db) })
    }

    /// Create a new node with the given labels.
    ///
    /// Returns the new node's ID.
    fn create_node(&mut self, labels: Vec<String>) -> PyResult<u32> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        db.create_node(&label_refs).map_err(to_py_err)
    }

    /// Get a node by ID. Returns None if not found.
    fn get_node(&self, id: u32) -> PyResult<Option<PyNode>> {
        let db = self.db.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        Ok(db.get_node(id).map(|n| PyNode {
            id: n.id,
            first_rel_id: n.first_rel_id,
            first_prop_id: n.first_prop_id,
            labels: n.get_labels(),
            access_count: n.access_count,
            significance: n.significance,
            created_at: n.created_at,
            last_accessed_at: n.last_accessed_at,
        }))
    }

    /// Delete a node by ID.
    fn delete_node(&mut self, id: u32) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        db.delete_node(id).map_err(to_py_err)
    }

    /// Create a relationship between two nodes.
    ///
    /// Returns the new relationship's ID.
    fn create_rel(&mut self, source_id: u32, target_id: u32, rel_type: &str) -> PyResult<u32> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        db.create_rel(source_id, target_id, rel_type).map_err(to_py_err)
    }

    /// Get a relationship by ID. Returns None if not found.
    fn get_rel(&self, id: u32) -> PyResult<Option<PyRel>> {
        let db = self.db.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        Ok(db.get_rel(id).map(|r| PyRel {
            id: r.id,
            source_id: r.source_id,
            target_id: r.target_id,
            rel_type_id: r.rel_type_id,
        }))
    }

    /// Delete a relationship by ID.
    fn delete_rel(&mut self, id: u32) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        db.delete_rel(id).map_err(to_py_err)
    }

    /// Get all neighbor node IDs for a given node.
    fn get_neighbors(&self, node_id: u32) -> PyResult<Vec<u32>> {
        let db = self.db.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Database is closed")
        })?;
        
        Ok(db.get_neighbors(node_id))
    }

    // ─── Property Operations ─────────────────────────────────────────────────

    /// Set a property on a node. Accepts bool, int, or float values.
    fn set_node_property(&mut self, node_id: u32, key: &str, value: &PyAny) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let val = py_to_property_value(value)?;
        db.set_node_property(node_id, key, val).map_err(to_py_err)
    }

    /// Get a property from a node. Returns None if not found.
    fn get_node_property(&self, py: Python, node_id: u32, key: &str) -> PyResult<PyObject> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        match db.get_node_property(node_id, key).map_err(to_py_err)? {
            Some(val) => Ok(property_value_to_py(py, val)),
            None => Ok(py.None()),
        }
    }

    /// Delete a property from a node.
    fn delete_node_property(&mut self, node_id: u32, key: &str) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.delete_node_property(node_id, key).map_err(to_py_err)
    }

    /// Set a property on a relationship.
    fn set_rel_property(&mut self, rel_id: u32, key: &str, value: &PyAny) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let val = py_to_property_value(value)?;
        db.set_rel_property(rel_id, key, val).map_err(to_py_err)
    }

    /// Get a property from a relationship.
    fn get_rel_property(&self, py: Python, rel_id: u32, key: &str) -> PyResult<PyObject> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        match db.get_rel_property(rel_id, key).map_err(to_py_err)? {
            Some(val) => Ok(property_value_to_py(py, val)),
            None => Ok(py.None()),
        }
    }

    /// Delete a property from a relationship.
    fn delete_rel_property(&mut self, rel_id: u32, key: &str) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.delete_rel_property(rel_id, key).map_err(to_py_err)
    }

    // ─── Query & Retrieval ───────────────────────────────────────────────────

    /// Get all properties of a node as a dict.
    fn get_all_node_properties(&self, py: Python, node_id: u32) -> PyResult<PyObject> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let map = db.get_all_node_properties(node_id).map_err(to_py_err)?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in map {
            dict.set_item(k, property_value_to_py(py, v))?;
        }
        Ok(dict.into())
    }

    /// Get all properties of a relationship as a dict.
    fn get_all_rel_properties(&self, py: Python, rel_id: u32) -> PyResult<PyObject> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let map = db.get_all_rel_properties(rel_id).map_err(to_py_err)?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in map {
            dict.set_item(k, property_value_to_py(py, v))?;
        }
        Ok(dict.into())
    }

    /// Get relationships for a node, filtered by direction and optional type.
    ///
    /// direction: "OUTGOING", "INCOMING", or "BOTH"
    /// rel_type: optional type filter (e.g., "KNOWS")
    fn get_relationships(
        &self,
        node_id: u32,
        direction: &str,
        rel_type: Option<&str>,
    ) -> PyResult<Vec<PyRelInfo>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let dir = match direction.to_uppercase().as_str() {
            "OUTGOING" | "OUT" => Direction::Outgoing,
            "INCOMING" | "IN" => Direction::Incoming,
            "BOTH" | "ALL" => Direction::Both,
            _ => return Err(PyRuntimeError::new_err(
                format!("Invalid direction: '{}'. Use OUTGOING, INCOMING, or BOTH", direction)
            )),
        };
        let rels = db.get_relationships(node_id, dir, rel_type).map_err(to_py_err)?;
        Ok(rels.into_iter().map(|r| PyRelInfo {
            rel_id: r.rel_id,
            source_id: r.source_id,
            target_id: r.target_id,
            rel_type: r.rel_type,
        }).collect())
    }

    /// Get the type name of a relationship.
    fn get_rel_type_name(&self, rel_id: u32) -> PyResult<Option<String>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.get_rel_type_name(rel_id).map_err(to_py_err)
    }

    /// Find all node IDs with a given label.
    fn find_nodes_by_label(&self, label: &str) -> PyResult<Vec<u32>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.find_nodes_by_label(label).map_err(to_py_err)
    }

    /// Find all node IDs where a property matches a value.
    fn find_nodes_by_property(&self, key: &str, value: &PyAny) -> PyResult<Vec<u32>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        let val = py_to_property_value(value)?;
        db.find_nodes_by_property(key, &val).map_err(to_py_err)
    }

    // ─── Bio Operations ─────────────────────────────────────────────────────

    /// Touch a node — strengthens its memory.
    fn touch_node(&mut self, id: u32) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.touch_node(id).map_err(to_py_err)
    }

    /// Set the significance (importance) of a node. 0-255.
    fn set_significance(&mut self, id: u32, significance: u8) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.set_significance(id, significance).map_err(to_py_err)
    }

    /// Get the current bio score (life force) of a node.
    fn get_bio_score(&self, id: u32) -> PyResult<f64> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        Ok(db.get_bio_score(id))
    }

    /// Set the database-level bio scoring parameters.
    ///
    /// These are persisted to meta.db and used for all nodes.
    fn set_bio_params(&mut self, w_sig: f64, w_freq: f64, gravity: f64) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        db.set_bio_params(w_sig, w_freq, gravity);
        Ok(())
    }

    /// Get the current bio scoring parameters as (w_sig, w_freq, gravity).
    fn get_bio_params(&self) -> PyResult<(f64, f64, f64)> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        Ok(db.get_bio_params())
    }

    /// Get all live (non-deleted) node IDs.
    fn get_all_node_ids(&self) -> PyResult<Vec<u32>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        Ok(db.get_all_node_ids())
    }

    /// Number of live (non-deleted) nodes.
    fn node_count(&self) -> PyResult<usize> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database is closed"))?;
        Ok(db.node_count())
    }

    // ─────────────────────────────────────────────────────────────────────────

    /// Close the database, flushing all data to disk.
    fn close(&mut self) -> PyResult<()> {
        if let Some(mut db) = self.db.take() {
            db.close().map_err(to_py_err)?;
        }
        Ok(())
    }

    /// Check if database is open.
    fn is_open(&self) -> bool {
        self.db.is_some()
    }

    // ─── Content Operations ──────────────────────────────────────────────────

    /// Store binary content and create a node for it.
    ///
    /// MIME type is auto-detected. Returns (node_id, blob_hash).
    fn store_content(&mut self, data: &[u8], name: &str, labels: Vec<String>) -> PyResult<(u32, String)> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("DB closed"))?;
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        db.store_content(data, name, &label_refs).map_err(to_py_err)
    }

    /// Read binary content for a content node.
    fn read_content<'a>(&self, py: Python<'a>, node_id: u32) -> PyResult<&'a PyBytes> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("DB closed"))?;
        let data = db.read_content(node_id).map_err(to_py_err)?;
        Ok(PyBytes::new(py, &data))
    }

    /// Delete a content node and decrement blob ref count.
    fn delete_content_node(&mut self, node_id: u32) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("DB closed"))?;
        db.delete_content_node(node_id).map_err(to_py_err)
    }

    /// Content store stats: (blob_count, total_bytes).
    fn content_stats(&self) -> PyResult<(usize, u64)> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("DB closed"))?;
        Ok(db.content_stats())
    }

    /// Garbage collect unreferenced blobs. Returns count removed.
    fn content_gc(&mut self) -> PyResult<usize> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("DB closed"))?;
        db.content_gc().map_err(to_py_err)
    }
}

// ─── Type Conversion Helpers ─────────────────────────────────────────────────

/// Convert a Python value to PropertyValue.
/// Checks bool first because Python `bool` is a subclass of `int`.
fn py_to_property_value(obj: &PyAny) -> PyResult<PropertyValue> {
    // bool MUST be checked before i64 (True/False are also ints in Python)
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
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        format!("Unsupported property type: {}", obj.get_type())
    ))
}

/// Convert PropertyValue to a Python object.
fn property_value_to_py(py: Python, val: PropertyValue) -> PyObject {
    match val {
        PropertyValue::Bool(v) => v.to_object(py),
        PropertyValue::Int(v) => v.to_object(py),
        PropertyValue::Float(v) => v.to_object(py),
        PropertyValue::String(v) => v.to_object(py),
    }
}

/// Python wrapper for NodeRecord.
#[pyclass]
#[derive(Clone)]
pub struct PyNode {
    #[pyo3(get)]
    id: u32,
    #[pyo3(get)]
    first_rel_id: u32,
    #[pyo3(get)]
    first_prop_id: u32,
    #[pyo3(get)]
    labels: Vec<u8>,
    #[pyo3(get)]
    access_count: u32,
    #[pyo3(get)]
    significance: u8,
    #[pyo3(get)]
    created_at: u32,
    #[pyo3(get)]
    last_accessed_at: u32,
}

#[pymethods]
impl PyNode {
    fn __repr__(&self) -> String {
        format!(
            "Node(id={}, labels={:?}, access={}, sig={})",
            self.id, self.labels, self.access_count, self.significance
        )
    }
}

/// Python wrapper for RelRecord.
#[pyclass]
#[derive(Clone)]
pub struct PyRel {
    #[pyo3(get)]
    id: u32,
    #[pyo3(get)]
    source_id: u32,
    #[pyo3(get)]
    target_id: u32,
    #[pyo3(get)]
    rel_type_id: u8,
}

#[pymethods]
impl PyRel {
    fn __repr__(&self) -> String {
        format!(
            "Rel(id={}, {}->{})",
            self.id, self.source_id, self.target_id
        )
    }
}

/// Python wrapper for RelInfo (query result).
#[pyclass]
#[derive(Clone)]
pub struct PyRelInfo {
    #[pyo3(get)]
    rel_id: u32,
    #[pyo3(get)]
    source_id: u32,
    #[pyo3(get)]
    target_id: u32,
    #[pyo3(get)]
    rel_type: String,
}

#[pymethods]
impl PyRelInfo {
    fn __repr__(&self) -> String {
        format!(
            "RelInfo(id={}, {}-[{}]->{} )",
            self.rel_id, self.source_id, self.rel_type, self.target_id
        )
    }
}

/// Register Python classes with the module.
pub fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySpiderDB>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyRel>()?;
    m.add_class::<PyRelInfo>()?;
    Ok(())
}
