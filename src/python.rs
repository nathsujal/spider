//! Python bindings for SpiderDB.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
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
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = SpiderDB::open(PathBuf::from(path)).map_err(to_py_err)?;
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
}

#[pymethods]
impl PyNode {
    fn __repr__(&self) -> String {
        format!("Node(id={}, labels={:?})", self.id, self.labels)
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
