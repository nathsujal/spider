//! Spider - Bio-Inspired AI Agent Memory Graph
//!
//! A context-aware graph database that behaves like a human brain.

use pyo3::prelude::*;

// TODO: These modules need refactoring to use new schema
// pub mod bio;
// pub mod cluster;
// pub mod ranking;

pub mod schema;

// Re-export schema types
pub use schema::{
    NodeRecord, RelRecord, PropertyRecord, PropertyBlock, PropertyType,
    DynamicStringRecord, DynamicArrayRecord, TokenStore,
};

/// A Python module implemented in Rust.
#[pymodule]
fn spider(_py: Python, _m: &PyModule) -> PyResult<()> {
    // TODO: Add Python bindings for schema types
    Ok(())
}
