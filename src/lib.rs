//! Spider - Bio-Inspired AI Agent Memory Graph
//!
//! A context-aware graph database that behaves like a human brain.

use pyo3::prelude::*;

// TODO: These modules need refactoring to use new schema
// pub mod bio;
// pub mod cluster;
// pub mod ranking;

pub mod schema;
pub mod store;
pub mod db;
mod python;

// Re-export schema types
pub use schema::{
    NodeRecord, RelRecord, PropertyRecord, PropertyBlock, PropertyType,
    DynamicStringRecord, DynamicArrayRecord, TokenStore,
};

// Re-export store types
pub use store::{RecordFile, FreeList, Metadata, Record, StoreError};

// Re-export database types
pub use db::{SpiderDB, DbError};

/// A Python module implemented in Rust.
#[pymodule]
fn spider(py: Python, m: &PyModule) -> PyResult<()> {
    python::register(py, m)?;
    Ok(())
}

