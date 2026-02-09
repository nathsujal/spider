//! # Spider Storage Schema
//!
//! Fixed-size record types for Spider's graph storage layer.
//!
//! This module provides the core data structures for persisting graph data:
//!
//! - [`NodeRecord`] - Graph nodes with up to 4 labels (16 bytes)
//! - [`RelRecord`] - Doubly-linked relationships (33 bytes)
//! - [`PropertyRecord`] - Key-value properties in linked blocks (40 bytes)
//! - [`DynamicStringRecord`] - Large string storage (128 bytes)
//! - [`DynamicArrayRecord`] - Large array storage (128 bytes)
//! - [`TokenStore`] - String interning for labels, types, and keys
//!
//! ## Design Principles
//!
//! - **Fixed-size records**: Enable O(1) lookups via `id * record_size`
//! - **ID = 0 means deleted**: No separate tombstone tracking needed
//! - **Manual serialization**: Precise byte layout for on-disk format
//!
//! ## Storage Files
//!
//! ```text
//! data/spider.db/
//! ├── nodes.db         # NodeRecord (16 bytes each)
//! ├── rels.db          # RelRecord (33 bytes each)
//! ├── props.db         # PropertyRecord (40 bytes each)
//! ├── strings.db       # DynamicStringRecord (128 bytes each)
//! ├── arrays.db        # DynamicArrayRecord (128 bytes each)
//! ├── labels.tok       # TokenStore for label names
//! ├── reltypes.tok     # TokenStore for relationship type names
//! └── propkeys.tok     # TokenStore for property key names
//! ```

mod dynamic;
mod node;
mod property;
mod relationship;
mod token;

pub use dynamic::{DynamicArrayRecord, DynamicStringRecord};
pub use node::NodeRecord;
pub use property::{PropertyBlock, PropertyRecord, PropertyType};
pub use relationship::RelRecord;
pub use token::TokenStore;
