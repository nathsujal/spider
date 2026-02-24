//! # Store Layer
//!
//! Memory-mapped file storage for fixed-size records.
//!
//! This module provides the persistence layer for Spider's graph data:
//! - [`RecordFile`] - Generic memory-mapped file for any record type
//! - [`FreeList`] - Track deleted slots for ID reuse
//! - [`Metadata`] - Database state (next IDs, counts)
//!
//! ## Design Principles
//!
//! - **Fixed-size records**: O(1) lookup via `offset = HEADER + (id-1) * SIZE`
//! - **Memory-mapped I/O**: OS handles caching, fast random access
//! - **Periodic flush**: Durability without sacrificing speed
//! - **Never panic**: Return `Result`/`Option` for all errors
//!
//! ## File Layout
//!
//! ```text
//! RecordFile (e.g., nodes.db)
//! ┌────────────────────────────────────┐
//! │ HEADER (64 bytes)                  │
//! │ ├─ magic: [u8; 4]  = "SPDR"        │
//! │ ├─ version: u8                     │
//! │ ├─ record_size: u16                │
//! │ └─ capacity: u32                   │
//! ├────────────────────────────────────┤
//! │ RECORDS                            │
//! │ [record 1] [record 2] [record 3]...│
//! └────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use spider::store::{RecordFile, Record};
//!
//! // Open or create a file
//! let mut store = RecordFile::<NodeRecord>::open("nodes.db")?;
//!
//! // Write a record
//! store.write(1, &node)?;
//!
//! // Read it back
//! if let Some(node) = store.read(1) {
//!     println!("Node: {:?}", node);
//! }
//!
//! // Sync to disk
//! store.sync()?;
//! ```

mod error;
mod free_list;
mod metadata;
mod record_file;

pub use error::{Result, StoreError};
pub use free_list::FreeList;
pub use metadata::Metadata;
pub use record_file::RecordFile;

// ─────────────────────────────────────────────────────────────────────────────
// Record Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for types that can be stored in a [`RecordFile`].
///
/// Implementors must be fixed-size and provide serialization.
///
/// # Requirements
///
/// - `SIZE` must match the byte count returned by `to_bytes()`
/// - `from_bytes()` must be the inverse of `to_bytes()`
/// - `is_deleted()` should return true when `id == 0`
pub trait Record: Sized + Copy {
    /// Fixed size in bytes when serialized.
    const SIZE: usize;

    /// Serialize to bytes (must return exactly `SIZE` bytes).
    fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from bytes.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Check if this record is deleted (typically `id == 0`).
    fn is_deleted(&self) -> bool;
}
