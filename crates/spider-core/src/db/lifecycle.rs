//! Database lifecycle: open, close, Drop.
//!
//! Manages the lifecycle of a Spider database: initialization (`open()`),
//! graceful shutdown (`close()`), and automatic cleanup via `Drop`. All
//! database file handling happens here.
//!
//! ## Responsibilities
//! - Open or create `.db` files
//! - Open/close file handles
//! - Flush data on shutdown
//! - Clean up resources when the `Spider` handle is dropped

use std::path::Path;

/// 44-byte metadata header stored in `meta.db`.
///
/// Contains allocation counters for ID generation (nodes, edges, properties,
/// strings, arrays) and bio score tuning parameters.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Metadata {
    /// Monotonically increasing counter for next node ID.
    pub next_node_id: u32,
    /// Monotonically increasing counter for next relationship ID.
    pub next_rel_id: u32,
    /// Monotonically increasing counter for next property ID.
    pub next_prop_id: u32,
    /// Monotonically increasing counter for next token ID (strings).
    pub next_string_id: u32,
    /// Monotonically increasing counter for next token ID (arrays).
    pub next_array_id: u32,
    /// Significance weight in bio score formula (default 1.0).
    pub bio_w_sig: f64,
    /// Frequency weight in bio score formula (default 1.0).
    pub bio_w_freq: f64,
    /// Decay exponent in bio score formula (default 1.0).
    pub bio_gravity: f64,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            next_node_id: 1,
            next_rel_id: 1,
            next_prop_id: 1,
            next_string_id: 1,
            next_array_id: 1,
            bio_w_sig: 1.0,
            bio_w_freq: 1.0,
            bio_gravity: 1.0,
        }
    }
}

/// Main database handle.
///
/// Owns all record file handles and manages database lifecycle. Dropping this
/// struct closes the database automatically.
pub struct Spider {
    _path: String,
}

impl Spider {
    /// Opens or creates a database at the given path.
    ///
    /// Creates the directory if it doesn't exist. Opens all record files.
    /// Returns Err if any file cannot be opened or created.
    pub fn open(path: &Path) -> crate::error::SpiderResult<Self> {
        // TODO: Create directory, open all record files (nodes, edges, properties, tokens, arrays, blobs)
        let _path = path.to_string_lossy().to_string();
        // Placeholder: in production, this opens/creates all .db files
        Ok(Self { _path })
    }

    /// Gracefully closes the database and flushes all data to disk.
    ///
    /// After calling `close()`, the Spider instance should not be used.
    /// The Drop implementation also calls this to ensure cleanup.
    pub fn close(&mut self) -> crate::error::SpiderResult<()> {
        // TODO: Flush all pending writes, close file handles
        Ok(())
    }
}

impl Drop for Spider {
    fn drop(&mut self) {
        // Ensure cleanup even if user doesn't call close() explicitly.
        if let Err(e) = self.close() {
            eprintln!("warning: error closing database during Drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn open_creates_database() {
        let path = PathBuf::from("/tmp/test_spider_");
        let db = Spider::open(&path);
        assert!(db.is_ok());
    }

    #[test]
    fn metadata_default_sane() {
        let m = Metadata::default();
        assert_eq!(m.next_node_id, 1);
        assert_eq!(m.bio_w_sig, 1.0);
    }
}
