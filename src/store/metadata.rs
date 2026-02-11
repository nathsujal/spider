//! # Metadata Storage
//!
//! Database metadata persisted to `meta.db`.
//!
//! Tracks next IDs for all record types and database-level
//! bio scoring parameters.
//!
//! ## Layout (44 bytes)
//!
//! ```text
//! Metadata
//! ├── next_node_id: u32
//! ├── next_rel_id: u32
//! ├── next_prop_id: u32
//! ├── next_string_id: u32
//! ├── next_array_id: u32
//! ├── bio_w_sig: f64       (significance weight)
//! ├── bio_w_freq: f64      (frequency weight)
//! └── bio_gravity: f64     (decay exponent)
//! ```

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use super::Result;

/// Database metadata (44 bytes).
///
/// All IDs start at 1. ID 0 is reserved for deleted/empty.
/// Bio params default to 1.0 and are tunable per-database.
#[derive(Debug, Clone)]
pub struct Metadata {
    // ── ID Counters (20 bytes) ──
    pub next_node_id: u32,
    pub next_rel_id: u32,
    pub next_prop_id: u32,
    pub next_string_id: u32,
    pub next_array_id: u32,

    // ── Bio Scoring Config (24 bytes) ──
    /// Weight for significance component in bio score.
    pub bio_w_sig: f64,
    /// Weight for frequency component in bio score.
    pub bio_w_freq: f64,
    /// Decay exponent (higher = faster forgetting).
    pub bio_gravity: f64,
}

impl Metadata {
    /// Serialized size in bytes.
    pub const SIZE: usize = 44;

    /// Create new metadata with defaults.
    pub fn new() -> Self {
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

    /// Load from file, or create new if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let mut file = File::open(path)?;
        let mut bytes = [0u8; Self::SIZE];
        file.read_exact(&mut bytes)?;

        Ok(Self::from_bytes(&bytes))
    }

    /// Save to file.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(path)?;
        file.write_all(&self.to_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Serialize to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.next_node_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.next_rel_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.next_prop_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.next_string_id.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.next_array_id.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.bio_w_sig.to_le_bytes());
        bytes[28..36].copy_from_slice(&self.bio_w_freq.to_le_bytes());
        bytes[36..44].copy_from_slice(&self.bio_gravity.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes (little-endian).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            next_node_id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            next_rel_id: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            next_prop_id: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            next_string_id: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            next_array_id: u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            bio_w_sig: f64::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27]]),
            bio_w_freq: f64::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31], bytes[32], bytes[33], bytes[34], bytes[35]]),
            bio_gravity: f64::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39], bytes[40], bytes[41], bytes[42], bytes[43]]),
        }
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn new_starts_at_one() {
        let meta = Metadata::new();
        assert_eq!(meta.next_node_id, 1);
        assert_eq!(meta.next_rel_id, 1);
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.db");

        let mut meta = Metadata::new();
        meta.next_node_id = 100;
        meta.next_rel_id = 200;
        meta.save(&path).unwrap();

        let loaded = Metadata::load(&path).unwrap();
        assert_eq!(loaded.next_node_id, 100);
        assert_eq!(loaded.next_rel_id, 200);
    }

    #[test]
    fn load_nonexistent_creates_new() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.db");

        let meta = Metadata::load(&path).unwrap();
        assert_eq!(meta.next_node_id, 1);
    }
}
