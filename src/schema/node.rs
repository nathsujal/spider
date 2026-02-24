//! # Node Storage
//!
//! Nodes are the vertices in Spider's graph. Each node can have:
//! - Up to 4 labels (e.g., Person, Document, Entity)
//! - A linked list of properties
//! - A linked list of relationships
//! - Bio metrics (access count, timestamps, significance)
//!
//! ## Storage Layout (29 bytes)
//!
//! ```text
//! NodeRecord
//! ├── id: u32              (0 = deleted)
//! ├── first_rel_id: u32    (0 = no relationships)
//! ├── first_prop_id: u32   (0 = no properties)
//! ├── labels: [u8; 4]      (0 = empty slot)
//! ├── access_count: u32    (frequency of access)
//! ├── created_at: u32      (unix seconds)
//! ├── last_accessed_at: u32(unix seconds)
//! └── significance: u8     (0-255, importance)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use spider::schema::NodeRecord;
//!
//! let mut node = NodeRecord::new(1, &[1, 2], 1000);
//! node.add_label(3);
//! assert!(node.has_label(1));
//! assert!(node.has_label(2));
//! assert!(node.has_label(3));
//! ```

/// A graph node (29 bytes serialized).
///
/// Node IDs start at 1. ID 0 indicates a deleted/unused slot.
/// Labels use token IDs from [`TokenStore`](super::TokenStore).
#[derive(Debug, Clone, Copy)]
pub struct NodeRecord {
    // ── Structure (16 bytes) ──
    /// Node ID. 0 = deleted/unused slot.
    pub id: u32,
    /// First relationship ID in the chain. 0 = no relationships.
    pub first_rel_id: u32,
    /// First property record ID. 0 = no properties.
    pub first_prop_id: u32,
    /// Up to 4 label token IDs. 0 = empty slot.
    pub labels: [u8; 4],

    // ── Bio Metrics (12 bytes) ──
    /// Number of times this node was accessed/retrieved.
    pub access_count: u32,
    /// Unix timestamp (seconds) when node was created.
    pub created_at: u32,
    /// Unix timestamp (seconds) of last access. Used for decay calculation.
    pub last_accessed_at: u32,

    // ── Tuning (1 byte) ──
    /// User-assigned importance (0-255). Maps to 0.0-1.0. Default 128 (0.5).
    pub significance: u8,
}

impl NodeRecord {
    /// Serialized size in bytes.
    pub const SIZE: usize = 29;

    /// Create a new node with the given ID, labels, and current timestamp.
    ///
    /// Initializes bio metrics: access_count = 1, significance = 128 (0.5).
    ///
    /// # Arguments
    /// * `id` - Node ID (must be > 0)
    /// * `labels` - Up to 4 label token IDs
    /// * `now` - Current Unix timestamp in seconds
    pub fn new(id: u32, labels: &[u8], now: u32) -> Self {
        let mut label_arr = [0u8; 4];
        for (i, &label) in labels.iter().take(4).enumerate() {
            label_arr[i] = label;
        }
        Self {
            id,
            first_rel_id: 0,
            first_prop_id: 0,
            labels: label_arr,
            access_count: 1,
            created_at: now,
            last_accessed_at: now,
            significance: 128,
        }
    }

    /// Create an empty/deleted node record.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            id: 0,
            first_rel_id: 0,
            first_prop_id: 0,
            labels: [0; 4],
            access_count: 0,
            created_at: 0,
            last_accessed_at: 0,
            significance: 0,
        }
    }

    /// Returns true if this node slot is deleted/unused.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.id == 0
    }

    /// Returns true if this node has any relationships.
    #[inline]
    pub fn has_relationships(&self) -> bool {
        self.first_rel_id != 0
    }

    /// Returns true if this node has any properties.
    #[inline]
    pub fn has_properties(&self) -> bool {
        self.first_prop_id != 0
    }

    /// Count of assigned labels (0-4).
    #[inline]
    pub fn label_count(&self) -> usize {
        self.labels.iter().filter(|&&l| l != 0).count()
    }

    /// Returns true if the node has the given label.
    #[inline]
    pub fn has_label(&self, label_id: u8) -> bool {
        label_id != 0 && self.labels.contains(&label_id)
    }

    /// Add a label. Returns false if label is 0 or all slots are full.
    pub fn add_label(&mut self, label_id: u8) -> bool {
        if label_id == 0 {
            return false;
        }
        if self.has_label(label_id) {
            return true; // Already present
        }
        for slot in &mut self.labels {
            if *slot == 0 {
                *slot = label_id;
                return true;
            }
        }
        false // No empty slot
    }

    /// Remove a label. Returns true if the label was found and removed.
    pub fn remove_label(&mut self, label_id: u8) -> bool {
        for slot in &mut self.labels {
            if *slot == label_id {
                *slot = 0;
                return true;
            }
        }
        false
    }

    /// Get all non-zero labels as a Vec.
    pub fn get_labels(&self) -> Vec<u8> {
        self.labels.iter().copied().filter(|&l| l != 0).collect()
    }

    /// Serialize to bytes (little-endian, 32 bytes).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        // Links (16 bytes)
        bytes[0..4].copy_from_slice(&self.id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.first_rel_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.first_prop_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.labels);
        // Bio Metrics (12 bytes)
        bytes[16..20].copy_from_slice(&self.access_count.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.created_at.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.last_accessed_at.to_le_bytes());
        // Tuning (1 byte)
        bytes[28] = self.significance;
        bytes
    }

    /// Deserialize from bytes (little-endian, 32 bytes).
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self {
            id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            first_rel_id: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            first_prop_id: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            labels: [bytes[12], bytes[13], bytes[14], bytes[15]],
            access_count: u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            created_at: u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            last_accessed_at: u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            significance: bytes[28],
        }
    }
}

impl Default for NodeRecord {
    fn default() -> Self {
        Self::empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Record Trait
// ─────────────────────────────────────────────────────────────────────────────

impl crate::store::Record for NodeRecord {
    const SIZE: usize = 29;

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 29];
        arr.copy_from_slice(&bytes[..29]);
        Self::from_bytes(arr)
    }

    fn is_deleted(&self) -> bool {
        self.id == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_size() {
        // Serialized size (manual layout) — not the same as mem::size_of
        // because Rust adds padding for alignment
        assert_eq!(NodeRecord::SIZE, 29);
    }

    #[test]
    fn node_creation() {
        let node = NodeRecord::new(1, &[1, 2], 1000);
        assert_eq!(node.id, 1);
        assert_eq!(node.label_count(), 2);
        assert!(node.has_label(1));
        assert!(node.has_label(2));
        assert!(!node.has_label(3));
        // Bio defaults
        assert_eq!(node.access_count, 1);
        assert_eq!(node.created_at, 1000);
        assert_eq!(node.last_accessed_at, 1000);
        assert_eq!(node.significance, 128);
    }

    #[test]
    fn node_deleted() {
        let node = NodeRecord::empty();
        assert!(node.is_deleted());

        let node = NodeRecord::new(1, &[], 1000);
        assert!(!node.is_deleted());
    }

    #[test]
    fn add_remove_label() {
        let mut node = NodeRecord::new(1, &[], 1000);

        assert!(node.add_label(1));
        assert!(node.has_label(1));
        assert_eq!(node.label_count(), 1);

        assert!(node.add_label(2));
        assert!(node.add_label(3));
        assert!(node.add_label(4));
        assert!(!node.add_label(5)); // Full
        assert_eq!(node.label_count(), 4);

        assert!(node.remove_label(2));
        assert!(!node.has_label(2));
        assert_eq!(node.label_count(), 3);
    }

    #[test]
    fn node_serialization() {
        let mut node = NodeRecord::new(42, &[1, 5], 1000);
        node.first_rel_id = 100;
        node.first_prop_id = 200;
        node.access_count = 50;
        node.significance = 200;

        let restored = NodeRecord::from_bytes(node.to_bytes());

        assert_eq!(node.id, restored.id);
        assert_eq!(node.first_rel_id, restored.first_rel_id);
        assert_eq!(node.first_prop_id, restored.first_prop_id);
        assert_eq!(node.labels, restored.labels);
        assert_eq!(restored.access_count, 50);
        assert_eq!(restored.created_at, 1000);
        assert_eq!(restored.last_accessed_at, 1000);
        assert_eq!(restored.significance, 200);
    }
}
