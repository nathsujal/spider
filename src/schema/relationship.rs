//! # Relationship Storage
//!
//! Relationships are edges connecting two nodes. Each relationship:
//! - Links a source node to a target node
//! - Has a type (e.g., KNOWS, MENTIONS, CONTAINS)
//! - Is part of TWO doubly-linked lists (one per endpoint)
//!
//! ## Doubly-Linked Chains
//!
//! Each relationship appears in two chains:
//! 1. **Source chain**: All relationships where this node is the source
//! 2. **Target chain**: All relationships where this node is the target
//!
//! This enables O(1) traversal in both directions (outgoing/incoming).
//!
//! ## Storage Layout (33 bytes)
//!
//! ```text
//! RelRecord
//! ├── id: u32              (0 = deleted)
//! ├── source_id: u32       (start node)
//! ├── target_id: u32       (end node)
//! ├── rel_type_id: u8      (relationship type token)
//! ├── prev_rel_source: u32 (source's chain - previous)
//! ├── next_rel_source: u32 (source's chain - next)
//! ├── prev_rel_target: u32 (target's chain - previous)
//! ├── next_rel_target: u32 (target's chain - next)
//! └── first_prop_id: u32   (0 = no properties)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use spider::schema::RelRecord;
//!
//! // Alice (node 1) KNOWS Bob (node 2)
//! let rel = RelRecord::new(1, 1, 2, 1); // rel_type_id 1 = KNOWS
//! assert_eq!(rel.source_id, 1);
//! assert_eq!(rel.target_id, 2);
//! ```

/// A relationship between two nodes (33 bytes serialized).
///
/// ID 0 indicates a deleted/unused slot.
/// Relationship types use token IDs from [`TokenStore`](super::TokenStore).
#[derive(Debug, Clone, Copy)]
pub struct RelRecord {
    /// Relationship ID. 0 = deleted/unused slot.
    pub id: u32,
    /// Source (start) node ID.
    pub source_id: u32,
    /// Target (end) node ID.
    pub target_id: u32,
    /// Relationship type token ID (max 256 types).
    pub rel_type_id: u8,
    /// Previous relationship in source node's chain. 0 = head.
    pub prev_rel_source: u32,
    /// Next relationship in source node's chain. 0 = tail.
    pub next_rel_source: u32,
    /// Previous relationship in target node's chain. 0 = head.
    pub prev_rel_target: u32,
    /// Next relationship in target node's chain. 0 = tail.
    pub next_rel_target: u32,
    /// First property record ID. 0 = no properties.
    pub first_prop_id: u32,
}

impl RelRecord {
    /// Serialized size in bytes.
    pub const SIZE: usize = 33;

    /// Create a new relationship.
    pub fn new(id: u32, source_id: u32, target_id: u32, rel_type_id: u8) -> Self {
        Self {
            id,
            source_id,
            target_id,
            rel_type_id,
            prev_rel_source: 0,
            next_rel_source: 0,
            prev_rel_target: 0,
            next_rel_target: 0,
            first_prop_id: 0,
        }
    }

    /// Create an empty/deleted relationship record.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            id: 0,
            source_id: 0,
            target_id: 0,
            rel_type_id: 0,
            prev_rel_source: 0,
            next_rel_source: 0,
            prev_rel_target: 0,
            next_rel_target: 0,
            first_prop_id: 0,
        }
    }

    /// Returns true if this relationship slot is deleted/unused.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.id == 0
    }

    /// Returns true if this relationship has properties.
    #[inline]
    pub fn has_properties(&self) -> bool {
        self.first_prop_id != 0
    }

    /// Returns true if this is the first relationship in source's chain.
    #[inline]
    pub fn is_first_for_source(&self) -> bool {
        self.prev_rel_source == 0
    }

    /// Returns true if this is the last relationship in source's chain.
    #[inline]
    pub fn is_last_for_source(&self) -> bool {
        self.next_rel_source == 0
    }

    /// Returns true if this is the first relationship in target's chain.
    #[inline]
    pub fn is_first_for_target(&self) -> bool {
        self.prev_rel_target == 0
    }

    /// Returns true if this is the last relationship in target's chain.
    #[inline]
    pub fn is_last_for_target(&self) -> bool {
        self.next_rel_target == 0
    }

    /// Serialize to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.source_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.target_id.to_le_bytes());
        bytes[12] = self.rel_type_id;
        bytes[13..17].copy_from_slice(&self.prev_rel_source.to_le_bytes());
        bytes[17..21].copy_from_slice(&self.next_rel_source.to_le_bytes());
        bytes[21..25].copy_from_slice(&self.prev_rel_target.to_le_bytes());
        bytes[25..29].copy_from_slice(&self.next_rel_target.to_le_bytes());
        bytes[29..33].copy_from_slice(&self.first_prop_id.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes (little-endian).
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self {
            id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            source_id: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            target_id: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            rel_type_id: bytes[12],
            prev_rel_source: u32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]),
            next_rel_source: u32::from_le_bytes([bytes[17], bytes[18], bytes[19], bytes[20]]),
            prev_rel_target: u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]),
            next_rel_target: u32::from_le_bytes([bytes[25], bytes[26], bytes[27], bytes[28]]),
            first_prop_id: u32::from_le_bytes([bytes[29], bytes[30], bytes[31], bytes[32]]),
        }
    }
}

impl Default for RelRecord {
    fn default() -> Self {
        Self::empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Record Trait
// ─────────────────────────────────────────────────────────────────────────────

impl crate::store::Record for RelRecord {
    const SIZE: usize = 33;

    fn to_bytes(&self) -> Vec<u8> {
        self.to_bytes().to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 33];
        arr.copy_from_slice(&bytes[..33]);
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
    fn rel_serialized_size() {
        let rel = RelRecord::empty();
        assert_eq!(rel.to_bytes().len(), RelRecord::SIZE);
    }

    #[test]
    fn rel_creation() {
        let rel = RelRecord::new(1, 10, 20, 5);
        assert_eq!(rel.id, 1);
        assert_eq!(rel.source_id, 10);
        assert_eq!(rel.target_id, 20);
        assert_eq!(rel.rel_type_id, 5);
        assert!(!rel.is_deleted());
    }

    #[test]
    fn rel_deleted() {
        let rel = RelRecord::empty();
        assert!(rel.is_deleted());
    }

    #[test]
    fn rel_chain_checks() {
        let rel = RelRecord::new(1, 10, 20, 1);
        assert!(rel.is_first_for_source());
        assert!(rel.is_last_for_source());
        assert!(rel.is_first_for_target());
        assert!(rel.is_last_for_target());
    }

    #[test]
    fn rel_serialization() {
        let mut rel = RelRecord::new(42, 100, 200, 7);
        rel.prev_rel_source = 10;
        rel.next_rel_source = 20;
        rel.prev_rel_target = 30;
        rel.next_rel_target = 40;
        rel.first_prop_id = 500;

        let restored = RelRecord::from_bytes(rel.to_bytes());

        assert_eq!(rel.id, restored.id);
        assert_eq!(rel.source_id, restored.source_id);
        assert_eq!(rel.target_id, restored.target_id);
        assert_eq!(rel.rel_type_id, restored.rel_type_id);
        assert_eq!(rel.prev_rel_source, restored.prev_rel_source);
        assert_eq!(rel.next_rel_source, restored.next_rel_source);
        assert_eq!(rel.prev_rel_target, restored.prev_rel_target);
        assert_eq!(rel.next_rel_target, restored.next_rel_target);
        assert_eq!(rel.first_prop_id, restored.first_prop_id);
    }
}
