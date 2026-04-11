//! On-disk representation of a directed graph edge (33 bytes, little-endian).
//!
//! Each edge participates in two doubly-linked lists simultaneously — one
//! anchored at the source node, one at the target — enabling O(1) insertion,
//! deletion, and bidirectional traversal without a separate index.
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ──────────────────────────────────────────────────
//!  0       4    id                 (u32; 0 = deleted tombstone)
//!  4       4    source_id          (u32)
//!  8       4    target_id          (u32)
//! 12       1    type_id            (u8; 0 = empty sentinel)
//! 13       4    prev_edge_source   (u32; 0 = head of source chain)
//! 17       4    next_edge_source   (u32; 0 = tail of source chain)
//! 21       4    prev_edge_target   (u32; 0 = head of target chain)
//! 25       4    next_edge_target   (u32; 0 = tail of target chain)
//! 29       4    first_prop_id      (u32; 0 = no properties)
//! ──────  ────
//! Total   33
//! ```

use crate::store::record::{self, Record};

// Compile-time Send + Sync check — zero runtime cost.
// Fails to compile if Edge ever gains a non-Send/Sync field.
fn _assert_edge_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Edge>();
    assert_sync::<Edge>();
}

// --- EdgeTypeId ---

/// A non-zero edge type token ID.
///
/// `0` is the empty sentinel and cannot be constructed through the public API.
/// [`EdgeTypeId::new_unchecked`] bypasses the check for the storage layer,
/// which only reads bytes it previously validated on write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeTypeId(u8);

impl EdgeTypeId {
    /// Returns `Err` if `id == 0`.
    #[inline]
    pub fn new(id: u8) -> Result<Self, EdgeError> {
        if id == 0 {
            return Err(EdgeError::InvalidTypeId);
        }
        Ok(Self(id))
    }

    /// Construct without checking for zero.
    ///
    /// Caller must guarantee `id != 0`.
    #[inline]
    pub(crate) const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    /// The underlying raw value.
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for EdgeTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EdgeTypeId({})", self.0)
    }
}

// --- Errors ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeError {
    /// `id == 0` is reserved for the deleted-slot tombstone.
    InvalidId(u32),
    /// `source_id` or `target_id` is `0`.
    InvalidNodeId(u32),
    /// `source_id == target_id`. Self-loops are not permitted.
    SelfLoop(u32),
    /// A raw type value of `0` was given to [`EdgeTypeId::new`].
    InvalidTypeId,
}

impl std::fmt::Display for EdgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId(id) =>
                write!(f, "edge id {id} is invalid (0 is the deleted-slot tombstone)"),
            Self::InvalidNodeId(id) =>
                write!(f, "node id {id} is invalid (0 is the deleted-slot tombstone)"),
            Self::SelfLoop(id) =>
                write!(f, "self-loops are not permitted (source == target == {id})"),
            Self::InvalidTypeId =>
                write!(f, "edge type id 0 is invalid (reserved as the empty sentinel)"),
        }
    }
}

impl std::error::Error for EdgeError {}

// --- Edge ---

/// A directed graph edge — 33 bytes on disk.
///
/// `id == 0` is the deleted-slot tombstone. Source and target node IDs are
/// always nonzero for live edges; self-loops are rejected at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Edge {
    pub id: u32,
    pub source_id: u32,
    pub target_id: u32,
    // Raw byte kept for serialization; public API always uses EdgeTypeId.
    pub(crate) type_id: u8,
    pub prev_edge_source: u32,
    pub next_edge_source: u32,
    pub prev_edge_target: u32,
    pub next_edge_target: u32,
    pub first_prop_id: u32,
}

impl Edge {
    pub const SIZE: usize = 33;

    // --- Constructors ---

    /// Create a new live edge.
    ///
    /// # Errors
    /// - [`EdgeError::InvalidId`]      — `id == 0`
    /// - [`EdgeError::InvalidNodeId`]  — `source_id == 0` or `target_id == 0`
    /// - [`EdgeError::SelfLoop`]       — `source_id == target_id`
    pub fn new(
        id: u32,
        source_id: u32,
        target_id: u32,
        type_id: EdgeTypeId,
    ) -> Result<Self, EdgeError> {
        if id == 0 {
            return Err(EdgeError::InvalidId(0));
        }
        if source_id == 0 {
            return Err(EdgeError::InvalidNodeId(0));
        }
        if target_id == 0 {
            return Err(EdgeError::InvalidNodeId(0));
        }
        if source_id == target_id {
            return Err(EdgeError::SelfLoop(source_id));
        }

        Ok(Self {
            id,
            source_id,
            target_id,
            type_id: type_id.get(),
            prev_edge_source: 0,
            next_edge_source: 0,
            prev_edge_target: 0,
            next_edge_target: 0,
            first_prop_id: 0,
        })
    }

    /// The deleted-slot tombstone value (`id == 0`).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            id: 0,
            source_id: 0,
            target_id: 0,
            type_id: 0,
            prev_edge_source: 0,
            next_edge_source: 0,
            prev_edge_target: 0,
            next_edge_target: 0,
            first_prop_id: 0,
        }
    }

    // --- State queries ---

    #[inline] pub const fn is_deleted(&self)    -> bool { self.id == 0 }
    #[inline] pub const fn has_properties(&self) -> bool { self.first_prop_id != 0 }

    /// The edge type as a typed [`EdgeTypeId`].
    ///
    /// Returns `None` for the tombstone (type_id == 0).
    #[inline]
    pub fn edge_type(&self) -> Option<EdgeTypeId> {
        if self.type_id == 0 { None } else { Some(EdgeTypeId::new_unchecked(self.type_id)) }
    }

    /// `true` if this is the head of the source node's chain.
    #[inline] pub const fn is_first_for_source(&self) -> bool { self.prev_edge_source == 0 }
    /// `true` if this is the tail of the source node's chain.
    #[inline] pub const fn is_last_for_source(&self)  -> bool { self.next_edge_source == 0 }
    /// `true` if this is the head of the target node's chain.
    #[inline] pub const fn is_first_for_target(&self) -> bool { self.prev_edge_target == 0 }
    /// `true` if this is the tail of the target node's chain.
    #[inline] pub const fn is_last_for_target(&self)  -> bool { self.next_edge_target == 0 }

    /// `true` if this is the only edge in both the source and target chains.
    #[inline]
    pub const fn is_only_edge(&self) -> bool {
        self.is_first_for_source()
            && self.is_last_for_source()
            && self.is_first_for_target()
            && self.is_last_for_target()
    }

    // --- Serialization ---

    /// Serialize to 33 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.source_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.target_id.to_le_bytes());
        buf[12] = self.type_id;
        buf[13..17].copy_from_slice(&self.prev_edge_source.to_le_bytes());
        buf[17..21].copy_from_slice(&self.next_edge_source.to_le_bytes());
        buf[21..25].copy_from_slice(&self.prev_edge_target.to_le_bytes());
        buf[25..29].copy_from_slice(&self.next_edge_target.to_le_bytes());
        buf[29..33].copy_from_slice(&self.first_prop_id.to_le_bytes());
        buf
    }

    /// Deserialize from 33 bytes, little-endian.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self {
            id:               u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            source_id:        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            target_id:        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            type_id:          bytes[12],
            prev_edge_source: u32::from_le_bytes(bytes[13..17].try_into().unwrap()),
            next_edge_source: u32::from_le_bytes(bytes[17..21].try_into().unwrap()),
            prev_edge_target: u32::from_le_bytes(bytes[21..25].try_into().unwrap()),
            next_edge_target: u32::from_le_bytes(bytes[25..29].try_into().unwrap()),
            first_prop_id:    u32::from_le_bytes(bytes[29..33].try_into().unwrap()),
        }
    }
}

// --- Record impl ---

// Declares Edge as a permitted storage type (sealed trait requirement).
impl record::private::Sealed for Edge {}

impl Record for Edge {
    const SIZE: usize = Edge::SIZE;

    /// Zero-alloc: lives entirely on the stack.
    type Bytes = [u8; Edge::SIZE];

    #[inline]
    fn to_bytes(&self) -> [u8; Edge::SIZE] {
        self.to_bytes()
    }

    /// # Panics
    /// The store layer guarantees exactly `SIZE` bytes; a wrong-size input
    /// means on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    fn from_bytes(bytes: [u8; Edge::SIZE]) -> Self {
        Edge::from_bytes(bytes)
    }

    #[inline]
    fn from_raw(bytes: &[u8]) -> Self {
        Edge::from_bytes(bytes.try_into().unwrap())
    }

    #[inline]
    fn is_deleted(&self) -> bool {
        self.is_deleted()
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(id: u8) -> EdgeTypeId {
        EdgeTypeId::new(id).expect("test passed type id 0")
    }

    // --- EdgeTypeId ---

    #[test]
    fn type_id_rejects_zero() {
        assert_eq!(EdgeTypeId::new(0), Err(EdgeError::InvalidTypeId));
    }

    #[test]
    fn type_id_accepts_nonzero() {
        assert_eq!(EdgeTypeId::new(5).unwrap().get(), 5);
    }

    #[test]
    fn type_id_unchecked_preserves_value() {
        assert_eq!(EdgeTypeId::new_unchecked(7).get(), 7);
    }

    #[test]
    fn type_id_max_value() {
        assert_eq!(EdgeTypeId::new(255).unwrap().get(), 255);
    }

    // --- Edge::new ---

    #[test]
    fn new_valid_edge() {
        let edge = Edge::new(1, 10, 20, tid(5)).unwrap();
        assert_eq!(edge.id, 1);
        assert_eq!(edge.source_id, 10);
        assert_eq!(edge.target_id, 20);
        assert_eq!(edge.type_id, 5);
        assert_eq!(edge.first_prop_id, 0);
        assert!(!edge.is_deleted());
    }

    #[test]
    fn new_chain_pointers_start_at_zero() {
        let edge = Edge::new(1, 1, 2, tid(1)).unwrap();
        assert_eq!(edge.prev_edge_source, 0);
        assert_eq!(edge.next_edge_source, 0);
        assert_eq!(edge.prev_edge_target, 0);
        assert_eq!(edge.next_edge_target, 0);
    }

    #[test]
    fn new_rejects_id_zero() {
        assert_eq!(Edge::new(0, 1, 2, tid(1)).unwrap_err(), EdgeError::InvalidId(0));
    }

    #[test]
    fn new_rejects_source_id_zero() {
        assert_eq!(Edge::new(1, 0, 2, tid(1)).unwrap_err(), EdgeError::InvalidNodeId(0));
    }

    #[test]
    fn new_rejects_target_id_zero() {
        assert_eq!(Edge::new(1, 1, 0, tid(1)).unwrap_err(), EdgeError::InvalidNodeId(0));
    }

    #[test]
    fn new_rejects_self_loop() {
        assert_eq!(Edge::new(1, 5, 5, tid(1)).unwrap_err(), EdgeError::SelfLoop(5));
    }

    // --- Edge::empty ---

    #[test]
    fn empty_is_deleted() {
        let edge = Edge::empty();
        assert!(edge.is_deleted());
        assert_eq!(edge.id, 0);
        assert_eq!(edge.source_id, 0);
        assert_eq!(edge.target_id, 0);
    }

    #[test]
    fn default_equals_empty() {
        assert_eq!(Edge::default(), Edge::empty());
    }

    // --- State queries ---

    #[test]
    fn has_properties_false_by_default() {
        assert!(!Edge::new(1, 1, 2, tid(1)).unwrap().has_properties());
    }

    #[test]
    fn has_properties_true_when_set() {
        let mut edge = Edge::new(1, 1, 2, tid(1)).unwrap();
        edge.first_prop_id = 7;
        assert!(edge.has_properties());
    }

    #[test]
    fn edge_type_returns_typed_id() {
        let edge = Edge::new(1, 1, 2, tid(3)).unwrap();
        assert_eq!(edge.edge_type(), Some(tid(3)));
    }

    #[test]
    fn edge_type_none_for_tombstone() {
        assert_eq!(Edge::empty().edge_type(), None);
    }

    #[test]
    fn chain_head_tail_on_new_edge() {
        let edge = Edge::new(1, 1, 2, tid(1)).unwrap();
        assert!(edge.is_first_for_source());
        assert!(edge.is_last_for_source());
        assert!(edge.is_first_for_target());
        assert!(edge.is_last_for_target());
        assert!(edge.is_only_edge());
    }

    #[test]
    fn chain_middle_edge_not_head_or_tail() {
        let mut edge = Edge::new(3, 1, 2, tid(1)).unwrap();
        edge.prev_edge_source = 1;
        edge.next_edge_source = 5;
        edge.prev_edge_target = 2;
        edge.next_edge_target = 6;

        assert!(!edge.is_first_for_source());
        assert!(!edge.is_last_for_source());
        assert!(!edge.is_first_for_target());
        assert!(!edge.is_last_for_target());
        assert!(!edge.is_only_edge());
    }

    #[test]
    fn is_only_edge_false_when_chained() {
        let mut edge = Edge::new(1, 1, 2, tid(1)).unwrap();
        edge.next_edge_source = 3;
        assert!(!edge.is_only_edge());
    }

    // --- Serialization ---

    #[test]
    fn round_trip_preserves_all_fields() {
        let mut original = Edge::new(42, 100, 200, tid(7)).unwrap();
        original.prev_edge_source = 10;
        original.next_edge_source = 20;
        original.prev_edge_target = 30;
        original.next_edge_target = 40;
        original.first_prop_id    = 500;

        assert_eq!(original, Edge::from_bytes(original.to_bytes()));
    }

    #[test]
    fn round_trip_empty_edge() {
        let original = Edge::empty();
        assert_eq!(original, Edge::from_bytes(original.to_bytes()));
    }

    #[test]
    fn round_trip_max_values() {
        let mut edge = Edge::new(u32::MAX, u32::MAX - 1, u32::MAX - 2, tid(255)).unwrap();
        edge.prev_edge_source = u32::MAX;
        edge.next_edge_source = u32::MAX;
        edge.prev_edge_target = u32::MAX;
        edge.next_edge_target = u32::MAX;
        edge.first_prop_id    = u32::MAX;

        assert_eq!(edge, Edge::from_bytes(edge.to_bytes()));
    }

    #[test]
    fn serialization_is_little_endian() {
        let edge  = Edge::new(0x01020304, 1, 2, tid(1)).unwrap();
        let bytes = edge.to_bytes();
        assert_eq!(bytes[0], 0x04);
        assert_eq!(bytes[1], 0x03);
        assert_eq!(bytes[2], 0x02);
        assert_eq!(bytes[3], 0x01);
    }

    #[test]
    fn type_id_at_correct_offset() {
        let edge  = Edge::new(1, 1, 2, tid(0xAB)).unwrap();
        let bytes = edge.to_bytes();
        assert_eq!(bytes[12], 0xAB);
    }

    #[test]
    fn size_constant_matches_serialized_size() {
        assert_eq!(Edge::empty().to_bytes().len(), Edge::SIZE);
    }

    // --- Record trait ---

    #[test]
    fn record_round_trip() {
        let edge     = Edge::new(5, 1, 2, tid(3)).unwrap();
        let bytes    = Record::to_bytes(&edge);    // [u8; 33] — no allocation
        let restored: Edge = Record::from_bytes(bytes);
        assert_eq!(edge, restored);
    }

    #[test]
    fn record_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Edge>();
        assert_sync::<Edge>();
    }

    // --- Error display ---

    #[test]
    fn error_messages_are_readable() {
        assert!(EdgeError::InvalidId(0).to_string().contains("0"));
        assert!(EdgeError::SelfLoop(5).to_string().contains("5"));
        assert!(EdgeError::InvalidTypeId.to_string().contains("0"));
    }
}