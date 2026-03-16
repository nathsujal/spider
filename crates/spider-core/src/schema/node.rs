//! On-disk representation of a graph node (29 bytes, little-endian).
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ──────────────────────────────────────────────
//!  0       4    id                (u32; 0 = deleted tombstone)
//!  4       4    first_edge_id     (u32; 0 = no edges)
//!  8       4    first_prop_id     (u32; 0 = no properties)
//! 12       4    labels            ([u8; 4]; 0 = empty slot)
//! 16       4    access_count      (u32)
//! 20       4    created_at        (u32 Unix seconds)
//! 24       4    last_accessed_at  (u32 Unix seconds)
//! 28       1    significance      (u8; default 128)
//! ──────  ────
//! Total   29
//! ```

use crate::store::record::{self, Record};

// Compile-time Send + Sync check — zero runtime cost.
// Fails to compile if Node ever gains a non-Send/Sync field.
fn _assert_node_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Node>();
    assert_sync::<Node>();
}

// --- LabelId ---

/// A non-zero label token ID.
///
/// `0` is the empty-slot sentinel on disk and cannot be constructed through
/// the public API. [`LabelId::new_unchecked`] bypasses the check for the
/// storage layer, which only reads bytes it previously validated on write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelId(u8);

impl LabelId {
    /// Returns `Err` if `id == 0`.
    #[inline]
    pub fn new(id: u8) -> Result<Self, NodeError> {
        if id == 0 {
            return Err(NodeError::InvalidLabelId);
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

impl std::fmt::Display for LabelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LabelId({})", self.0)
    }
}

// --- Errors ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeError {
    /// `id == 0` is reserved for the deleted-slot tombstone.
    InvalidNodeId(u32),
    /// More than [`MAX_LABELS`] labels were supplied.
    TooManyLabels { given: usize, max: usize },
    /// A raw label value of `0` was given to [`LabelId::new`].
    InvalidLabelId,
    /// Label is already present on this node.
    LabelAlreadyExists(LabelId),
    /// Label is not present on this node.
    LabelNotFound(LabelId),
    /// All [`MAX_LABELS`] label slots are occupied.
    LabelSlotsFull,
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNodeId(id) =>
                write!(f, "node id {id} is invalid (0 is the deleted-slot tombstone)"),
            Self::TooManyLabels { given, max } =>
                write!(f, "too many labels: given {given}, max is {max}"),
            Self::InvalidLabelId =>
                write!(f, "label id 0 is invalid (reserved as the empty-slot sentinel)"),
            Self::LabelAlreadyExists(id) =>
                write!(f, "label {id} is already present on this node"),
            Self::LabelNotFound(id) =>
                write!(f, "label {id} is not present on this node"),
            Self::LabelSlotsFull =>
                write!(f, "all {MAX_LABELS} label slots are occupied"),
        }
    }
}

impl std::error::Error for NodeError {}

// --- Node ---

/// Maximum labels per node.
pub const MAX_LABELS: usize = 4;

/// Fixed-size graph node record — 29 bytes on disk.
///
/// `id == 0` is the deleted-slot tombstone. All non-zero bytes in `labels`
/// are valid [`LabelId`]s; zero bytes are empty slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Node {
    pub id: u32,
    pub first_edge_id: u32,
    pub first_prop_id: u32,
    // Raw bytes kept for serialization; public API always uses LabelId.
    pub(crate) labels: [u8; 4],
    pub access_count: u32,
    pub created_at: u32,
    pub last_accessed_at: u32,
    pub significance: u8,
}

impl Node {
    pub const SIZE: usize = 29;
    pub const DEFAULT_SIGNIFICANCE: u8 = 128;

    // --- Constructors ---

    /// Create a new live node.
    ///
    /// # Errors
    /// - [`NodeError::InvalidNodeId`] — `id == 0`
    /// - [`NodeError::TooManyLabels`] — `labels.len() > MAX_LABELS`
    pub fn new(
        id: u32,
        labels: &[LabelId],
        now: u32,
        significance: Option<u8>,
    ) -> Result<Self, NodeError> {
        if id == 0 {
            return Err(NodeError::InvalidNodeId(0));
        }
        if labels.len() > MAX_LABELS {
            return Err(NodeError::TooManyLabels { given: labels.len(), max: MAX_LABELS });
        }

        let mut label_arr = [0u8; 4];
        for (slot, label) in label_arr.iter_mut().zip(labels.iter()) {
            *slot = label.get();
        }

        Ok(Self {
            id,
            first_edge_id: 0,
            first_prop_id: 0,
            labels: label_arr,
            access_count: 1,
            created_at: now,
            last_accessed_at: now,
            significance: significance.unwrap_or(Self::DEFAULT_SIGNIFICANCE),
        })
    }

    /// The deleted-slot tombstone value (`id == 0`).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            id: 0,
            first_edge_id: 0,
            first_prop_id: 0,
            labels: [0u8; 4],
            access_count: 0,
            created_at: 0,
            last_accessed_at: 0,
            significance: 0,
        }
    }

    // --- State queries ---

    #[inline] pub const fn is_deleted(&self)    -> bool { self.id == 0 }
    #[inline] pub const fn has_edges(&self)      -> bool { self.first_edge_id != 0 }
    #[inline] pub const fn has_properties(&self) -> bool { self.first_prop_id != 0 }

    /// Number of assigned labels (`0..=4`).
    #[inline]
    pub fn label_count(&self) -> usize {
        self.labels.iter().filter(|&&l| l != 0).count()
    }

    /// `true` if `label` is assigned to this node.
    #[inline]
    pub fn has_label(&self, label: LabelId) -> bool {
        self.labels.contains(&label.get())
    }

    /// All assigned labels as `[Option<LabelId>; 4]` — no allocation.
    #[inline]
    pub fn labels(&self) -> [Option<LabelId>; MAX_LABELS] {
        let mut out = [None; MAX_LABELS];
        for (out_slot, &raw) in out.iter_mut().zip(self.labels.iter()) {
            if raw != 0 {
                *out_slot = Some(LabelId::new_unchecked(raw));
            }
        }
        out
    }

    // --- Label mutation ---

    /// Add a label.
    ///
    /// # Errors
    /// - [`NodeError::LabelAlreadyExists`]
    /// - [`NodeError::LabelSlotsFull`]
    pub fn add_label(&mut self, label: LabelId) -> Result<(), NodeError> {
        let raw = label.get();
        let mut free_slot: Option<usize> = None;

        for (i, &slot) in self.labels.iter().enumerate() {
            if slot == raw {
                return Err(NodeError::LabelAlreadyExists(label));
            }
            if slot == 0 && free_slot.is_none() {
                free_slot = Some(i);
            }
        }

        match free_slot {
            Some(i) => { self.labels[i] = raw; Ok(()) }
            None    => Err(NodeError::LabelSlotsFull),
        }
    }

    /// Remove a label.
    ///
    /// # Errors
    /// - [`NodeError::LabelNotFound`]
    pub fn remove_label(&mut self, label: LabelId) -> Result<(), NodeError> {
        for slot in &mut self.labels {
            if *slot == label.get() {
                *slot = 0;
                return Ok(());
            }
        }
        Err(NodeError::LabelNotFound(label))
    }

    // --- Serialization ---

    /// Serialize to 29 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.first_edge_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.first_prop_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.labels);
        buf[16..20].copy_from_slice(&self.access_count.to_le_bytes());
        buf[20..24].copy_from_slice(&self.created_at.to_le_bytes());
        buf[24..28].copy_from_slice(&self.last_accessed_at.to_le_bytes());
        buf[28] = self.significance;
        buf
    }

    /// Deserialize from 29 bytes, little-endian.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self {
            id:               u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            first_edge_id:    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            first_prop_id:    u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            labels:           bytes[12..16].try_into().unwrap(),
            access_count:     u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            created_at:       u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            last_accessed_at: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            significance:     bytes[28],
        }
    }
}

// --- Record impl ---

// Declares Node as a permitted storage type (sealed trait requirement).
impl record::private::Sealed for Node {}

impl Record for Node {
    const SIZE: usize = Node::SIZE;

    /// Zero-alloc: lives entirely on the stack.
    type Bytes = [u8; Node::SIZE];

    #[inline]
    fn to_bytes(&self) -> [u8; Node::SIZE] {
        self.to_bytes()
    }

    /// # Panics
    /// The store layer guarantees exactly `SIZE` bytes; a wrong-size input
    /// means on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    fn from_bytes(bytes: [u8; Node::SIZE]) -> Self {
        Node::from_bytes(bytes)
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

    const NOW: u32 = 1_700_000_000;

    fn lid(id: u8) -> LabelId {
        LabelId::new(id).expect("test passed label id 0")
    }

    // --- LabelId ---

    #[test]
    fn label_id_rejects_zero() {
        assert_eq!(LabelId::new(0), Err(NodeError::InvalidLabelId));
    }

    #[test]
    fn label_id_accepts_nonzero() {
        assert_eq!(LabelId::new(42).unwrap().get(), 42);
    }

    #[test]
    fn label_id_unchecked_preserves_value() {
        assert_eq!(LabelId::new_unchecked(7).get(), 7);
    }

    // --- Node::new ---

    #[test]
    fn new_valid_node() {
        let node = Node::new(1, &[lid(1), lid(2)], NOW, None).unwrap();
        assert_eq!(node.id, 1);
        assert_eq!(node.labels, [1, 2, 0, 0]);
        assert_eq!(node.significance, Node::DEFAULT_SIGNIFICANCE);
        assert_eq!(node.created_at, NOW);
        assert_eq!(node.last_accessed_at, NOW);
        assert_eq!(node.access_count, 1);
    }

    #[test]
    fn new_custom_significance() {
        assert_eq!(Node::new(1, &[lid(1)], NOW, Some(200)).unwrap().significance, 200);
    }

    #[test]
    fn new_no_labels() {
        let node = Node::new(1, &[], NOW, None).unwrap();
        assert_eq!(node.labels, [0u8; 4]);
        assert_eq!(node.label_count(), 0);
    }

    #[test]
    fn new_max_labels() {
        let node = Node::new(1, &[lid(1), lid(2), lid(3), lid(4)], NOW, None).unwrap();
        assert_eq!(node.labels, [1, 2, 3, 4]);
        assert_eq!(node.label_count(), 4);
    }

    #[test]
    fn new_rejects_id_zero() {
        assert_eq!(
            Node::new(0, &[], NOW, None).unwrap_err(),
            NodeError::InvalidNodeId(0)
        );
    }

    #[test]
    fn new_rejects_too_many_labels() {
        let labels = [lid(1), lid(2), lid(3), lid(4), lid(5)];
        assert_eq!(
            Node::new(1, &labels, NOW, None).unwrap_err(),
            NodeError::TooManyLabels { given: 5, max: 4 }
        );
    }

    // --- Node::empty ---

    #[test]
    fn empty_is_deleted() {
        let node = Node::empty();
        assert!(node.is_deleted());
        assert_eq!(node.id, 0);
        assert_eq!(node.label_count(), 0);
    }

    // --- has_label ---

    #[test]
    fn has_label_present_and_absent() {
        let node = Node::new(1, &[lid(3), lid(7)], NOW, None).unwrap();
        assert!(node.has_label(lid(3)));
        assert!(node.has_label(lid(7)));
        assert!(!node.has_label(lid(5)));
    }

    // --- labels() accessor ---

    #[test]
    fn labels_returns_typed_view() {
        let node = Node::new(1, &[lid(2), lid(5)], NOW, None).unwrap();
        let labels = node.labels();
        assert_eq!(labels[0], Some(lid(2)));
        assert_eq!(labels[1], Some(lid(5)));
        assert_eq!(labels[2], None);
        assert_eq!(labels[3], None);
    }

    #[test]
    fn labels_empty_node_all_none() {
        assert_eq!(Node::empty().labels(), [None; 4]);
    }

    // --- add_label ---

    #[test]
    fn add_label_success() {
        let mut node = Node::new(1, &[], NOW, None).unwrap();
        node.add_label(lid(5)).unwrap();
        assert!(node.has_label(lid(5)));
        assert_eq!(node.label_count(), 1);
    }

    #[test]
    fn add_label_fills_all_slots() {
        let mut node = Node::new(1, &[], NOW, None).unwrap();
        for i in 1u8..=4 { node.add_label(lid(i)).unwrap(); }
        assert_eq!(node.label_count(), 4);
    }

    #[test]
    fn add_label_rejects_duplicate() {
        let mut node = Node::new(1, &[lid(3)], NOW, None).unwrap();
        assert_eq!(node.add_label(lid(3)).unwrap_err(), NodeError::LabelAlreadyExists(lid(3)));
    }

    #[test]
    fn add_label_rejects_when_full() {
        let mut node = Node::new(1, &[lid(1), lid(2), lid(3), lid(4)], NOW, None).unwrap();
        assert_eq!(node.add_label(lid(5)).unwrap_err(), NodeError::LabelSlotsFull);
    }

    // --- remove_label ---

    #[test]
    fn remove_label_success() {
        let mut node = Node::new(1, &[lid(1), lid(2), lid(3)], NOW, None).unwrap();
        node.remove_label(lid(2)).unwrap();
        assert!(!node.has_label(lid(2)));
        assert_eq!(node.label_count(), 2);
    }

    #[test]
    fn remove_label_not_found() {
        let mut node = Node::new(1, &[lid(1)], NOW, None).unwrap();
        assert_eq!(node.remove_label(lid(9)).unwrap_err(), NodeError::LabelNotFound(lid(9)));
    }

    #[test]
    fn remove_then_readd_label() {
        let mut node = Node::new(1, &[lid(1), lid(2), lid(3), lid(4)], NOW, None).unwrap();
        node.remove_label(lid(2)).unwrap();
        assert_eq!(node.label_count(), 3);
        node.add_label(lid(2)).unwrap();
        assert_eq!(node.label_count(), 4);
        assert!(node.has_label(lid(2)));
    }

    // --- Serialization ---

    #[test]
    fn round_trip_preserves_all_fields() {
        let original = Node::new(42, &[lid(1), lid(3)], NOW, Some(200)).unwrap();
        let restored = Node::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }

    #[test]
    fn round_trip_empty_node() {
        let original = Node::empty();
        assert_eq!(original, Node::from_bytes(original.to_bytes()));
    }

    #[test]
    fn round_trip_all_field_values() {
        let mut node = Node::new(99, &[lid(1)], NOW, Some(255)).unwrap();
        node.first_edge_id = 5;
        node.first_prop_id = 10;
        node.access_count  = 1000;
        let restored = Node::from_bytes(node.to_bytes());
        assert_eq!(restored.id,            99);
        assert_eq!(restored.first_edge_id, 5);
        assert_eq!(restored.first_prop_id, 10);
        assert_eq!(restored.access_count,  1000);
        assert_eq!(restored.significance,  255);
    }

    #[test]
    fn serialization_is_little_endian() {
        let node  = Node::new(0x01020304, &[], NOW, None).unwrap();
        let bytes = node.to_bytes();
        assert_eq!(bytes[0], 0x04);
        assert_eq!(bytes[1], 0x03);
        assert_eq!(bytes[2], 0x02);
        assert_eq!(bytes[3], 0x01);
    }

    #[test]
    fn size_constant_matches_serialized_size() {
        assert_eq!(Node::empty().to_bytes().len(), Node::SIZE);
    }

    // --- Record trait ---

    #[test]
    fn record_round_trip() {
        let node     = Node::new(5, &[lid(2)], NOW, Some(64)).unwrap();
        let bytes    = Record::to_bytes(&node);    // [u8; 29] — no allocation
        let restored: Node = Record::from_bytes(bytes);
        assert_eq!(node, restored);
    }

    #[test]
    fn record_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Node>();
        assert_sync::<Node>();
    }

    // --- State queries ---

    #[test]
    fn has_edges_and_properties() {
        let mut node = Node::new(1, &[], NOW, None).unwrap();
        assert!(!node.has_edges());
        assert!(!node.has_properties());
        node.first_edge_id = 3;
        node.first_prop_id = 7;
        assert!(node.has_edges());
        assert!(node.has_properties());
    }

    #[test]
    fn label_count_after_mutations() {
        let mut node = Node::new(1, &[lid(1), lid(2)], NOW, None).unwrap();
        assert_eq!(node.label_count(), 2);
        node.add_label(lid(3)).unwrap();
        assert_eq!(node.label_count(), 3);
        node.remove_label(lid(1)).unwrap();
        assert_eq!(node.label_count(), 2);
    }
}