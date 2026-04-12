//! Fixed-size 128-byte records for values too large to fit inline in a
//! [`PropertyBlock`](super::property::PropertyBlock).
//!
//! - [`DynamicStringRecord`] — UTF-8 strings longer than 6 bytes
//! - [`DynamicArrayRecord`]  — arrays of any primitive type
//!
//! Both types chain via `next_block` to store arbitrarily large values across
//! multiple 128-byte blocks.
//!
//! ```text
//! DynamicStringRecord (128 bytes)
//! ──────  ────  ──────────────────────────────────────────────
//!  0       1    flags       (u8; bit 0 = in_use, bit 1 = is_start)
//!  1       3    length      (u24 LE; total bytes, start block only)
//!  4       4    next_block  (u32 LE; 0 = end of chain)
//!  8     120    data        ([u8; 120]; UTF-8 payload)
//!
//! DynamicArrayRecord (128 bytes)
//! ──────  ────  ──────────────────────────────────────────────
//!  0       1    flags        (u8; bit 0 = in_use, bit 1 = is_start)
//!  1       3    length       (u24 LE; element count, start block only)
//!  4       4    next_block   (u32 LE; 0 = end of chain)
//!  8       1    element_type (u8; PropertyType discriminant)
//!  9     119    data         ([u8; 119]; array payload)
//! ```
//!
//! The 3-byte length field can represent up to [`MAX_LENGTH`] (16,777,215).

use crate::schema::property::PropertyType;
use crate::store::record::{self, Record};

// Compile-time Send + Sync checks — zero runtime cost.
// Fails to compile if either type ever gains a non-Send/Sync field.
fn _assert_dynamic_string_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<DynamicStringRecord>();
    assert_sync::<DynamicStringRecord>();
}

fn _assert_dynamic_array_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<DynamicArrayRecord>();
    assert_sync::<DynamicArrayRecord>();
}

// --- Errors ---

/// Errors from dynamic record operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicError {
    /// Length value exceeds the 24-bit maximum (16,777,215).
    LengthOverflow { value: u32, max: u32 },
}

impl std::fmt::Display for DynamicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthOverflow { value, max } =>
                write!(f, "length {value} exceeds 24-bit maximum {max}"),
        }
    }
}

impl std::error::Error for DynamicError {}

// --- Shared constants ---

/// Maximum value representable in the 3-byte length field (2^24 - 1).
pub const MAX_LENGTH: u32 = 0x00FF_FFFF;

const FLAG_IN_USE:  u8 = 0b0000_0001;
const FLAG_IS_START: u8 = 0b0000_0010;

// --- DynamicStringRecord ---

/// A 128-byte block for storing long UTF-8 strings.
///
/// Strings longer than 6 bytes (the [`PropertyBlock`] inline limit) are split
/// into one or more of these blocks, chained via `next_block`.
///
/// Only the **start block** carries a valid `length` and has `is_start == true`.
/// Continuation blocks set only `FLAG_IN_USE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicStringRecord {
    /// Raw bit flags — use [`is_in_use`](Self::is_in_use),
    /// [`is_start`](Self::is_start), and [`delete`](Self::delete) instead.
    pub(crate) flags: u8,
    /// 3-byte LE length — use [`get_length`](Self::get_length) and
    /// [`set_length`](Self::set_length) instead.
    pub(crate) length: [u8; 3],
    /// Next block in chain. `0` = end of chain.
    pub next_block: u32,
    /// UTF-8 payload — up to [`Self::DATA_SIZE`] bytes per block.
    pub data: [u8; 120],
}

impl DynamicStringRecord {
    /// Serialized size on disk in bytes.
    pub const SIZE: usize = 128;
    /// Maximum UTF-8 payload bytes per block.
    pub const DATA_SIZE: usize = 120;

    // --- Constructors ---

    /// Create an empty (deleted) record.
    #[inline]
    pub const fn new() -> Self {
        Self { flags: 0, length: [0; 3], next_block: 0, data: [0; 120] }
    }

    /// Create the start block of a string chain.
    ///
    /// # Errors
    /// Returns [`DynamicError::LengthOverflow`] if `total_length > MAX_LENGTH`.
    pub fn new_start(data: &[u8], total_length: u32, next_block: u32) -> Result<Self, DynamicError> {
        if total_length > MAX_LENGTH {
            return Err(DynamicError::LengthOverflow { value: total_length, max: MAX_LENGTH });
        }
        let mut r = Self::new();
        r.flags = FLAG_IN_USE | FLAG_IS_START;
        r.set_length(total_length);
        r.next_block = next_block;
        let n = data.len().min(Self::DATA_SIZE);
        r.data[..n].copy_from_slice(&data[..n]);
        Ok(r)
    }

    /// Create a continuation block (not the start of a chain).
    pub fn new_continuation(data: &[u8], next_block: u32) -> Self {
        let mut r = Self::new();
        r.flags = FLAG_IN_USE;
        r.next_block = next_block;
        let n = data.len().min(Self::DATA_SIZE);
        r.data[..n].copy_from_slice(&data[..n]);
        r
    }

    // --- State queries ---

    /// `true` if this block is allocated (not deleted).
    #[inline]
    pub fn is_in_use(&self) -> bool { self.flags & FLAG_IN_USE != 0 }

    /// `true` if this is the first block of a string chain.
    #[inline]
    pub fn is_start(&self) -> bool { self.flags & FLAG_IS_START != 0 }

    /// `true` if this is the last block in the chain.
    #[inline]
    pub fn is_end(&self) -> bool { self.next_block == 0 }

    // --- Length ---

    /// Read the 3-byte LE length field as `u32`. Only valid on the start block.
    #[inline]
    pub fn get_length(&self) -> u32 {
        u32::from_le_bytes([self.length[0], self.length[1], self.length[2], 0])
    }

    /// Write a `u32` into the 3-byte LE length field.
    ///
    /// Caller must ensure `len <= MAX_LENGTH`.
    #[inline]
    pub fn set_length(&mut self, len: u32) {
        let b = len.to_le_bytes();
        self.length = [b[0], b[1], b[2]];
    }

    // --- Data access ---

    /// Payload slice capped at `expected_len` bytes.
    #[inline]
    pub fn get_data(&self, expected_len: usize) -> &[u8] {
        &self.data[..expected_len.min(Self::DATA_SIZE)]
    }

    // --- Mutation ---

    /// Mark this block as deleted by clearing all flags.
    #[inline]
    pub fn delete(&mut self) { self.flags = 0; }

    // --- Serialization ---

    /// Serialize to 128 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.flags;
        buf[1..4].copy_from_slice(&self.length);
        buf[4..8].copy_from_slice(&self.next_block.to_le_bytes());
        buf[8..128].copy_from_slice(&self.data);
        buf
    }

    /// Deserialize from 128 bytes, little-endian.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let mut data = [0u8; 120];
        data.copy_from_slice(&bytes[8..128]);
        Self {
            flags:      bytes[0],
            length:     [bytes[1], bytes[2], bytes[3]],
            next_block: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            data,
        }
    }
}

impl Default for DynamicStringRecord {
    fn default() -> Self { Self::new() }
}

// --- DynamicArrayRecord ---

/// A 128-byte block for storing arrays of primitive values.
///
/// Arrays that don't fit inline are split into one or more blocks, chained via
/// `next_block`. Only the **start block** carries a valid `length` and
/// `element_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicArrayRecord {
    /// Raw bit flags — use [`is_in_use`](Self::is_in_use),
    /// [`is_start`](Self::is_start), and [`delete`](Self::delete) instead.
    pub(crate) flags: u8,
    /// 3-byte LE element count — use [`get_length`](Self::get_length) and
    /// [`set_length`](Self::set_length) instead.
    pub(crate) length: [u8; 3],
    /// Next block in chain. `0` = end of chain.
    pub next_block: u32,
    /// Element type of the array. Only valid on the start block.
    pub element_type: PropertyType,
    /// Array payload — up to [`Self::DATA_SIZE`] bytes per block.
    pub data: [u8; 119],
}

impl DynamicArrayRecord {
    /// Serialized size on disk in bytes.
    pub const SIZE: usize = 128;
    /// Maximum array payload bytes per block.
    pub const DATA_SIZE: usize = 119;

    // --- Constructors ---

    /// Create an empty (deleted) record.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: 0,
            length: [0; 3],
            next_block: 0,
            element_type: PropertyType::Empty,
            data: [0; 119],
        }
    }

    /// Create the start block of an array chain.
    ///
    /// # Errors
    /// Returns [`DynamicError::LengthOverflow`] if `element_count > MAX_LENGTH`.
    pub fn new_start(
        element_type: PropertyType,
        element_count: u32,
        data: &[u8],
        next_block: u32,
    ) -> Result<Self, DynamicError> {
        if element_count > MAX_LENGTH {
            return Err(DynamicError::LengthOverflow { value: element_count, max: MAX_LENGTH });
        }
        let mut r = Self::new();
        r.flags = FLAG_IN_USE | FLAG_IS_START;
        r.element_type = element_type;
        r.set_length(element_count);
        r.next_block = next_block;
        let n = data.len().min(Self::DATA_SIZE);
        r.data[..n].copy_from_slice(&data[..n]);
        Ok(r)
    }

    /// Create a continuation block (not the start of a chain).
    pub fn new_continuation(data: &[u8], next_block: u32) -> Self {
        let mut r = Self::new();
        r.flags = FLAG_IN_USE;
        r.next_block = next_block;
        let n = data.len().min(Self::DATA_SIZE);
        r.data[..n].copy_from_slice(&data[..n]);
        r
    }

    // --- State queries ---

    /// `true` if this block is allocated (not deleted).
    #[inline]
    pub fn is_in_use(&self) -> bool { self.flags & FLAG_IN_USE != 0 }

    /// `true` if this is the first block of an array chain.
    #[inline]
    pub fn is_start(&self) -> bool { self.flags & FLAG_IS_START != 0 }

    /// `true` if this is the last block in the chain.
    #[inline]
    pub fn is_end(&self) -> bool { self.next_block == 0 }

    // --- Length ---

    /// Read the 3-byte LE length field as `u32`. Only valid on the start block.
    #[inline]
    pub fn get_length(&self) -> u32 {
        u32::from_le_bytes([self.length[0], self.length[1], self.length[2], 0])
    }

    /// Write a `u32` into the 3-byte LE length field.
    ///
    /// Caller must ensure `len <= MAX_LENGTH`.
    #[inline]
    pub fn set_length(&mut self, len: u32) {
        let b = len.to_le_bytes();
        self.length = [b[0], b[1], b[2]];
    }

    // --- Data access ---

    /// Payload slice capped at `expected_len` bytes.
    #[inline]
    pub fn get_data(&self, expected_len: usize) -> &[u8] {
        &self.data[..expected_len.min(Self::DATA_SIZE)]
    }

    // --- Mutation ---

    /// Mark this block as deleted by clearing all flags.
    #[inline]
    pub fn delete(&mut self) { self.flags = 0; }

    // --- Serialization ---

    /// Serialize to 128 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.flags;
        buf[1..4].copy_from_slice(&self.length);
        buf[4..8].copy_from_slice(&self.next_block.to_le_bytes());
        buf[8] = self.element_type as u8;
        buf[9..128].copy_from_slice(&self.data);
        buf
    }

    /// Deserialize from 128 bytes, little-endian.
    ///
    /// # Panics
    /// Panics if `element_type` byte is unrecognised — this indicates
    /// on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let mut data = [0u8; 119];
        data.copy_from_slice(&bytes[9..128]);
        let element_type = PropertyType::from_u8(bytes[8]);

        // PropertyType::from_u8 maps unknown discriminants to Empty. An Empty
        // element_type on a live start block means the byte is corrupt.
        // Continuation blocks legitimately have element_type = 0 (Empty),
        // so only flag the mismatch on start blocks.
        debug_assert!(
            bytes[0] & FLAG_IS_START == 0
                || element_type != PropertyType::Empty
                || bytes[8] == 0,
            "Storage corruption: unrecognised element_type byte {:#04x} in \
             DynamicArrayRecord start block. \
             Run `spider gc --verify` to assess integrity.",
            bytes[8]
        );

        Self {
            flags:        bytes[0],
            length:       [bytes[1], bytes[2], bytes[3]],
            next_block:   u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            element_type,
            data,
        }
    }
}

impl Default for DynamicArrayRecord {
    fn default() -> Self { Self::new() }
}

// --- Record impls ---

impl record::private::Sealed for DynamicStringRecord {}

impl Record for DynamicStringRecord {
    const SIZE: usize = DynamicStringRecord::SIZE;

    /// Zero-alloc: lives entirely on the stack.
    type Bytes = [u8; DynamicStringRecord::SIZE];

    #[inline]
    fn to_bytes(&self) -> [u8; DynamicStringRecord::SIZE] {
        self.to_bytes()
    }

    /// # Panics
    /// The store layer guarantees exactly `SIZE` bytes; a wrong-size input
    /// means on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    fn from_bytes(bytes: [u8; DynamicStringRecord::SIZE]) -> Self {
        DynamicStringRecord::from_bytes(bytes)
    }

    #[inline]
    fn from_raw(bytes: &[u8]) -> Self {
        DynamicStringRecord::from_bytes(bytes.try_into().unwrap())
    }

    #[inline]
    fn is_deleted(&self) -> bool { !self.is_in_use() }
}

impl record::private::Sealed for DynamicArrayRecord {}

impl Record for DynamicArrayRecord {
    const SIZE: usize = DynamicArrayRecord::SIZE;

    /// Zero-alloc: lives entirely on the stack.
    type Bytes = [u8; DynamicArrayRecord::SIZE];

    #[inline]
    fn to_bytes(&self) -> [u8; DynamicArrayRecord::SIZE] {
        self.to_bytes()
    }

    /// # Panics
    /// The store layer guarantees exactly `SIZE` bytes; a wrong-size input
    /// means on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    fn from_bytes(bytes: [u8; DynamicArrayRecord::SIZE]) -> Self {
        DynamicArrayRecord::from_bytes(bytes)
    }

    #[inline]
    fn from_raw(bytes: &[u8]) -> Self {
        DynamicArrayRecord::from_bytes(bytes.try_into().unwrap())
    }

    #[inline]
    fn is_deleted(&self) -> bool { !self.is_in_use() }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // --- DynamicStringRecord constructors ---

    #[test]
    fn string_new_is_deleted() {
        let r = DynamicStringRecord::new();
        assert!(!r.is_in_use());
        assert!(!r.is_start());
        assert!(r.is_end());
        assert_eq!(r.get_length(), 0);
    }

    #[test]
    fn string_default_equals_new() {
        assert_eq!(DynamicStringRecord::default(), DynamicStringRecord::new());
    }

    #[test]
    fn string_new_start_sets_flags_and_length() {
        let data = b"Hello, World!";
        let r = DynamicStringRecord::new_start(data, data.len() as u32, 0).unwrap();
        assert!(r.is_in_use());
        assert!(r.is_start());
        assert!(r.is_end());
        assert_eq!(r.get_length(), 13);
        assert_eq!(r.next_block, 0);
    }

    #[test]
    fn string_new_continuation_sets_flags() {
        let r = DynamicStringRecord::new_continuation(b"cont", 7);
        assert!(r.is_in_use());
        assert!(!r.is_start());
        assert!(!r.is_end());
        assert_eq!(r.next_block, 7);
    }

    #[test]
    fn string_chain_start_to_continuation() {
        let start = DynamicStringRecord::new_start(b"first", 250, 2).unwrap();
        let cont  = DynamicStringRecord::new_continuation(b"second", 0);
        assert!(start.is_start() && !start.is_end());
        assert!(!cont.is_start() && cont.is_end());
    }

    #[test]
    fn string_new_start_rejects_length_overflow() {
        assert_eq!(
            DynamicStringRecord::new_start(&[], MAX_LENGTH + 1, 0).unwrap_err(),
            DynamicError::LengthOverflow { value: MAX_LENGTH + 1, max: MAX_LENGTH }
        );
    }

    #[test]
    fn string_new_start_accepts_max_length() {
        let r = DynamicStringRecord::new_start(&[], MAX_LENGTH, 0).unwrap();
        assert_eq!(r.get_length(), MAX_LENGTH);
    }

    // --- DynamicStringRecord data ---

    #[test]
    fn string_data_truncated_to_data_size() {
        let big = vec![0xAAu8; DynamicStringRecord::DATA_SIZE + 10];
        let r = DynamicStringRecord::new_start(&big, big.len() as u32, 0).unwrap();
        assert_eq!(&r.data[..DynamicStringRecord::DATA_SIZE], &big[..DynamicStringRecord::DATA_SIZE]);
    }

    #[test]
    fn string_get_data_capped() {
        let r = DynamicStringRecord::new_start(b"hello", 5, 0).unwrap();
        assert_eq!(r.get_data(5), b"hello");
        assert_eq!(r.get_data(3), b"hel");
    }

    // --- DynamicStringRecord length encoding ---

    #[test]
    fn string_length_roundtrip_small()  {
        let mut r = DynamicStringRecord::new();
        r.set_length(42);
        assert_eq!(r.get_length(), 42);
    }

    #[test]
    fn string_length_roundtrip_medium() {
        let mut r = DynamicStringRecord::new();
        r.set_length(65535);
        assert_eq!(r.get_length(), 65535);
    }

    #[test]
    fn string_length_roundtrip_max() {
        let mut r = DynamicStringRecord::new();
        r.set_length(MAX_LENGTH);
        assert_eq!(r.get_length(), MAX_LENGTH);
    }

    // --- DynamicStringRecord delete ---

    #[test]
    fn string_delete_clears_flags() {
        let mut r = DynamicStringRecord::new_start(b"hi", 2, 0).unwrap();
        assert!(r.is_in_use());
        r.delete();
        assert!(!r.is_in_use());
        assert!(!r.is_start());
    }

    // --- DynamicStringRecord serialization ---

    #[test]
    fn string_round_trip_start_block() {
        let data = b"Test string data";
        let original = DynamicStringRecord::new_start(data, data.len() as u32, 42).unwrap();
        assert_eq!(original, DynamicStringRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn string_round_trip_continuation_block() {
        let original = DynamicStringRecord::new_continuation(b"cont data", 99);
        assert_eq!(original, DynamicStringRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn string_round_trip_empty() {
        let original = DynamicStringRecord::new();
        assert_eq!(original, DynamicStringRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn string_size_constant_matches_serialized() {
        assert_eq!(DynamicStringRecord::new().to_bytes().len(), DynamicStringRecord::SIZE);
    }

    // --- DynamicArrayRecord constructors ---

    #[test]
    fn array_new_is_deleted() {
        let r = DynamicArrayRecord::new();
        assert!(!r.is_in_use());
        assert!(!r.is_start());
        assert!(r.is_end());
        assert_eq!(r.get_length(), 0);
        assert_eq!(r.element_type, PropertyType::Empty);
    }

    #[test]
    fn array_default_equals_new() {
        assert_eq!(DynamicArrayRecord::default(), DynamicArrayRecord::new());
    }

    #[test]
    fn array_new_start_sets_flags_and_type() {
        let data: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
        let r = DynamicArrayRecord::new_start(PropertyType::Int, 2, &data, 0).unwrap();
        assert!(r.is_in_use() && r.is_start() && r.is_end());
        assert_eq!(r.element_type, PropertyType::Int);
        assert_eq!(r.get_length(), 2);
    }

    #[test]
    fn array_new_continuation_sets_flags() {
        let r = DynamicArrayRecord::new_continuation(b"payload", 5);
        assert!(r.is_in_use() && !r.is_start() && !r.is_end());
        assert_eq!(r.next_block, 5);
    }

    #[test]
    fn array_new_start_rejects_length_overflow() {
        assert_eq!(
            DynamicArrayRecord::new_start(PropertyType::Int, MAX_LENGTH + 1, &[], 0).unwrap_err(),
            DynamicError::LengthOverflow { value: MAX_LENGTH + 1, max: MAX_LENGTH }
        );
    }

    #[test]
    fn array_new_start_accepts_max_length() {
        let r = DynamicArrayRecord::new_start(PropertyType::Float, MAX_LENGTH, &[], 0).unwrap();
        assert_eq!(r.get_length(), MAX_LENGTH);
    }

    // --- DynamicArrayRecord length encoding ---

    #[test]
    fn array_length_roundtrip_small() {
        let mut r = DynamicArrayRecord::new();
        r.set_length(100);
        assert_eq!(r.get_length(), 100);
    }

    #[test]
    fn array_length_roundtrip_max() {
        let mut r = DynamicArrayRecord::new();
        r.set_length(MAX_LENGTH);
        assert_eq!(r.get_length(), MAX_LENGTH);
    }

    // --- DynamicArrayRecord delete ---

    #[test]
    fn array_delete_clears_flags() {
        let mut r = DynamicArrayRecord::new_start(PropertyType::Int, 10, &[], 0).unwrap();
        assert!(r.is_in_use());
        r.delete();
        assert!(!r.is_in_use() && !r.is_start());
    }

    // --- DynamicArrayRecord serialization ---

    #[test]
    fn array_round_trip_start_block() {
        let data = [0x01u8, 0x02, 0x03, 0x04];
        let original = DynamicArrayRecord::new_start(PropertyType::Int, 1, &data, 99).unwrap();
        assert_eq!(original, DynamicArrayRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn array_round_trip_continuation_block() {
        let original = DynamicArrayRecord::new_continuation(&[0xBBu8; 50], 0);
        assert_eq!(original, DynamicArrayRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn array_round_trip_empty() {
        let original = DynamicArrayRecord::new();
        assert_eq!(original, DynamicArrayRecord::from_bytes(original.to_bytes()));
    }

    #[test]
    fn array_size_constant_matches_serialized() {
        assert_eq!(DynamicArrayRecord::new().to_bytes().len(), DynamicArrayRecord::SIZE);
    }

    #[test]
    fn array_element_type_at_correct_offset() {
        let r = DynamicArrayRecord::new_start(PropertyType::Float, 1, &[], 0).unwrap();
        let bytes = r.to_bytes();
        assert_eq!(bytes[8], PropertyType::Float as u8);
    }

    #[test]
    fn array_all_property_types_roundtrip() {
        // Verify every known PropertyType survives serialization
        let types = [
            PropertyType::Bool, PropertyType::Byte, PropertyType::Short,
            PropertyType::Int, PropertyType::Long, PropertyType::Float,
            PropertyType::Double, PropertyType::Char, PropertyType::String,
            PropertyType::ShortString, PropertyType::IntArray,
        ];
        for pt in types {
            let original = DynamicArrayRecord::new_start(pt, 1, &[], 0).unwrap();
            let restored = DynamicArrayRecord::from_bytes(original.to_bytes());
            assert_eq!(restored.element_type, pt, "failed for {pt:?}");
        }
    }

    // --- Record trait ---

    #[test]
    fn string_record_round_trip_via_trait() {
        let original = DynamicStringRecord::new_start(b"via trait", 9, 0).unwrap();
        let bytes    = Record::to_bytes(&original);   // [u8; 128] — no allocation
        let restored: DynamicStringRecord = Record::from_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn array_record_round_trip_via_trait() {
        let original = DynamicArrayRecord::new_start(PropertyType::Int, 3, &[1, 2, 3], 0).unwrap();
        let bytes    = Record::to_bytes(&original);   // [u8; 128] — no allocation
        let restored: DynamicArrayRecord = Record::from_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn records_are_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<DynamicStringRecord>();
        assert_sync::<DynamicStringRecord>();
        assert_send::<DynamicArrayRecord>();
        assert_sync::<DynamicArrayRecord>();
    }

    // --- Error display ---

    #[test]
    fn error_messages_are_readable() {
        assert!(DynamicError::LengthOverflow { value: 99, max: 50 }
            .to_string().contains("99"));
    }
}