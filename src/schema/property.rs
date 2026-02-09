//! # Property Storage
//!
//! Properties are key-value pairs stored as doubly-linked lists of [`PropertyRecord`]s.
//! Each record contains 4 [`PropertyBlock`]s, where each block holds one property.
//!
//! ## Storage Layout
//!
//! ```text
//! PropertyRecord (40 bytes)
//! ├── prev_prop_id: u32 (0 = head of chain)
//! ├── next_prop_id: u32 (0 = tail of chain)
//! └── blocks: [PropertyBlock; 4] (32 bytes)
//!
//! PropertyBlock (8 bytes) - bit layout:
//! ├── bits 0-7:   key_id (property key token ID)
//! ├── bits 8-12:  value_type (PropertyType enum)
//! └── bits 13-63: value or pointer (51 bits)
//! ```
//!
//! ## Example
//!
//! ```rust
//! use spider::schema::{PropertyBlock, PropertyType, PropertyRecord};
//!
//! // Create a property block for "age: 30"
//! let age_key_id = 1; // from TokenStore
//! let block = PropertyBlock::with_value(age_key_id, PropertyType::Int, 30);
//!
//! assert_eq!(block.key_id(), 1);
//! assert_eq!(block.value_type(), PropertyType::Int);
//! assert_eq!(block.value_bits(), 30);
//! ```

/// Property value types.
///
/// These types determine how to interpret the value bits in a [`PropertyBlock`].
/// Small values are stored inline; large values use pointers to dynamic stores.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    /// Block is empty/unused
    Empty = 0,

    // ─── Primitives (stored inline) ─────────────────────
    /// Boolean: 1 byte inline (0 = false, 1 = true)
    Bool = 1,
    /// Signed byte: 1 byte inline (i8)
    Byte = 2,
    /// Short integer: 2 bytes inline (i16)
    Short = 3,
    /// Integer: 4 bytes inline (i32)
    Int = 4,
    /// Long integer: 8 bytes, uses 2 blocks (i64)
    Long = 5,
    /// Float: 4 bytes inline (f32)
    Float = 6,
    /// Double: 8 bytes, uses 2 blocks (f64)
    Double = 7,
    /// Character: 2 bytes inline (UTF-16)
    Char = 8,

    // ─── Strings ────────────────────────────────────────
    /// Long string: pointer to strings.db
    String = 9,
    /// Short string: inline (≤6 bytes UTF-8)
    ShortString = 10,

    // ─── Arrays (pointers to arrays.db) ─────────────────
    ByteArray = 11,
    ShortArray = 12,
    IntArray = 13,
    LongArray = 14,
    FloatArray = 15,
    DoubleArray = 16,
    BoolArray = 17,
    CharArray = 18,
    StringArray = 19,

    // ─── Temporal ───────────────────────────────────────
    /// Date: days since Unix epoch (4 bytes)
    Date = 20,
    /// LocalTime: nanoseconds since midnight (8 bytes)
    LocalTime = 21,
    /// LocalDateTime: epoch seconds (8 bytes)
    LocalDateTime = 22,
    /// DateTime: with timezone info
    DateTime = 23,
    /// Duration: months, days, seconds, nanos
    Duration = 24,

    // ─── Spatial ────────────────────────────────────────
    /// 2D point: (x, y) coordinates
    Point2D = 25,
    /// 3D point: (x, y, z) coordinates
    Point3D = 26,
}

impl PropertyType {
    /// Convert from raw u8 value. Unknown values become `Empty`.
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Bool,
            2 => Self::Byte,
            3 => Self::Short,
            4 => Self::Int,
            5 => Self::Long,
            6 => Self::Float,
            7 => Self::Double,
            8 => Self::Char,
            9 => Self::String,
            10 => Self::ShortString,
            11 => Self::ByteArray,
            12 => Self::ShortArray,
            13 => Self::IntArray,
            14 => Self::LongArray,
            15 => Self::FloatArray,
            16 => Self::DoubleArray,
            17 => Self::BoolArray,
            18 => Self::CharArray,
            19 => Self::StringArray,
            20 => Self::Date,
            21 => Self::LocalTime,
            22 => Self::LocalDateTime,
            23 => Self::DateTime,
            24 => Self::Duration,
            25 => Self::Point2D,
            26 => Self::Point3D,
            _ => Self::Empty,
        }
    }

    /// Returns true if this type requires 2 consecutive blocks.
    #[inline]
    pub fn uses_two_blocks(&self) -> bool {
        matches!(self, Self::Long | Self::Double)
    }

    /// Returns true if the value is a pointer to dynamic storage.
    #[inline]
    pub fn is_pointer(&self) -> bool {
        matches!(
            self,
            Self::String
                | Self::ByteArray
                | Self::ShortArray
                | Self::IntArray
                | Self::LongArray
                | Self::FloatArray
                | Self::DoubleArray
                | Self::BoolArray
                | Self::CharArray
                | Self::StringArray
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyBlock
// ─────────────────────────────────────────────────────────────────────────────

/// A single property value (8 bytes).
///
/// Packs key ID, type, and value into a u64:
/// - Bits 0-7: property key ID (max 256 keys)
/// - Bits 8-12: value type (5 bits, max 32 types)
/// - Bits 13-63: value or pointer (51 bits)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PropertyBlock {
    data: u64,
}

impl PropertyBlock {
    /// Serialized size in bytes.
    pub const SIZE: usize = 8;

    /// Mask for 51-bit value field.
    const VALUE_MASK: u64 = 0x0007_FFFF_FFFF_FFFF;

    /// Create an empty block.
    #[inline]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Create a block with the given key, type, and value.
    #[inline]
    pub fn with_value(key_id: u8, value_type: PropertyType, value: u64) -> Self {
        let data =
            (key_id as u64) | ((value_type as u64) << 8) | ((value & Self::VALUE_MASK) << 13);
        Self { data }
    }

    /// Get the property key ID (0-255).
    #[inline]
    pub fn key_id(&self) -> u8 {
        (self.data & 0xFF) as u8
    }

    /// Get the value type.
    #[inline]
    pub fn value_type(&self) -> PropertyType {
        PropertyType::from_u8(((self.data >> 8) & 0x1F) as u8)
    }

    /// Get the raw value bits (51 bits).
    #[inline]
    pub fn value_bits(&self) -> u64 {
        self.data >> 13
    }

    /// Check if this block is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data == 0
    }

    /// Get raw u64 representation.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.data
    }

    /// Deserialize from bytes (little-endian).
    #[inline]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        Self {
            data: u64::from_le_bytes(bytes),
        }
    }

    /// Serialize to bytes (little-endian).
    #[inline]
    pub fn to_bytes(&self) -> [u8; 8] {
        self.data.to_le_bytes()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropertyRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A property record containing up to 4 properties (40 bytes).
///
/// Records form a doubly-linked list for nodes/relationships with >4 properties.
#[derive(Debug, Clone, Copy)]
pub struct PropertyRecord {
    /// Previous record in chain (0 = this is the head).
    pub prev_prop_id: u32,
    /// Next record in chain (0 = this is the tail).
    pub next_prop_id: u32,
    /// Up to 4 property blocks.
    pub blocks: [PropertyBlock; 4],
}

impl PropertyRecord {
    /// Serialized size in bytes.
    pub const SIZE: usize = 40;

    /// Create an empty property record.
    pub fn new() -> Self {
        Self {
            prev_prop_id: 0,
            next_prop_id: 0,
            blocks: [PropertyBlock::new(); 4],
        }
    }

    /// Returns true if this is the first record in the chain.
    #[inline]
    pub fn is_head(&self) -> bool {
        self.prev_prop_id == 0
    }

    /// Returns true if this is the last record in the chain.
    #[inline]
    pub fn is_tail(&self) -> bool {
        self.next_prop_id == 0
    }

    /// Count of non-empty property blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.iter().filter(|b| !b.is_empty()).count()
    }

    /// Index of first empty block, or None if all are full.
    pub fn first_empty_slot(&self) -> Option<usize> {
        self.blocks.iter().position(|b| b.is_empty())
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.prev_prop_id.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.next_prop_id.to_le_bytes());
        for (i, block) in self.blocks.iter().enumerate() {
            let offset = 8 + i * 8;
            bytes[offset..offset + 8].copy_from_slice(&block.to_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let prev_prop_id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let next_prop_id = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let mut blocks = [PropertyBlock::new(); 4];
        for i in 0..4 {
            let offset = 8 + i * 8;
            let block_bytes: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
            blocks[i] = PropertyBlock::from_bytes(block_bytes);
        }

        Self {
            prev_prop_id,
            next_prop_id,
            blocks,
        }
    }
}

impl Default for PropertyRecord {
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

    #[test]
    fn property_type_roundtrip() {
        for i in 0..=26 {
            let t = PropertyType::from_u8(i);
            assert_eq!(t as u8, i);
        }
    }

    #[test]
    fn property_block_encode_decode() {
        let block = PropertyBlock::with_value(5, PropertyType::Int, 42);
        assert_eq!(block.key_id(), 5);
        assert_eq!(block.value_type(), PropertyType::Int);
        assert_eq!(block.value_bits(), 42);
    }

    #[test]
    fn property_block_serialization() {
        let block = PropertyBlock::with_value(10, PropertyType::String, 12345);
        let restored = PropertyBlock::from_bytes(block.to_bytes());
        assert_eq!(block.raw(), restored.raw());
    }

    #[test]
    fn property_record_size() {
        assert_eq!(std::mem::size_of::<PropertyRecord>(), PropertyRecord::SIZE);
    }

    #[test]
    fn property_record_serialization() {
        let mut record = PropertyRecord::new();
        record.prev_prop_id = 100;
        record.next_prop_id = 200;
        record.blocks[0] = PropertyBlock::with_value(1, PropertyType::Int, 42);

        let restored = PropertyRecord::from_bytes(record.to_bytes());

        assert_eq!(record.prev_prop_id, restored.prev_prop_id);
        assert_eq!(record.next_prop_id, restored.next_prop_id);
        assert_eq!(record.blocks[0].raw(), restored.blocks[0].raw());
    }
}
