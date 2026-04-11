//! Property storage — inline key-value blocks and chained property records.
//!
//! Small values (bool, i32, f32, strings ≤6 bytes) live entirely inside a
//! [`PropertyBlock`]. Large values store a record ID pointing into the
//! dynamic store (`strings.db`, `arrays.db`).
//!
//! ```text
//! PropertyRecord (40 bytes)
//! ├── prev_prop_id : u32              (0 = head of chain)
//! ├── next_prop_id : u32              (0 = tail of chain)
//! └── blocks       : [PropertyBlock; 4]   (4 × 8 = 32 bytes)
//!
//! PropertyBlock (8 bytes) — packed u64:
//! ├── bits  0– 7 : key_id     (u8)
//! ├── bits  8–12 : value_type (5-bit discriminant)
//! └── bits 13–63 : value/ptr  (51 bits)
//! ```

use crate::store::record::{self, Record};

// Compile-time Send + Sync checks — zero runtime cost.
// Fails to compile if either type ever gains a non-Send/Sync field.
fn _assert_property_block_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<PropertyBlock>();
    assert_sync::<PropertyBlock>();
}

fn _assert_property_record_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<PropertyRecord>();
    assert_sync::<PropertyRecord>();
}

// --- PropKeyId ---

/// A non-zero property key token ID.
///
/// `0` is the empty sentinel and cannot be constructed through the public API.
/// [`PropKeyId::new_unchecked`] bypasses the check for the storage layer,
/// which only reads bytes it previously validated on write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PropKeyId(u8);

impl PropKeyId {
    /// Returns `Err` if `id == 0`.
    #[inline]
    pub fn new(id: u8) -> Result<Self, PropertyError> {
        if id == 0 {
            return Err(PropertyError::InvalidKeyId);
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

impl std::fmt::Display for PropKeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PropKeyId({})", self.0)
    }
}

// --- Errors ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyError {
    /// A raw key value of `0` was given to [`PropKeyId::new`].
    InvalidKeyId,
    /// Integer value is outside the 51-bit signed range `[-(2^50), 2^50-1]`.
    IntOutOfRange { value: i64, min: i64, max: i64 },
    /// String is too long to store inline (max 6 UTF-8 bytes).
    ShortStringTooLong { len: usize, max: usize },
}

impl std::fmt::Display for PropertyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKeyId =>
                write!(f, "prop key id 0 is invalid (reserved as the empty sentinel)"),
            Self::IntOutOfRange { value, min, max } =>
                write!(f, "integer {value} is out of 51-bit range [{min}, {max}]"),
            Self::ShortStringTooLong { len, max } =>
                write!(f, "short string is {len} bytes, maximum inline is {max}"),
        }
    }
}

impl std::error::Error for PropertyError {}

// --- PropertyType ---

/// Determines how to interpret the 51 value bits in a [`PropertyBlock`].
///
/// `Empty = 0` marks an unused block slot. Variants 1–8 are stored inline.
/// Variants 9+ use a pointer into the dynamic store or a special layout.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    /// Block is empty / unused.
    Empty = 0,

    // -- Inline primitives --
    /// Boolean — 1 bit inline.
    Bool = 1,
    /// Signed byte — 1 byte inline (`i8`).
    Byte = 2,
    /// Short integer — 2 bytes inline (`i16`).
    Short = 3,
    /// Integer — 4 bytes inline (`i32`).
    Int = 4,
    /// Long integer — 8 bytes; uses 2 consecutive blocks (`i64`).
    Long = 5,
    /// Float — 4 bytes inline (`f32`).
    Float = 6,
    /// Double — 8 bytes; uses 2 consecutive blocks (`f64`).
    Double = 7,
    /// Character — 2 bytes inline (UTF-16 code unit).
    Char = 8,

    // -- Strings --
    /// Long string — pointer to `strings.db`.
    String = 9,
    /// Short string — up to 6 UTF-8 bytes stored inline.
    ShortString = 10,

    // -- Arrays (pointer to arrays.db) --
    ByteArray   = 11,
    ShortArray  = 12,
    IntArray    = 13,
    LongArray   = 14,
    FloatArray  = 15,
    DoubleArray = 16,
    BoolArray   = 17,
    CharArray   = 18,
    StringArray = 19,

    // -- Temporal --
    /// Days since Unix epoch (4 bytes inline).
    Date          = 20,
    /// Nanoseconds since midnight (8 bytes; 2 blocks).
    LocalTime     = 21,
    /// Seconds since Unix epoch (8 bytes; 2 blocks).
    LocalDateTime = 22,
    /// Datetime with timezone info.
    DateTime      = 23,
    /// Duration — months, days, seconds, nanos.
    Duration      = 24,

    // -- Spatial --
    /// 2-D point `(x, y)`.
    Point2D = 25,
    /// 3-D point `(x, y, z)`.
    Point3D = 26,
}

impl PropertyType {
    /// Convert from a raw `u8`. Unknown discriminants map to [`Self::Empty`].
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            1  => Self::Bool,
            2  => Self::Byte,
            3  => Self::Short,
            4  => Self::Int,
            5  => Self::Long,
            6  => Self::Float,
            7  => Self::Double,
            8  => Self::Char,
            9  => Self::String,
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
            _  => Self::Empty,
        }
    }

    /// `true` if this type requires 2 consecutive blocks to store its value.
    #[inline]
    pub fn uses_two_blocks(self) -> bool {
        matches!(self, Self::Long | Self::Double | Self::LocalTime | Self::LocalDateTime)
    }

    /// `true` if the value bits hold a pointer into the dynamic store.
    #[inline]
    pub fn is_pointer(self) -> bool {
        matches!(
            self,
            Self::String
                | Self::ByteArray  | Self::ShortArray | Self::IntArray
                | Self::LongArray  | Self::FloatArray  | Self::DoubleArray
                | Self::BoolArray  | Self::CharArray   | Self::StringArray
        )
    }
}

// --- PropertyBlock ---

/// A single packed property — 8 bytes on disk.
///
/// ```text
/// bits  0– 7 → key_id     (u8)
/// bits  8–12 → value_type (5-bit discriminant)
/// bits 13–63 → value      (51 bits, inline or pointer)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PropertyBlock {
    data: u64,
}

impl PropertyBlock {
    /// Serialized size in bytes.
    pub const SIZE: usize = 8;
    /// Maximum inline string length in bytes.
    pub const MAX_SHORT_STRING: usize = 6;
    /// Minimum storable signed integer (`-(2^50)`).
    pub const INT_MIN: i64 = -(1i64 << 50);
    /// Maximum storable signed integer (`2^50 - 1`).
    pub const INT_MAX: i64 = (1i64 << 50) - 1;

    const VALUE_MASK: u64 = 0x0007_FFFF_FFFF_FFFF;

    // --- Constructors ---

    /// Create an empty (unused) block.
    #[inline]
    pub const fn new() -> Self {
        Self { data: 0 }
    }

    /// Create a block with an explicit key, type, and raw value bits.
    ///
    /// Prefer the typed constructors (`from_bool`, `from_int`, …) over this.
    #[inline]
    pub fn with_value(key_id: PropKeyId, value_type: PropertyType, value: u64) -> Self {
        let data = (key_id.get() as u64)
            | ((value_type as u64) << 8)
            | ((value & Self::VALUE_MASK) << 13);
        Self { data }
    }

    // --- Field accessors ---

    /// Property key token ID.
    ///
    /// Returns `None` for an empty block (`key_id == 0`).
    #[inline]
    pub fn key_id(&self) -> Option<PropKeyId> {
        let raw = (self.data & 0xFF) as u8;
        if raw == 0 { None } else { Some(PropKeyId::new_unchecked(raw)) }
    }

    /// Value type discriminant.
    #[inline]
    pub fn value_type(&self) -> PropertyType {
        PropertyType::from_u8(((self.data >> 8) & 0x1F) as u8)
    }

    /// Raw 51-bit value (inline value or pointer).
    #[inline]
    pub fn value_bits(&self) -> u64 {
        (self.data >> 13) & Self::VALUE_MASK
    }

    /// `true` if this block is unused (`data == 0`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data == 0
    }

    /// Raw `u64` representation.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.data
    }

    // --- Serialization ---

    /// Serialize to exactly 8 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        self.data.to_le_bytes()
    }

    /// Deserialize from exactly 8 bytes, little-endian.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self { data: u64::from_le_bytes(bytes) }
    }

    // --- Typed constructors ---

    /// Store a boolean value.
    #[inline]
    pub fn from_bool(key_id: PropKeyId, value: bool) -> Self {
        Self::with_value(key_id, PropertyType::Bool, value as u64)
    }

    /// Store a signed integer.
    ///
    /// # Errors
    /// Returns [`PropertyError::IntOutOfRange`] if `value` is outside
    /// `[INT_MIN, INT_MAX]` (±2^50).
    pub fn from_int(key_id: PropKeyId, value: i64) -> Result<Self, PropertyError> {
        if !(Self::INT_MIN..=Self::INT_MAX).contains(&value) {
            return Err(PropertyError::IntOutOfRange {
                value,
                min: Self::INT_MIN,
                max: Self::INT_MAX,
            });
        }
        let bits = (value as u64) & Self::VALUE_MASK;
        Ok(Self::with_value(key_id, PropertyType::Int, bits))
    }

    /// Store a 32-bit float.
    #[inline]
    pub fn from_float(key_id: PropKeyId, value: f32) -> Self {
        Self::with_value(key_id, PropertyType::Float, value.to_bits() as u64)
    }

    /// Store a short string inline (max [`Self::MAX_SHORT_STRING`] UTF-8 bytes).
    ///
    /// Inline layout within the 51 value bits:
    /// - bits 0–2  : length (0–6)
    /// - bits 3–50 : up to 6 raw UTF-8 bytes
    ///
    /// # Errors
    /// Returns [`PropertyError::ShortStringTooLong`] if `s.len() > 6`.
    pub fn from_short_string(key_id: PropKeyId, s: &str) -> Result<Self, PropertyError> {
        let bytes = s.as_bytes();
        if bytes.len() > Self::MAX_SHORT_STRING {
            return Err(PropertyError::ShortStringTooLong {
                len: bytes.len(),
                max: Self::MAX_SHORT_STRING,
            });
        }
        let mut val = bytes.len() as u64;
        for (i, &b) in bytes.iter().enumerate() {
            val |= (b as u64) << (3 + i * 8);
        }
        Ok(Self::with_value(key_id, PropertyType::ShortString, val))
    }

    /// Store a pointer to a `DynamicStringRecord` in `strings.db`.
    #[inline]
    pub fn from_dyn_string_ptr(key_id: PropKeyId, record_id: u32) -> Self {
        Self::with_value(key_id, PropertyType::String, record_id as u64)
    }

    // --- Typed decoders ---

    /// Decode as boolean. Returns `None` if the type does not match.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        (self.value_type() == PropertyType::Bool).then(|| self.value_bits() != 0)
    }

    /// Decode as signed integer. Returns `None` if the type does not match.
    ///
    /// Sign-extends the 51-bit two's complement value back to `i64`.
    pub fn as_int(&self) -> Option<i64> {
        if self.value_type() != PropertyType::Int {
            return None;
        }
        let raw = self.value_bits();
        // bit 50 is the sign bit for the 51-bit signed integer
        Some(if raw & (1 << 50) != 0 {
            (raw | !Self::VALUE_MASK) as i64  // fill upper 13 bits with 1s
        } else {
            raw as i64
        })
    }

    /// Decode as `f32`. Returns `None` if the type does not match.
    #[inline]
    pub fn as_float(&self) -> Option<f32> {
        (self.value_type() == PropertyType::Float)
            .then(|| f32::from_bits(self.value_bits() as u32))
    }

    /// Decode an inline short string. Returns `None` if the type does not match.
    ///
    /// # Panics
    /// Panics if the stored bytes are not valid UTF-8. This indicates on-disk
    /// corruption — the codebase only ever writes valid UTF-8 into this field.
    /// Run `spider gc --verify` to assess integrity.
    pub fn as_short_string(&self) -> Option<String> {
        if self.value_type() != PropertyType::ShortString {
            return None;
        }
        let raw = self.value_bits();
        let len = (raw & 0x7) as usize;
        let mut bytes = Vec::with_capacity(len);
        for i in 0..len {
            bytes.push(((raw >> (3 + i * 8)) & 0xFF) as u8);
        }
        Some(String::from_utf8(bytes).unwrap_or_else(|_| {
            panic!(
                "Storage corruption: inline ShortString bytes are not valid UTF-8. \
                 Run `spider gc --verify` to assess integrity."
            )
        }))
    }

    /// Get the `DynamicString` record ID. Returns `None` if the type does not match.
    #[inline]
    pub fn dyn_string_ptr(&self) -> Option<u32> {
        (self.value_type() == PropertyType::String).then(|| self.value_bits() as u32)
    }
}

// --- PropertyRecord ---

/// A property record holding up to 4 [`PropertyBlock`]s — 40 bytes on disk.
///
/// Records chain into a doubly-linked list when a node or edge has more than
/// 4 properties. The tombstone condition is all blocks empty with no chain
/// links — the store layer never produces empty mid-chain records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyRecord {
    /// Previous record in the chain. `0` = head.
    pub prev_prop_id: u32,
    /// Next record in the chain. `0` = tail.
    pub next_prop_id: u32,
    /// Up to 4 packed property blocks.
    pub blocks: [PropertyBlock; 4],
}

impl PropertyRecord {
    pub const SIZE: usize = 40;

    /// Create an empty property record (all blocks unused).
    #[inline]
    pub fn new() -> Self {
        Self {
            prev_prop_id: 0,
            next_prop_id: 0,
            blocks: [PropertyBlock::new(); 4],
        }
    }

    // --- State queries ---

    /// `true` if this is the head of the property chain.
    #[inline] pub fn is_head(&self) -> bool { self.prev_prop_id == 0 }
    /// `true` if this is the tail of the property chain.
    #[inline] pub fn is_tail(&self) -> bool { self.next_prop_id == 0 }

    /// `true` if all 4 blocks are empty.
    #[inline]
    pub fn is_empty_record(&self) -> bool {
        self.blocks.iter().all(|b| b.is_empty())
    }

    /// Number of non-empty blocks (`0..=4`).
    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.iter().filter(|b| !b.is_empty()).count()
    }

    /// Index of the first empty block slot, or `None` if all 4 are occupied.
    #[inline]
    pub fn first_empty_slot(&self) -> Option<usize> {
        self.blocks.iter().position(|b| b.is_empty())
    }

    // --- Serialization ---

    /// Serialize to 40 bytes, little-endian.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.prev_prop_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.next_prop_id.to_le_bytes());
        for (i, block) in self.blocks.iter().enumerate() {
            let off = 8 + i * 8;
            buf[off..off + 8].copy_from_slice(&block.to_bytes());
        }
        buf
    }

    /// Deserialize from 40 bytes, little-endian.
    #[inline]
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let prev_prop_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let next_prop_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());

        let mut blocks = [PropertyBlock::new(); 4];
        for (i, block) in blocks.iter_mut().enumerate() {
            let off = 8 + i * 8;
            *block = PropertyBlock::from_bytes(bytes[off..off + 8].try_into().unwrap());
        }

        Self { prev_prop_id, next_prop_id, blocks }
    }
}

impl Default for PropertyRecord {
    fn default() -> Self {
        Self::new()
    }
}

// --- Record impl ---

// Declares PropertyRecord as a permitted storage type (sealed trait requirement).
impl record::private::Sealed for PropertyRecord {}

impl Record for PropertyRecord {
    const SIZE: usize = PropertyRecord::SIZE;

    /// Zero-alloc: lives entirely on the stack.
    type Bytes = [u8; PropertyRecord::SIZE];

    #[inline]
    fn to_bytes(&self) -> [u8; PropertyRecord::SIZE] {
        self.to_bytes()
    }

    /// # Panics
    /// The store layer guarantees exactly `SIZE` bytes; a wrong-size input
    /// means on-disk corruption. Run `spider gc --verify` to assess integrity.
    #[inline]
    fn from_bytes(bytes: [u8; PropertyRecord::SIZE]) -> Self {
        PropertyRecord::from_bytes(bytes)
    }

    #[inline]
    fn from_raw(bytes: &[u8]) -> Self {
        PropertyRecord::from_bytes(bytes.try_into().unwrap())
    }

    /// `true` if this slot is a tombstone (all blocks empty, no chain links).
    ///
    /// The store layer never produces empty mid-chain records, so
    /// `prev == 0 && next == 0 && no blocks` is a reliable tombstone signal.
    #[inline]
    fn is_deleted(&self) -> bool {
        self.prev_prop_id == 0 && self.next_prop_id == 0 && self.is_empty_record()
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn kid(id: u8) -> PropKeyId {
        PropKeyId::new(id).expect("test passed key id 0")
    }

    // --- PropKeyId ---

    #[test]
    fn key_id_rejects_zero() {
        assert_eq!(PropKeyId::new(0), Err(PropertyError::InvalidKeyId));
    }

    #[test]
    fn key_id_accepts_nonzero() {
        assert_eq!(PropKeyId::new(5).unwrap().get(), 5);
    }

    #[test]
    fn key_id_unchecked_preserves_value() {
        assert_eq!(PropKeyId::new_unchecked(7).get(), 7);
    }

    #[test]
    fn key_id_max_value() {
        assert_eq!(PropKeyId::new(255).unwrap().get(), 255);
    }

    // --- PropertyType ---

    #[test]
    fn property_type_known_values_roundtrip() {
        for i in 1u8..=26 {
            let t = PropertyType::from_u8(i);
            assert_eq!(t as u8, i, "failed at discriminant {i}");
        }
    }

    #[test]
    fn property_type_unknown_maps_to_empty() {
        assert_eq!(PropertyType::from_u8(27), PropertyType::Empty);
        assert_eq!(PropertyType::from_u8(255), PropertyType::Empty);
    }

    #[test]
    fn property_type_empty_is_zero() {
        assert_eq!(PropertyType::Empty as u8, 0);
    }

    #[test]
    fn uses_two_blocks() {
        assert!(PropertyType::Long.uses_two_blocks());
        assert!(PropertyType::Double.uses_two_blocks());
        assert!(PropertyType::LocalTime.uses_two_blocks());
        assert!(PropertyType::LocalDateTime.uses_two_blocks());
        assert!(!PropertyType::Int.uses_two_blocks());
        assert!(!PropertyType::Float.uses_two_blocks());
        assert!(!PropertyType::Bool.uses_two_blocks());
    }

    #[test]
    fn is_pointer() {
        assert!(PropertyType::String.is_pointer());
        assert!(PropertyType::IntArray.is_pointer());
        assert!(PropertyType::StringArray.is_pointer());
        assert!(!PropertyType::Int.is_pointer());
        assert!(!PropertyType::ShortString.is_pointer());
    }

    // --- PropertyBlock basics ---

    #[test]
    fn new_block_is_empty() {
        let b = PropertyBlock::new();
        assert!(b.is_empty());
        assert_eq!(b.key_id(), None);
        assert_eq!(b.value_type(), PropertyType::Empty);
        assert_eq!(b.value_bits(), 0);
    }

    #[test]
    fn with_value_encodes_all_fields() {
        let b = PropertyBlock::with_value(kid(5), PropertyType::Int, 42);
        assert_eq!(b.key_id(), Some(kid(5)));
        assert_eq!(b.value_type(), PropertyType::Int);
        assert_eq!(b.value_bits(), 42);
    }

    #[test]
    fn block_serialization_roundtrip() {
        let b = PropertyBlock::with_value(kid(10), PropertyType::String, 12345);
        let restored = PropertyBlock::from_bytes(b.to_bytes());
        assert_eq!(b, restored);
    }

    #[test]
    fn block_size_constant_matches_serialized() {
        assert_eq!(PropertyBlock::new().to_bytes().len(), PropertyBlock::SIZE);
    }

    // --- Bool ---

    #[test]
    fn bool_true_roundtrip() {
        let b = PropertyBlock::from_bool(kid(1), true);
        assert_eq!(b.value_type(), PropertyType::Bool);
        assert_eq!(b.as_bool(), Some(true));
    }

    #[test]
    fn bool_false_roundtrip() {
        assert_eq!(PropertyBlock::from_bool(kid(1), false).as_bool(), Some(false));
    }

    #[test]
    fn bool_wrong_type_returns_none() {
        assert_eq!(PropertyBlock::with_value(kid(1), PropertyType::Int, 1).as_bool(), None);
    }

    // --- Int ---

    #[test]
    fn int_zero_roundtrip() {
        assert_eq!(PropertyBlock::from_int(kid(1), 0).unwrap().as_int(), Some(0));
    }

    #[test]
    fn int_positive_roundtrip() {
        assert_eq!(PropertyBlock::from_int(kid(1), 12345).unwrap().as_int(), Some(12345));
    }

    #[test]
    fn int_negative_roundtrip() {
        assert_eq!(PropertyBlock::from_int(kid(1), -42).unwrap().as_int(), Some(-42));
    }

    #[test]
    fn int_max_boundary_roundtrip() {
        let b = PropertyBlock::from_int(kid(1), PropertyBlock::INT_MAX).unwrap();
        assert_eq!(b.as_int(), Some(PropertyBlock::INT_MAX));
    }

    #[test]
    fn int_min_boundary_roundtrip() {
        let b = PropertyBlock::from_int(kid(1), PropertyBlock::INT_MIN).unwrap();
        assert_eq!(b.as_int(), Some(PropertyBlock::INT_MIN));
    }

    #[test]
    fn int_above_max_rejected() {
        assert!(matches!(
            PropertyBlock::from_int(kid(1), PropertyBlock::INT_MAX + 1).unwrap_err(),
            PropertyError::IntOutOfRange { .. }
        ));
    }

    #[test]
    fn int_below_min_rejected() {
        assert!(matches!(
            PropertyBlock::from_int(kid(1), PropertyBlock::INT_MIN - 1).unwrap_err(),
            PropertyError::IntOutOfRange { .. }
        ));
    }

    #[test]
    fn int_wrong_type_returns_none() {
        assert_eq!(PropertyBlock::from_bool(kid(1), true).as_int(), None);
    }

    // --- Float ---

    #[test]
    fn float_positive_roundtrip() {
        let decoded = PropertyBlock::from_float(kid(1), 3.14f32).as_float().unwrap();
        assert!((decoded - 3.14f32).abs() < f32::EPSILON);
    }

    #[test]
    fn float_negative_roundtrip() {
        assert_eq!(PropertyBlock::from_float(kid(1), -1.5f32).as_float(), Some(-1.5f32));
    }

    #[test]
    fn float_zero_roundtrip() {
        assert_eq!(PropertyBlock::from_float(kid(1), 0.0f32).as_float(), Some(0.0f32));
    }

    #[test]
    fn float_wrong_type_returns_none() {
        assert_eq!(PropertyBlock::from_bool(kid(1), true).as_float(), None);
    }

    // --- Short string ---

    #[test]
    fn short_string_empty_roundtrip() {
        let b = PropertyBlock::from_short_string(kid(1), "").unwrap();
        assert_eq!(b.as_short_string(), Some(String::new()));
    }

    #[test]
    fn short_string_ascii_roundtrip() {
        let b = PropertyBlock::from_short_string(kid(1), "hello").unwrap();
        assert_eq!(b.as_short_string(), Some("hello".to_string()));
    }

    #[test]
    fn short_string_max_length_roundtrip() {
        let b = PropertyBlock::from_short_string(kid(1), "abcdef").unwrap();
        assert_eq!(b.as_short_string(), Some("abcdef".to_string()));
    }

    #[test]
    fn short_string_too_long_rejected() {
        assert_eq!(
            PropertyBlock::from_short_string(kid(1), "toolong7").unwrap_err(),
            PropertyError::ShortStringTooLong { len: 8, max: 6 }
        );
    }

    #[test]
    fn short_string_wrong_type_returns_none() {
        assert!(PropertyBlock::from_bool(kid(1), true).as_short_string().is_none());
    }

    // --- Dyn string pointer ---

    #[test]
    fn dyn_string_ptr_roundtrip() {
        let b = PropertyBlock::from_dyn_string_ptr(kid(2), 99999);
        assert_eq!(b.value_type(), PropertyType::String);
        assert_eq!(b.dyn_string_ptr(), Some(99999));
    }

    #[test]
    fn dyn_string_ptr_wrong_type_returns_none() {
        assert_eq!(PropertyBlock::from_bool(kid(1), true).dyn_string_ptr(), None);
    }

    // --- PropertyRecord basics ---

    #[test]
    fn new_record_is_empty() {
        let r = PropertyRecord::new();
        assert!(r.is_empty_record());
        assert_eq!(r.block_count(), 0);
        assert!(r.is_head());
        assert!(r.is_tail());
    }

    #[test]
    fn default_equals_new() {
        assert_eq!(PropertyRecord::default(), PropertyRecord::new());
    }

    #[test]
    fn record_with_blocks_is_not_empty() {
        let mut r = PropertyRecord::new();
        r.blocks[0] = PropertyBlock::from_bool(kid(1), true);
        assert!(!r.is_empty_record());
        assert_eq!(r.block_count(), 1);
    }

    #[test]
    fn first_empty_slot_on_fresh_record() {
        assert_eq!(PropertyRecord::new().first_empty_slot(), Some(0));
    }

    #[test]
    fn first_empty_slot_when_full() {
        let mut r = PropertyRecord::new();
        for i in 0..4 {
            r.blocks[i] = PropertyBlock::from_bool(kid(i as u8 + 1), true);
        }
        assert_eq!(r.first_empty_slot(), None);
        assert_eq!(r.block_count(), 4);
    }

    #[test]
    fn first_empty_slot_finds_correct_index() {
        let mut r = PropertyRecord::new();
        r.blocks[0] = PropertyBlock::from_bool(kid(1), true);
        r.blocks[1] = PropertyBlock::from_bool(kid(2), false);
        assert_eq!(r.first_empty_slot(), Some(2));
    }

    // --- PropertyRecord serialization ---

    #[test]
    fn record_round_trip_preserves_all_fields() {
        let mut r = PropertyRecord::new();
        r.prev_prop_id = 100;
        r.next_prop_id = 200;
        r.blocks[0] = PropertyBlock::from_int(kid(1), 42).unwrap();
        r.blocks[1] = PropertyBlock::from_bool(kid(2), true);
        r.blocks[2] = PropertyBlock::from_float(kid(3), 1.5f32);

        assert_eq!(r, PropertyRecord::from_bytes(r.to_bytes()));
    }

    #[test]
    fn record_round_trip_empty() {
        let r = PropertyRecord::new();
        assert_eq!(r, PropertyRecord::from_bytes(r.to_bytes()));
    }

    #[test]
    fn record_size_constant_matches_serialized() {
        assert_eq!(PropertyRecord::new().to_bytes().len(), PropertyRecord::SIZE);
    }

    // --- Record trait ---

    #[test]
    fn record_round_trip_via_trait() {
        let mut r = PropertyRecord::new();
        r.blocks[0] = PropertyBlock::from_int(kid(1), 99).unwrap();
        let bytes    = Record::to_bytes(&r);       // [u8; 40] — no allocation
        let restored: PropertyRecord = Record::from_bytes(bytes);
        assert_eq!(r, restored);
    }

    #[test]
    fn record_is_deleted_only_when_truly_empty() {
        // tombstone: no blocks, no chain links
        let empty = PropertyRecord::new();
        assert!(Record::is_deleted(&empty));

        // live record with chain links but no blocks — NOT deleted
        let mut chained = PropertyRecord::new();
        chained.prev_prop_id = 1;
        chained.next_prop_id = 3;
        assert!(!Record::is_deleted(&chained));

        // live record with blocks
        let mut with_blocks = PropertyRecord::new();
        with_blocks.blocks[0] = PropertyBlock::from_bool(kid(1), true);
        assert!(!Record::is_deleted(&with_blocks));
    }

    #[test]
    fn record_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<PropertyRecord>();
        assert_sync::<PropertyRecord>();
        assert_send::<PropertyBlock>();
        assert_sync::<PropertyBlock>();
    }

    // --- Error display ---

    #[test]
    fn error_messages_are_readable() {
        assert!(PropertyError::InvalidKeyId.to_string().contains("0"));
        assert!(PropertyError::IntOutOfRange { value: 999, min: -1, max: 1 }
            .to_string().contains("999"));
        assert!(PropertyError::ShortStringTooLong { len: 10, max: 6 }
            .to_string().contains("10"));
    }
}