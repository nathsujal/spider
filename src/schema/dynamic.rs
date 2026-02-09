//! # Dynamic Storage
//!
//! Records for storing values too large to fit inline in property blocks.
//!
//! - [`DynamicStringRecord`] - For strings > 6 bytes
//! - [`DynamicArrayRecord`] - For arrays of any primitive type
//!
//! Both use 128-byte fixed-size records that can be chained for larger data.
//!
//! ## String Storage
//!
//! ```text
//! DynamicStringRecord (128 bytes)
//! ├── flags: u8        (bit 0 = in_use, bit 1 = is_start)
//! ├── length: [u8; 3]  (total string length, first block only)
//! ├── next_block: u32  (0 = end of chain)
//! └── data: [u8; 120]  (UTF-8 string data)
//! ```
//!
//! ## Array Storage
//!
//! ```text
//! DynamicArrayRecord (128 bytes)
//! ├── flags: u8         (bit 0 = in_use, bit 1 = is_start)
//! ├── length: [u8; 3]   (element count, first block only)
//! ├── next_block: u32   (0 = end of chain)
//! ├── element_type: u8  (PropertyType of elements)
//! └── data: [u8; 119]   (array data)
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// DynamicStringRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A block for storing long strings (128 bytes).
///
/// Strings > 6 bytes are stored in these blocks. Multiple blocks
/// can be chained together using `next_block` for very long strings.
#[derive(Debug, Clone, Copy)]
pub struct DynamicStringRecord {
    /// Flags: bit 0 = in_use, bit 1 = is_start of chain.
    pub flags: u8,
    /// Total string length in bytes (only valid in first block).
    pub length: [u8; 3],
    /// Next block ID in chain. 0 = end of chain.
    pub next_block: u32,
    /// UTF-8 string data.
    pub data: [u8; 120],
}

impl DynamicStringRecord {
    /// Serialized size in bytes.
    pub const SIZE: usize = 128;
    /// Maximum data bytes per block.
    pub const DATA_SIZE: usize = 120;

    const FLAG_IN_USE: u8 = 0b0000_0001;
    const FLAG_IS_START: u8 = 0b0000_0010;

    /// Create an empty record.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: 0,
            length: [0; 3],
            next_block: 0,
            data: [0; 120],
        }
    }

    /// Create the first block of a string chain.
    pub fn new_start(data: &[u8], total_length: u32, next_block: u32) -> Self {
        let mut record = Self::new();
        record.flags = Self::FLAG_IN_USE | Self::FLAG_IS_START;
        record.set_length(total_length);
        record.next_block = next_block;

        let copy_len = data.len().min(Self::DATA_SIZE);
        record.data[..copy_len].copy_from_slice(&data[..copy_len]);

        record
    }

    /// Create a continuation block (not the start).
    pub fn new_continuation(data: &[u8], next_block: u32) -> Self {
        let mut record = Self::new();
        record.flags = Self::FLAG_IN_USE;
        record.next_block = next_block;

        let copy_len = data.len().min(Self::DATA_SIZE);
        record.data[..copy_len].copy_from_slice(&data[..copy_len]);

        record
    }

    /// Returns true if this block is allocated.
    #[inline]
    pub fn is_in_use(&self) -> bool {
        self.flags & Self::FLAG_IN_USE != 0
    }

    /// Returns true if this is the first block of a string.
    #[inline]
    pub fn is_start(&self) -> bool {
        self.flags & Self::FLAG_IS_START != 0
    }

    /// Returns true if this is the last block in the chain.
    #[inline]
    pub fn is_end(&self) -> bool {
        self.next_block == 0
    }

    /// Get the total string length (only valid for start block).
    #[inline]
    pub fn get_length(&self) -> u32 {
        u32::from_le_bytes([self.length[0], self.length[1], self.length[2], 0])
    }

    /// Set the total string length.
    #[inline]
    pub fn set_length(&mut self, len: u32) {
        let bytes = len.to_le_bytes();
        self.length = [bytes[0], bytes[1], bytes[2]];
    }

    /// Mark this block as deleted.
    #[inline]
    pub fn delete(&mut self) {
        self.flags = 0;
    }

    /// Get data slice up to the expected length.
    #[inline]
    pub fn get_data(&self, expected_len: usize) -> &[u8] {
        &self.data[..expected_len.min(Self::DATA_SIZE)]
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0] = self.flags;
        bytes[1..4].copy_from_slice(&self.length);
        bytes[4..8].copy_from_slice(&self.next_block.to_le_bytes());
        bytes[8..128].copy_from_slice(&self.data);
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let mut data = [0u8; 120];
        data.copy_from_slice(&bytes[8..128]);

        Self {
            flags: bytes[0],
            length: [bytes[1], bytes[2], bytes[3]],
            next_block: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            data,
        }
    }
}

impl Default for DynamicStringRecord {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicArrayRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A block for storing arrays (128 bytes).
///
/// Arrays that don't fit inline are stored in these blocks.
/// The `element_type` indicates the type of array elements.
#[derive(Debug, Clone, Copy)]
pub struct DynamicArrayRecord {
    /// Flags: bit 0 = in_use, bit 1 = is_start.
    pub flags: u8,
    /// Element count (only valid in first block).
    pub length: [u8; 3],
    /// Next block ID in chain. 0 = end.
    pub next_block: u32,
    /// Element type (PropertyType enum value).
    pub element_type: u8,
    /// Array data.
    pub data: [u8; 119],
}

impl DynamicArrayRecord {
    /// Serialized size in bytes.
    pub const SIZE: usize = 128;
    /// Maximum data bytes per block.
    pub const DATA_SIZE: usize = 119;

    const FLAG_IN_USE: u8 = 0b0000_0001;
    const FLAG_IS_START: u8 = 0b0000_0010;

    /// Create an empty record.
    #[inline]
    pub const fn new() -> Self {
        Self {
            flags: 0,
            length: [0; 3],
            next_block: 0,
            element_type: 0,
            data: [0; 119],
        }
    }

    /// Create the first block of an array chain.
    pub fn new_start(element_type: u8, element_count: u32, data: &[u8], next_block: u32) -> Self {
        let mut record = Self::new();
        record.flags = Self::FLAG_IN_USE | Self::FLAG_IS_START;
        record.element_type = element_type;
        record.set_length(element_count);
        record.next_block = next_block;

        let copy_len = data.len().min(Self::DATA_SIZE);
        record.data[..copy_len].copy_from_slice(&data[..copy_len]);

        record
    }

    /// Returns true if this block is allocated.
    #[inline]
    pub fn is_in_use(&self) -> bool {
        self.flags & Self::FLAG_IN_USE != 0
    }

    /// Returns true if this is the first block of an array.
    #[inline]
    pub fn is_start(&self) -> bool {
        self.flags & Self::FLAG_IS_START != 0
    }

    /// Get the element count (only valid for start block).
    #[inline]
    pub fn get_length(&self) -> u32 {
        u32::from_le_bytes([self.length[0], self.length[1], self.length[2], 0])
    }

    /// Set the element count.
    #[inline]
    pub fn set_length(&mut self, len: u32) {
        let bytes = len.to_le_bytes();
        self.length = [bytes[0], bytes[1], bytes[2]];
    }

    /// Mark this block as deleted.
    #[inline]
    pub fn delete(&mut self) {
        self.flags = 0;
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0] = self.flags;
        bytes[1..4].copy_from_slice(&self.length);
        bytes[4..8].copy_from_slice(&self.next_block.to_le_bytes());
        bytes[8] = self.element_type;
        bytes[9..128].copy_from_slice(&self.data);
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let mut data = [0u8; 119];
        data.copy_from_slice(&bytes[9..128]);

        Self {
            flags: bytes[0],
            length: [bytes[1], bytes[2], bytes[3]],
            next_block: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            element_type: bytes[8],
            data,
        }
    }
}

impl Default for DynamicArrayRecord {
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
    fn string_record_size() {
        assert_eq!(
            std::mem::size_of::<DynamicStringRecord>(),
            DynamicStringRecord::SIZE
        );
    }

    #[test]
    fn array_record_size() {
        assert_eq!(
            std::mem::size_of::<DynamicArrayRecord>(),
            DynamicArrayRecord::SIZE
        );
    }

    #[test]
    fn string_record_creation() {
        let data = b"Hello, World!";
        let record = DynamicStringRecord::new_start(data, data.len() as u32, 0);

        assert!(record.is_in_use());
        assert!(record.is_start());
        assert!(record.is_end());
        assert_eq!(record.get_length(), 13);
    }

    #[test]
    fn string_record_serialization() {
        let data = b"Test string";
        let record = DynamicStringRecord::new_start(data, data.len() as u32, 42);

        let restored = DynamicStringRecord::from_bytes(record.to_bytes());

        assert_eq!(record.flags, restored.flags);
        assert_eq!(record.get_length(), restored.get_length());
        assert_eq!(record.next_block, restored.next_block);
        assert_eq!(&record.data[..11], &restored.data[..11]);
    }

    #[test]
    fn array_record_creation() {
        let data: [u8; 8] = [1, 0, 0, 0, 2, 0, 0, 0]; // Two i32s
        let record = DynamicArrayRecord::new_start(4, 2, &data, 0); // type 4 = Int

        assert!(record.is_in_use());
        assert!(record.is_start());
        assert_eq!(record.element_type, 4);
        assert_eq!(record.get_length(), 2);
    }
}
