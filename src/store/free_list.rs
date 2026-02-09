//! # Free List
//!
//! Track deleted record slots for ID reuse.
//!
//! When a record is deleted, its ID is pushed onto the free list.
//! On allocation, IDs are popped from the free list (LIFO) before
//! using new sequential IDs.
//!
//! ## Example
//!
//! ```rust
//! use spider::store::FreeList;
//!
//! let mut list = FreeList::new();
//! let mut next_id = 1;
//!
//! // Allocate IDs sequentially
//! assert_eq!(list.allocate(&mut next_id), 1);
//! assert_eq!(list.allocate(&mut next_id), 2);
//!
//! // Free ID 1
//! list.free(1);
//!
//! // Next allocation reuses ID 1
//! assert_eq!(list.allocate(&mut next_id), 1);
//! ```

use super::Result;

/// Track deleted record slots for ID reuse (LIFO).
#[derive(Debug, Clone, Default)]
pub struct FreeList {
    free_ids: Vec<u32>,
}

impl FreeList {
    /// Create an empty free list.
    #[inline]
    pub fn new() -> Self {
        Self { free_ids: Vec::new() }
    }

    /// Allocate an ID. Reuses a freed ID if available, otherwise uses next_id.
    pub fn allocate(&mut self, next_id: &mut u32) -> u32 {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = *next_id;
            *next_id += 1;
            id
        })
    }

    /// Return an ID to the free list for reuse.
    #[inline]
    pub fn free(&mut self, id: u32) {
        if id != 0 {
            self.free_ids.push(id);
        }
    }

    /// Number of free IDs available.
    #[inline]
    pub fn len(&self) -> usize {
        self.free_ids.len()
    }

    /// Check if free list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.free_ids.is_empty()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4 + self.free_ids.len() * 4);
        bytes.extend_from_slice(&(self.free_ids.len() as u32).to_le_bytes());
        for &id in &self.free_ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Ok(Self::new());
        }

        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut free_ids = Vec::with_capacity(count);

        let mut offset = 4;
        for _ in 0..count {
            if offset + 4 > bytes.len() {
                break;
            }
            let id = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            free_ids.push(id);
            offset += 4;
        }

        Ok(Self { free_ids })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_sequential() {
        let mut list = FreeList::new();
        let mut next_id = 1;

        assert_eq!(list.allocate(&mut next_id), 1);
        assert_eq!(list.allocate(&mut next_id), 2);
        assert_eq!(list.allocate(&mut next_id), 3);
        assert_eq!(next_id, 4);
    }

    #[test]
    fn reuse_freed_ids() {
        let mut list = FreeList::new();
        let mut next_id = 10;

        list.free(5);
        list.free(3);

        // LIFO order
        assert_eq!(list.allocate(&mut next_id), 3);
        assert_eq!(list.allocate(&mut next_id), 5);
        assert_eq!(list.allocate(&mut next_id), 10);
    }

    #[test]
    fn serialization() {
        let mut list = FreeList::new();
        list.free(10);
        list.free(20);
        list.free(30);

        let bytes = list.to_bytes();
        let restored = FreeList::from_bytes(&bytes).unwrap();

        assert_eq!(list.free_ids, restored.free_ids);
    }
}
